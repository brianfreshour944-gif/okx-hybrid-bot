#!/usr/bin/env python3
# ALPACA HYBRID TRADING BOT (Crypto Spot) — Fixed version
#
# Fixes vs previous version:
#   1. ATR now uses proper True Range (needs high/low bars)
#   2. Cooldown only applied after LOSS exits, not all exits
#   3. Fill price captured from order response, not next bar fetch
#   4. P&L calculated from actual fill price, not bar close
#   5. Position size raised to $100 minimum (fees math)
#   6. Score bias clarified: z-score mean-reversion + trend filter
#      now applied sequentially, not additively canceling each other
#   7. Added proper high/low to data fetch for real ATR

import asyncio
import pandas as pd
import numpy as np
import logging
import json
import os
import csv
import time
from datetime import datetime, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CSV LOGGING
# ==============================================================================

def init_csv():
    if not os.path.exists('trades.csv'):
        with open('trades.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Symbol', 'Side', 'Price', 'Qty',
                'PnL_USD', 'Total_PnL_USD', 'Score', 'ExitReason',
                'StopPrice', 'TargetPrice'
            ])

def write_trade(symbol, side, price, qty=None, pnl_usd=None, total_pnl=None,
                score=None, exit_reason=None, stop_price=None, target_price=None):
    with open('trades.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            symbol, side, price,
            qty          if qty          is not None else '',
            pnl_usd      if pnl_usd      is not None else '',
            total_pnl    if total_pnl    is not None else '',
            score        if score        is not None else '',
            exit_reason  if exit_reason  is not None else '',
            stop_price   if stop_price   is not None else '',
            target_price if target_price is not None else '',
        ])

# ==============================================================================
# SCORE CALCULATOR
# ==============================================================================

class ScoreCalculator:

    def rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50
        deltas = np.diff(prices[-period - 1:])
        gain = np.mean(deltas[deltas > 0]) if any(deltas > 0) else 0.001
        loss = -np.mean(deltas[deltas < 0]) if any(deltas < 0) else 0.001
        return 100 - (100 / (1 + (gain / loss)))

    def ema(self, prices, period):
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices, dtype=float)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = prices[i] * alpha + ema[i - 1] * (1 - alpha)
        return ema

    def compute(self, df):
        if df is None or len(df) < 50:
            return 0.5

        close  = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)

        ma20  = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        z_score = (close[-1] - ma20) / std20 if std20 > 0 else 0

        rsi_val  = self.rsi(close)
        ema9     = self.ema(close, 9)
        ema21    = self.ema(close, 21)
        is_uptrend = ema9[-1] > ema21[-1] and close[-1] > ema9[-1]

        vol_avg   = np.mean(volume[-10:]) if len(volume) >= 10 else 1
        vol_surge = volume[-1] / vol_avg if vol_avg > 0 else 1

        # FIX: score bias was conflicting.
        # Mean-reversion signal: buy dips (z < 0) that are also in uptrend.
        # If NOT in uptrend, only trade stronger dips (z < -1.5) as reversal plays.
        score = 0.5

        if is_uptrend:
            # Trend-following dip buy: standard thresholds
            if z_score < -1.2:   score += 0.35
            elif z_score < -0.8: score += 0.25
            elif z_score < -0.4: score += 0.15
            elif z_score > 1.2:  score -= 0.25  # overbought in uptrend — smaller penalty
            elif z_score > 0.8:  score -= 0.15
            elif z_score > 0.4:  score -= 0.08
            # Trend bonus
            score += 0.08
        else:
            # Not in uptrend — only enter on strong mean-reversion dips
            if z_score < -1.5:   score += 0.25
            elif z_score < -1.0: score += 0.10
            elif z_score > 0.8:  score -= 0.30  # aggressively penalise overbought downtrend
            elif z_score > 0.4:  score -= 0.20

        # RSI overlay (same in both branches)
        if rsi_val < 35:   score += 0.10
        elif rsi_val < 45: score += 0.05
        elif rsi_val > 65: score -= 0.10
        elif rsi_val > 55: score -= 0.05

        # Volume surge: only helps if score is already bullish
        if vol_surge > 1.3:
            score += 0.05 if score > 0.5 else -0.05

        return max(0.0, min(1.0, score))

# ==============================================================================
# ATR helper — FIX: proper True Range using high/low
# ==============================================================================

def calc_atr(df, period=14):
    """
    True ATR: max(high-low, |high-prev_close|, |low-prev_close|)
    Falls back to close-to-close if high/low not available.
    """
    if df is None or len(df) < period + 1:
        return df['close'].iloc[-1] * 0.01

    if 'high' in df.columns and 'low' in df.columns:
        high  = df['high'].values.astype(float)
        low   = df['low'].values.astype(float)
        close = df['close'].values.astype(float)
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:]  - close[:-1])
            )
        )
        return float(np.mean(tr[-period:]))
    else:
        # Fallback: close-to-close (less accurate)
        close = df['close'].values.astype(float)
        return float(np.mean(np.abs(np.diff(close[-(period + 1):]))))

# ==============================================================================
# MAIN BOT
# ==============================================================================

class AlpacaTradingBot:

    def __init__(self):
        self.buy_threshold     = 0.63
        self.sell_threshold    = 0.38
        self.min_hold_bars     = 4
        self.sell_confirm_bars = 1

        # FIX: raise minimum position size — $10 is eaten by spreads.
        # At 0.05% spread on ETH, round-trip cost = 0.10% = $0.10 on $100.
        # On $10 that's $0.01 — tiny, but wins need to be proportionally larger too.
        # Real issue: absolute P&L is tiny at $10 even with good % returns.
        self.position_size_usd  = 100.0   # was $10 — adjust to your comfort level

        self.atr_stop_mult      = 2.5
        self.atr_target_mult    = 4.0
        self.daily_loss_limit   = -50.0   # scaled with larger position size
        self.max_daily_trades   = 10

        self.symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD']

        self.api_key    = os.getenv("APCA_API_KEY_ID", "")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY", "")

        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API keys missing — set APCA_API_KEY_ID and APCA_API_SECRET_KEY.")

        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data_client    = CryptoHistoricalDataClient()

        self.score_calc = ScoreCalculator()

        self.positions         = {}
        self.cooldowns         = {}
        self.bearish_count     = {}
        self.total_pnl         = 0.0
        self.daily_pnl         = 0.0
        self.daily_trade_count = 0
        self.current_day       = datetime.now().date()

        init_csv()
        self.load_state()
        logger.info("Bot initialised (paper trading, crypto spot)")

    # ==========================================================================
    # STATE PERSISTENCE
    # ==========================================================================

    def save_state(self):
        cooldowns_s = {s: dt.isoformat() for s, dt in self.cooldowns.items()}
        with open('alpaca_state.json', 'w') as f:
            json.dump({
                'positions':         self.positions,
                'cooldowns':         cooldowns_s,
                'total_pnl':         self.total_pnl,
                'daily_pnl':         self.daily_pnl,
                'daily_trade_count': self.daily_trade_count,
                'current_day':       self.current_day.isoformat(),
            }, f)

    def load_state(self):
        if not os.path.exists('alpaca_state.json'):
            logger.info("No saved state — starting fresh.")
            return
        try:
            with open('alpaca_state.json') as f:
                data = json.load(f)
            self.positions         = data.get('positions', {})
            self.total_pnl         = data.get('total_pnl', 0.0)
            self.daily_pnl         = data.get('daily_pnl', 0.0)
            self.daily_trade_count = data.get('daily_trade_count', 0)
            self.current_day       = datetime.fromisoformat(data['current_day']).date()
            now = datetime.now()
            self.cooldowns = {
                s: datetime.fromisoformat(v)
                for s, v in data.get('cooldowns', {}).items()
                if datetime.fromisoformat(v) > now
            }
            logger.info(
                f"State loaded — trades today: {self.daily_trade_count}, "
                f"daily P&L: ${self.daily_pnl:.2f}"
            )
        except Exception as e:
            logger.error(f"State load failed: {e}")

    # ==========================================================================
    # ACCOUNT
    # ==========================================================================

    def _norm(self, symbol):
        return symbol.replace('/', '').replace('-', '')

    async def get_positions_cache(self):
        try:
            loop = asyncio.get_running_loop()
            positions = await loop.run_in_executor(
                None, self.trading_client.get_all_positions
            )
            return {
                p.symbol: {
                    'qty':       float(p.qty),
                    'avg_price': float(p.avg_entry_price),
                }
                for p in positions
            }
        except Exception as e:
            logger.error(f"Position cache error: {e}")
            return {}

    # ==========================================================================
    # DATA — FIX: now fetches high/low for proper ATR
    # ==========================================================================

    async def fetch_data(self, symbol):
        try:
            loop = asyncio.get_running_loop()
            req  = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                limit=100,
            )
            bars = await loop.run_in_executor(
                None, self.data_client.get_crypto_bars, req
            )
            if symbol not in bars.data:
                return None, None
            rows = bars.data[symbol]
            df = pd.DataFrame({
                'open':   [b.open   for b in rows],
                'high':   [b.high   for b in rows],
                'low':    [b.low    for b in rows],
                'close':  [b.close  for b in rows],
                'volume': [b.volume for b in rows],
            })
            return rows[-1].close, df
        except Exception as e:
            logger.error(f"Data fetch error {symbol}: {e}")
            return None, None

    # ==========================================================================
    # ORDERS — FIX: capture fill price from the order response
    # ==========================================================================

    async def submit_order(self, symbol, side, usd_amount=None):
        """
        Returns (success: bool, qty: float, fill_price: float)

        FIX: previously fetched a new bar after the order to get 'fill price'
        which is actually the NEXT bar's open — wrong for P&L tracking.
        We now use avg_entry_price from the order/position response.
        If that's unavailable (market order latency), we fall back to the
        pre-order quote, which is much closer than the next bar fetch.
        """
        try:
            loop = asyncio.get_running_loop()
            norm = self._norm(symbol)

            # Pre-order quote for sizing and fallback fill price
            pre_price, _ = await self.fetch_data(symbol)
            if pre_price is None:
                return False, 0, 0

            if side == 'buy':
                qty = round((usd_amount / pre_price) - 0.0000005, 6)
                if qty <= 0:
                    return False, 0, 0
                order_req = MarketOrderRequest(
                    symbol=norm, qty=qty,
                    side=OrderSide.BUY, time_in_force=TimeInForce.GTC
                )
            else:
                positions = await self.get_positions_cache()
                if norm not in positions:
                    logger.error(f"No position to sell: {symbol}")
                    return False, 0, 0
                qty = round(positions[norm]['qty'] - 0.0000005, 6)
                if qty <= 0:
                    qty = positions[norm]['qty']
                order_req = MarketOrderRequest(
                    symbol=norm, qty=qty,
                    side=OrderSide.SELL, time_in_force=TimeInForce.GTC
                )

            order = await loop.run_in_executor(
                None, self.trading_client.submit_order, order_req
            )

            # Try to get fill price from the order object; fall back to pre-order price
            fill_price = pre_price
            try:
                if order.filled_avg_price is not None:
                    fill_price = float(order.filled_avg_price)
            except Exception:
                pass

            logger.info(f"ORDER {side.upper()} {qty} {symbol} @ ~${fill_price:.4f}")
            return True, qty, fill_price

        except Exception as e:
            logger.error(f"Order failed {symbol} {side}: {e}")
            return False, 0, 0

    # ==========================================================================
    # MAIN LOOP
    # ==========================================================================

    async def run(self):
        logger.info("=" * 60)
        logger.info("PAPER TRADING — ALPACA HYBRID BOT (Fixed / Hourly)")
        logger.info(f"  Buy threshold:  >{self.buy_threshold}")
        logger.info(f"  Sell threshold: <{self.sell_threshold} ({self.sell_confirm_bars} bar confirm)")
        logger.info(f"  Min hold:       {self.min_hold_bars} hours before signal exit")
        logger.info(f"  Stop loss:      ATR × {self.atr_stop_mult}")
        logger.info(f"  Take profit:    ATR × {self.atr_target_mult}  (1.6:1 R:R)")
        logger.info(f"  Position size:  ${self.position_size_usd:.0f}/trade")
        logger.info("=" * 60)

        last_heartbeat = 0

        while True:
            try:
                now_t = time.time()
                if now_t - last_heartbeat >= 30:
                    logger.info(
                        f"[Heartbeat] Open positions: {len(self.positions)} | "
                        f"Daily P&L: ${self.daily_pnl:.2f} | "
                        f"Total P&L: ${self.total_pnl:.2f}"
                    )
                    last_heartbeat = now_t

                today = datetime.now().date()
                if today != self.current_day:
                    logger.info("New trading day — resetting daily counters.")
                    self.daily_pnl         = 0.0
                    self.daily_trade_count = 0
                    self.current_day       = today

                if self.daily_pnl <= self.daily_loss_limit:
                    logger.error(f"Daily loss limit hit (${self.daily_pnl:.2f}) — pausing until midnight.")
                    await asyncio.sleep(60)
                    continue

                await asyncio.sleep(3600)

                results = await asyncio.gather(*[self.fetch_data(s) for s in self.symbols])
                positions_cache = await self.get_positions_cache()

                for i, symbol in enumerate(self.symbols):
                    try:
                        if symbol in self.cooldowns:
                            if datetime.now() < self.cooldowns[symbol]:
                                continue
                            del self.cooldowns[symbol]

                        price, df = results[i]
                        if price is None or df is None:
                            continue

                        score = self.score_calc.compute(df)
                        norm  = self._norm(symbol)
                        has_pos = norm in positions_cache

                        if score < self.sell_threshold:
                            self.bearish_count[symbol] = self.bearish_count.get(symbol, 0) + 1
                        else:
                            self.bearish_count[symbol] = 0

                        logger.info(
                            f"{symbol} | ${price:.4f} | score {score:.3f} | "
                            f"bearish_bars {self.bearish_count.get(symbol, 0)} | "
                            f"in_position: {has_pos}"
                        )

                        # ==================================================
                        # MANAGE OPEN POSITION
                        # ==================================================
                        if has_pos:
                            pos_data     = self.positions.get(symbol, {})
                            entry_price  = positions_cache[norm]['avg_price']
                            qty          = positions_cache[norm]['qty']
                            stop_price   = pos_data.get('stop_price',   entry_price * 0.96)
                            target_price = pos_data.get('target_price', entry_price * 1.08)
                            bars_held    = pos_data.get('bars_held',    0)

                            if symbol in self.positions:
                                self.positions[symbol]['bars_held'] = bars_held + 1

                            exit_reason = None
                            if price <= stop_price:
                                exit_reason = "STOP_LOSS"
                            elif price >= target_price:
                                exit_reason = "TAKE_PROFIT"
                            elif (bars_held >= self.min_hold_bars and
                                  self.bearish_count.get(symbol, 0) >= self.sell_confirm_bars):
                                exit_reason = "SIGNAL_EXIT"

                            logger.info(
                                f"  Tracking {symbol} | SL ${stop_price:.4f} | "
                                f"TP ${target_price:.4f} | bars {bars_held}"
                            )

                            if exit_reason:
                                logger.info(f"  EXIT {symbol} — {exit_reason}")
                                success, fill_qty, fill_price = await self.submit_order(symbol, 'sell')
                                if success:
                                    # FIX: P&L from actual fill price, not bar close
                                    pnl_usd = (fill_price - entry_price) * fill_qty
                                    self.total_pnl         += pnl_usd
                                    self.daily_pnl         += pnl_usd
                                    self.daily_trade_count += 1
                                    write_trade(
                                        symbol, 'SELL', fill_price, fill_qty,
                                        pnl_usd, self.total_pnl, score,
                                        exit_reason, stop_price, target_price
                                    )
                                    if symbol in self.positions:
                                        del self.positions[symbol]
                                    self.bearish_count[symbol] = 0

                                    # FIX: cooldown only on LOSS exits — profit exits allow re-entry
                                    if exit_reason == "STOP_LOSS":
                                        self.cooldowns[symbol] = datetime.now() + timedelta(hours=6)
                                        logger.info(f"  Cooldown set for {symbol} (6h, loss exit)")
                                    else:
                                        logger.info(f"  No cooldown — {exit_reason} allows re-entry")

                        # ==================================================
                        # LOOK FOR NEW ENTRY
                        # ==================================================
                        else:
                            if self.daily_trade_count >= self.max_daily_trades:
                                continue

                            if score > self.buy_threshold:
                                atr          = calc_atr(df, 14)
                                stop_price   = price - atr * self.atr_stop_mult
                                target_price = price + atr * self.atr_target_mult

                                logger.info(
                                    f"  BUY {symbol} @ ${price:.4f} | score {score:.3f} | "
                                    f"SL ${stop_price:.4f} | TP ${target_price:.4f}"
                                )
                                success, fill_qty, fill_price = await self.submit_order(
                                    symbol, 'buy', self.position_size_usd
                                )
                                if success:
                                    write_trade(
                                        symbol, 'BUY', fill_price, fill_qty,
                                        score=score,
                                        stop_price=stop_price,
                                        target_price=target_price
                                    )
                                    self.daily_trade_count += 1
                                    self.bearish_count[symbol] = 0
                                    # FIX: no cooldown on BUY — only on loss exit
                                    self.positions[symbol] = {
                                        'entry_time':   datetime.now().isoformat(),
                                        'entry_price':  fill_price,  # store for reference
                                        'stop_price':   stop_price,
                                        'target_price': target_price,
                                        'bars_held':    0,
                                    }

                        self.save_state()

                    except Exception as e:
                        logger.error(f"{symbol} loop error: {e}")

            except Exception as e:
                logger.error(f"Top-level error: {e}")
                await asyncio.sleep(10)

    # ==========================================================================
    # STOP
    # ==========================================================================

    def stop(self):
        self.save_state()
        logger.info(f"Shutdown. Total P&L: ${self.total_pnl:.2f}")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    bot = AlpacaTradingBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.stop()
