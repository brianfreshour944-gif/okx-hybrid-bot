#!/usr/bin/env python3
# ALPACA HYBRID TRADING BOT (Crypto Spot)

import asyncio
import pandas as pd
import numpy as np
import logging
import os
import csv
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# CSV LOGGING (same as original)
# ==============================================================================

def write_trade(symbol, side, price, pnl_usd=None, total_pnl=None, score=None):
    with open('trades.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            symbol, side, price,
            pnl_usd if pnl_usd is not None else '',
            total_pnl if total_pnl is not None else '',
            score if score is not None else ''
        ])

if not os.path.exists('trades.csv'):
    with open('trades.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Symbol', 'Side', 'Price', 'PnL_USD', 'Total_PnL_USD', 'Score'])

# ==============================================================================
# SCORE CALCULATOR (unchanged from OKX version)
# ==============================================================================

class ScoreCalculator:
    def __init__(self):
        self.score_history = []
    
    def rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50
        deltas = np.diff(prices[-period-1:])
        gain = np.mean(deltas[deltas > 0]) if any(deltas > 0) else 0.001
        loss = -np.mean(deltas[deltas < 0]) if any(deltas < 0) else 0.001
        return 100 - (100 / (1 + (gain / loss)))
    
    def ema(self, prices, period):
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = prices[i] * alpha + ema[i-1] * (1 - alpha)
        return ema
    
    def compute(self, df):
        if df is None or len(df) < 50:
            return 0.5
        
        close = df['close'].values
        volume = df['volume'].values
        
        ma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        z_score = (close[-1] - ma20) / std20 if std20 > 0 else 0
        rsi_val = self.rsi(close)
        ema9 = self.ema(close, 9)
        ema21 = self.ema(close, 21)
        is_uptrend = ema9[-1] > ema21[-1] and close[-1] > ema9[-1]
        vol_avg = np.mean(volume[-10:]) if len(volume) >= 10 else 1
        vol_surge = volume[-1] / vol_avg if vol_avg > 0 else 1
        
        score = 0.5
        if z_score < -1.2:
            score += 0.35
        elif z_score < -0.8:
            score += 0.25
        elif z_score < -0.4:
            score += 0.15
        elif z_score > 1.2:
            score -= 0.35
        elif z_score > 0.8:
            score -= 0.25
        elif z_score > 0.4:
            score -= 0.15
        
        if rsi_val < 35:
            score += 0.10
        elif rsi_val < 45:
            score += 0.05
        elif rsi_val > 65:
            score -= 0.10
        elif rsi_val > 55:
            score -= 0.05
        
        if is_uptrend and score > 0.5:
            score += 0.08
        
        if vol_surge > 1.3:
            score += 0.05 if score > 0.5 else -0.05
        
        return max(0.0, min(1.0, score))

# ==============================================================================
# ALPACA TRADING BOT (spot only, paper trading)
# ==============================================================================

class AlpacaTradingBot:
    def __init__(self):
        self.buy_threshold = 0.51
        self.sell_threshold = 0.49
        self.position_size_usd = 10.0          # Each buy order uses this amount (in USD)
        self.symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD']   # Alpaca crypto spot pairs
        
        self.score_calc = ScoreCalculator()
        self.positions = {}          # stores entry price for each symbol
        self.total_pnl = 0.0
        
        # Alpaca API keys (paper trading)
        self.api_key = os.getenv("APCA_API_KEY_ID", "")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY", "")
        
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API keys missing. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY.")
        
        # Clients (trading is synchronous, we'll run it in threads)
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data_client = CryptoHistoricalDataClient()
        
        logger.info("Alpaca bot initialized (paper trading, crypto spot)")
    
    # --------------------------------------------------------------------------
    # Helper: normalise symbol for Alpaca (e.g., "BTC/USD" -> "BTCUSD")
    # --------------------------------------------------------------------------
    def _norm_symbol(self, symbol):
        return symbol.replace("/", "")
    
    # --------------------------------------------------------------------------
    # Fetch latest price + 100 bars (5m interval)
    # --------------------------------------------------------------------------
    async def fetch_data(self, symbol):
        try:
            # Use synchronous data client in a thread to avoid blocking
            loop = asyncio.get_running_loop()
            
            # Request 100 5-minute bars
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                limit=100
            )
            bars = await loop.run_in_executor(None, self.data_client.get_crypto_bars, request)
            
            if symbol not in bars.data:
                return None, None
            
            bars_list = bars.data[symbol]
            # Convert to DataFrame
            df = pd.DataFrame({
                'close': [b.close for b in bars_list],
                'volume': [b.volume for b in bars_list]
            })
            current_price = bars_list[-1].close
            return current_price, df
        except Exception as e:
            logger.error(f"Data fetch error for {symbol}: {e}")
            return None, None
    
    # --------------------------------------------------------------------------
    # Get current positions from Alpaca (to know if we hold the asset)
    # --------------------------------------------------------------------------
    async def get_positions_cache(self):
        try:
            loop = asyncio.get_running_loop()
            positions = await loop.run_in_executor(None, self.trading_client.get_all_positions)
            cache = {}
            for p in positions:
                # Alpaca returns symbol like "BTCUSD" for BTC/USD
                cache[p.symbol] = {
                    'qty': float(p.qty),
                    'avg_price': float(p.avg_entry_price)
                }
            return cache
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}
    
    # --------------------------------------------------------------------------
    # Place a market order (buy or sell)
    # --------------------------------------------------------------------------
    async def submit_order(self, symbol, side, usd_amount=None):
        try:
            loop = asyncio.get_running_loop()
            norm = self._norm_symbol(symbol)
            
            if side == 'buy':
                # Need to calculate quantity based on current price
                price, _ = await self.fetch_data(symbol)
                if price is None:
                    return False, 0, 0
                qty = round((usd_amount / price) - 0.0000005, 6)   # small safety subtraction
                if qty <= 0:
                    logger.error(f"Calculated qty <= 0 for {symbol} buy")
                    return False, 0, 0
                order = MarketOrderRequest(
                    symbol=norm,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC
                )
            else:  # sell
                positions = await self.get_positions_cache()
                if norm not in positions:
                    logger.error(f"No position to sell for {symbol}")
                    return False, 0, 0
                qty = positions[norm]['qty']
                qty = round(qty - 0.0000005, 6)
                if qty <= 0:
                    qty = positions[norm]['qty']
                order = MarketOrderRequest(
                    symbol=norm,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                )
            
            # Submit order (synchronous, wrapped in thread)
            await loop.run_in_executor(None, self.trading_client.submit_order, order)
            logger.info(f"Order placed: {side.upper()} {qty} {symbol}")
            
            # Get fill price (approx = current market price)
            fill_price, _ = await self.fetch_data(symbol)
            if fill_price is None:
                fill_price = price if side == 'buy' else 0
            
            return True, qty, fill_price
        
        except Exception as e:
            logger.error(f"Order failed for {symbol} {side}: {e}")
            return False, 0, 0
    
    # --------------------------------------------------------------------------
    # Main loop (polls every 5 minutes, same as original)
    # --------------------------------------------------------------------------
    async def run(self):
        logger.info("=" * 50)
        logger.info("PAPER TRADING MODE - ALPACA HYBRID BOT (Crypto Spot)")
        logger.info(f"Buy when score > {self.buy_threshold}")
        logger.info(f"Sell when score < {self.sell_threshold}")
        logger.info("=" * 50)
        
        while True:
            await asyncio.sleep(300)   # 5 minutes
            
            # Fetch data for all symbols concurrently
            tasks = [self.fetch_data(s) for s in self.symbols]
            results = await asyncio.gather(*tasks)
            
            # Also get current positions once per cycle
            positions_cache = await self.get_positions_cache()
            
            for i, symbol in enumerate(self.symbols):
                price, df = results[i]
                if price is None or df is None:
                    continue
                
                score = self.score_calc.compute(df)
                norm = self._norm_symbol(symbol)
                has_position = norm in positions_cache
                
                logger.info(f"{symbol} | ${price:.2f} | Score: {score:.3f} | Position: {has_position}")
                
                # --- SELL logic ---
                if has_position and score < self.sell_threshold:
                    logger.info(f"🔴 SELL signal for {symbol} @ ${price:.2f}")
                    entry_price = positions_cache[norm]['avg_price']
                    qty = positions_cache[norm]['qty']
                    pnl_usd = (price - entry_price) * qty
                    self.total_pnl += pnl_usd
                    
                    success, fill_qty, fill_price = await self.submit_order(symbol, 'sell')
                    if success:
                        write_trade(symbol, 'SELL', fill_price, pnl_usd, self.total_pnl, score)
                        if symbol in self.positions:
                            del self.positions[symbol]
                    else:
                        logger.error(f"Sell order failed for {symbol}")
                
                # --- BUY logic ---
                elif not has_position and score > self.buy_threshold:
                    logger.info(f"🟢 BUY signal for {symbol} @ ${price:.2f}")
                    success, fill_qty, fill_price = await self.submit_order(symbol, 'buy', self.position_size_usd)
                    if success:
                        write_trade(symbol, 'BUY', fill_price, score=score)
                        self.positions[symbol] = {'price': fill_price}
                    else:
                        logger.error(f"Buy order failed for {symbol}")
        
        # (loop never ends unless interrupted)

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    bot = AlpacaTradingBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
