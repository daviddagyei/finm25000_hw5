import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import uuid
from datetime import datetime, timezone

# Import trading system components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from order import Order
from oms import OrderManagementSystem
from order_book import LimitOrderBook
from position_tracker import PositionTracker


def run_backtest(history: pd.DataFrame, risk_params: Dict, symbol: str) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """
    Trend-Following Strategy Backtest
    
    Objective: Enter long when a short-term MA crosses above a long-term MA; 
    exit (flat) on the reverse cross.
    
    Args:
        history: DataFrame with columns ['timestamp', 'last_price']
        risk_params: Dict with keys:
            - 'starting_cash': float
            - 'position_size': int (shares to trade)
            - 'short_win': int (short moving average window)
            - 'long_win': int (long moving average window)
        symbol: str (asset symbol)
        
    Returns:
        Tuple[signals_df, trades_list, metrics_dict]
    """
    
    # Extract parameters
    starting_cash = risk_params['starting_cash']
    position_size = risk_params['position_size']
    short_win = risk_params['short_win']
    long_win = risk_params['long_win']
    
    # Create a copy of history to avoid modifying the original
    signals_df = history.copy()
    
    # Compute moving averages
    signals_df["ma_short"] = signals_df["last_price"].rolling(short_win).mean()
    signals_df["ma_long"] = signals_df["last_price"].rolling(long_win).mean()
    
    # Generate initial signals
    # When ma_short crosses above ma_long: signal = +1
    # When ma_short crosses below ma_long: signal = -1
    # Else: signal = 0
    signals_df['signal'] = 0
    
    # Identify crossovers using shift to detect changes
    # Only signal on actual crossover bars
    # First, remove rows with NaN values for the comparison
    valid_indices = signals_df.dropna(subset=['ma_short', 'ma_long']).index
    
    if len(valid_indices) > 1:
        for i in range(1, len(valid_indices)):
            curr_idx = valid_indices[i]
            prev_idx = valid_indices[i-1]
            
            curr_short = signals_df.loc[curr_idx, 'ma_short']
            curr_long = signals_df.loc[curr_idx, 'ma_long']
            prev_short = signals_df.loc[prev_idx, 'ma_short']
            prev_long = signals_df.loc[prev_idx, 'ma_long']
            
            # Check for crossovers
            curr_above = curr_short > curr_long
            prev_above = prev_short > prev_long
            
            if curr_above and not prev_above:
                # Short MA crosses above Long MA
                signals_df.loc[curr_idx, 'signal'] = 1
            elif not curr_above and prev_above:
                # Short MA crosses below Long MA  
                signals_df.loc[curr_idx, 'signal'] = -1
    
    # Clean signals to only trade on crossover bars
    # Forward fill the signal to maintain position until next crossover
    signals_df['position'] = signals_df['signal'].replace(0, np.nan).ffill().fillna(0)
    
    # Initialize trading system components
    oms = OrderManagementSystem()
    book = LimitOrderBook(symbol)
    oms.matching_engine = book  # Connect OMS to order book
    tracker = PositionTracker(starting_cash)
    trades_list = []
    
    # Backtest loop
    previous_signal = 0
    current_position = 0
    
    for idx, row in signals_df.iterrows():
        current_signal = row['signal']
        current_price = row['last_price']
        timestamp = row['timestamp']
        
        # Skip if signal is the same as previous or if we have NaN values
        if (current_signal == previous_signal or 
            pd.isna(current_price) or 
            pd.isna(row['ma_short']) or 
            pd.isna(row['ma_long'])):
            continue
            
        # Determine trade action
        side = None
        quantity = 0
        
        if current_signal == 1 and previous_signal <= 0:
            # Buy signal: enter long position
            if current_position <= 0:
                side = "buy"
                quantity = position_size + abs(current_position)  # Cover short + go long
        elif current_signal == -1 and previous_signal >= 0:
            # Sell signal: exit long position (go flat)
            if current_position > 0:
                side = "sell"
                quantity = current_position  # Close long position
        elif current_signal == 0:
            # Exit signal: close any open position
            if current_position > 0:
                side = "sell"
                quantity = current_position
            elif current_position < 0:
                side = "buy"
                quantity = abs(current_position)
        
        # Execute trade if there's an action
        if side and quantity > 0:
            # Create and submit order
            order_id = str(uuid.uuid4())
            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                type="market",
                timestamp=timestamp
            )
            
            # Submit order to OMS
            oms_response = oms.new_order(order)
            
            # The order book will execute the market order immediately
            # For simulation, we'll assume the order fills at the current price
            execution_report = {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "filled_qty": quantity,
                "price": current_price,
                "timestamp": timestamp
            }
            
            # Update position tracker
            tracker.update(execution_report)
            
            # Update current position
            if side == "buy":
                current_position += quantity
            else:  # sell
                current_position -= quantity
            
            # Record trade
            trade_record = {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": current_price,
                "signal": current_signal
            }
            trades_list.append(trade_record)
        
        previous_signal = current_signal
    
    # Calculate metrics
    blotter = tracker.get_blotter()
    
    if len(blotter) == 0:
        # No trades executed
        metrics_dict = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'num_trades': 0,
            'final_cash': starting_cash
        }
    else:
        # Compute equity curve
        blotter['cumulative_cash_flow'] = blotter['cash_flow'].cumsum()
        equity_curve = starting_cash + blotter['cumulative_cash_flow']
        
        # Calculate returns
        returns = equity_curve.diff().fillna(0)
        
        # Calculate Sharpe ratio (annualized)
        if returns.std() != 0:
            sharpe = returns.mean() / returns.std() * (252**0.5)
        else:
            sharpe = 0.0
        
        # Calculate maximum drawdown
        running_max = equity_curve.cummax()
        drawdown = equity_curve - running_max
        max_dd = drawdown.min() / starting_cash if starting_cash != 0 else 0.0
        
        # Calculate total return
        final_value = equity_curve.iloc[-1]
        total_return = (final_value - starting_cash) / starting_cash if starting_cash != 0 else 0.0
        
        metrics_dict = {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(trades_list),
            'final_cash': final_value
        }
    
    return signals_df, trades_list, metrics_dict


if __name__ == "__main__":
    # Test the strategy with sample data
    
    # Generate sample market data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'last_price': prices
    })
    
    # Test parameters
    test_params = {
        'starting_cash': 10000.0,
        'position_size': 100,
        'short_win': 5,
        'long_win': 20
    }
    
    # Run backtest
    signals, trades, metrics = run_backtest(sample_data, test_params, 'TEST')
    
    print("Trend Following Strategy Test Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Final Cash: ${metrics['final_cash']:,.2f}")
    
    if trades:
        print(f"\nFirst few trades:")
        for i, trade in enumerate(trades[:3]):
            print(f"  {i+1}: {trade}")
