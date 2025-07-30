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
    Mean-Reversion Strategy Backtest using Bollinger Bands
    
    Objective: Use Bollinger Bands to buy low and sell high relative to a rolling mean.
    
    Args:
        history: DataFrame with columns ['timestamp', 'last_price']
        risk_params: Dict with keys:
            - 'starting_cash': float
            - 'position_size': int (shares to trade) or float (fraction of cash)
            - 'bollinger_win': int (Bollinger Bands window)
            - 'num_std': float (number of standard deviations for bands)
        symbol: str (asset symbol)
        
    Returns:
        Tuple[signals_df, trades_list, metrics_dict]
    """
    
    # Extract parameters
    starting_cash = risk_params['starting_cash']
    position_size = risk_params['position_size']
    bollinger_win = risk_params['bollinger_win']
    num_std = risk_params['num_std']
    
    # Create a copy of history to avoid modifying the original
    signals_df = history.copy()
    
    # Compute Bollinger Bands
    m = signals_df["last_price"].rolling(bollinger_win).mean()
    s = signals_df["last_price"].rolling(bollinger_win).std()
    signals_df["upper"] = m + num_std * s
    signals_df["lower"] = m - num_std * s
    signals_df["mid"] = m
    
    # Generate initial signals
    signals_df['signal'] = 0
    
    # Identify band crossings
    valid_indices = signals_df.dropna(subset=['upper', 'lower', 'mid']).index
    
    if len(valid_indices) > 1:
        for i in range(1, len(valid_indices)):
            curr_idx = valid_indices[i]
            prev_idx = valid_indices[i-1]
            
            curr_price = signals_df.loc[curr_idx, 'last_price']
            prev_price = signals_df.loc[prev_idx, 'last_price']
            
            curr_upper = signals_df.loc[curr_idx, 'upper']
            curr_lower = signals_df.loc[curr_idx, 'lower']
            curr_mid = signals_df.loc[curr_idx, 'mid']
            
            prev_upper = signals_df.loc[prev_idx, 'upper']
            prev_lower = signals_df.loc[prev_idx, 'lower']
            prev_mid = signals_df.loc[prev_idx, 'mid']
            
            # Check for crossovers
            # If price crosses below lower band: signal = +1 (enter long)
            if prev_price >= prev_lower and curr_price < curr_lower:
                signals_df.loc[curr_idx, 'signal'] = 1
            
            # If price crosses above upper band: signal = -1 (enter short)
            elif prev_price <= prev_upper and curr_price > curr_upper:
                signals_df.loc[curr_idx, 'signal'] = -1
            
            # If price crosses back to mid: signal = 0 (exit position)
            elif ((prev_price < prev_mid and curr_price >= curr_mid) or 
                  (prev_price > prev_mid and curr_price <= curr_mid)):
                signals_df.loc[curr_idx, 'signal'] = 0
    
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
            pd.isna(row['upper']) or 
            pd.isna(row['lower']) or
            pd.isna(row['mid'])):
            continue
            
        # Determine trade action
        side = None
        quantity = 0
        
        # Calculate position size (can be fixed shares or fraction of cash)
        if isinstance(position_size, float) and position_size <= 1.0:
            # Position size as fraction of available cash
            available_cash = tracker.cash
            target_value = available_cash * position_size
            calculated_size = max(1, int(target_value / current_price))
        else:
            # Position size as fixed number of shares
            calculated_size = int(position_size)
        
        if current_signal == 1 and previous_signal <= 0:
            # Buy signal: enter long position
            if current_position <= 0:
                side = "buy"
                quantity = calculated_size + abs(current_position)  # Cover short + go long
        elif current_signal == -1 and previous_signal >= 0:
            # Sell signal: enter short position (or exit long)
            if current_position > 0:
                side = "sell"
                quantity = current_position  # Close long position first
                # Note: For simplicity, we're not implementing actual shorting
                # In a real implementation, you'd need to handle short positions
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
            # Check if we have enough cash for buy orders
            if side == "buy":
                required_cash = quantity * current_price
                if required_cash > tracker.cash:
                    # Adjust quantity to available cash
                    quantity = max(1, int(tracker.cash / current_price))
                    if quantity * current_price > tracker.cash:
                        continue  # Skip if still not enough cash
            
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
                "signal": current_signal,
                "position_type": "long_entry" if current_signal == 1 else "exit" if current_signal == 0 else "long_exit"
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
    
    # Generate sample market data with mean-reverting characteristics
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    
    # Create mean-reverting price series
    base_price = 100
    prices = [base_price]
    mean_reversion_speed = 0.1
    volatility = 0.02
    
    for i in range(1, len(dates)):
        # Mean reversion: price tends to revert to base_price
        drift = mean_reversion_speed * (base_price - prices[-1])
        shock = np.random.normal(0, volatility)
        new_price = prices[-1] + drift + shock * prices[-1]
        prices.append(max(new_price, 10))  # Prevent negative prices
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'last_price': prices
    })
    
    # Test parameters
    test_params = {
        'starting_cash': 10000.0,
        'position_size': 100,
        'bollinger_win': 20,
        'num_std': 2.0
    }
    
    # Run backtest
    signals, trades, metrics = run_backtest(sample_data, test_params, 'TEST')
    
    print("Mean Reversion Strategy Test Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Final Cash: ${metrics['final_cash']:,.2f}")
    
    if trades:
        print(f"\nFirst few trades:")
        for i, trade in enumerate(trades[:5]):
            print(f"  {i+1}: {trade}")
    
    # Show signal distribution
    signal_counts = signals['signal'].value_counts().sort_index()
    print(f"\nSignal distribution: {dict(signal_counts)}")
    
    # Show band statistics
    print(f"\nBollinger Band Statistics:")
    print(f"Average band width: {(signals['upper'] - signals['lower']).mean():.2f}")
    print(f"Price touched lower band: {(signals['last_price'] <= signals['lower']).sum()} times")
    print(f"Price touched upper band: {(signals['last_price'] >= signals['upper']).sum()} times")
