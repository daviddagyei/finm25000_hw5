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


def run_backtest(history1: pd.DataFrame, history2: pd.DataFrame, risk_params: Dict, symbol1: str, symbol2: str) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """
    Cross-Asset Arbitrage Strategy Backtest
    
    Objective: Trade the spread between two correlated assets when it deviates beyond a threshold.
    
    Args:
        history1: DataFrame with columns ['timestamp', 'last_price'] for asset 1
        history2: DataFrame with columns ['timestamp', 'last_price'] for asset 2
        risk_params: Dict with keys:
            - 'starting_cash': float
            - 'position_size': int (shares to trade per asset)
            - 'threshold': float (spread threshold for entry/exit)
            - 'transaction_cost': float (cost per trade as fraction, e.g., 0.001 = 0.1%)
            - 'lookback_window': int (window for hedge ratio calculation)
        symbol1: str (first asset symbol)
        symbol2: str (second asset symbol)
        
    Returns:
        Tuple[signals_df, trades_list, metrics_dict]
    """
    
    # Extract parameters
    starting_cash = risk_params['starting_cash']
    position_size = risk_params['position_size']
    threshold = risk_params['threshold']
    transaction_cost = risk_params.get('transaction_cost', 0.001)  # Default 0.1%
    lookback_window = risk_params.get('lookback_window', 60)  # Default 60 periods
    
    # Align histories by timestamp
    df1 = history1.copy().set_index('timestamp')
    df2 = history2.copy().set_index('timestamp')
    
    # Create aligned DataFrame
    df = pd.DataFrame({
        "p1": df1["last_price"],
        "p2": df2["last_price"]
    }).dropna()
    
    if len(df) < lookback_window + 10:
        raise ValueError(f"Not enough aligned data points. Need at least {lookback_window + 10}, got {len(df)}")
    
    # Reset index to get timestamp back as column
    df = df.reset_index()
    
    # Calculate rolling hedge ratio (beta) and spread
    df['beta'] = np.nan
    df['spread'] = np.nan
    df['spread_mean'] = np.nan
    df['spread_std'] = np.nan
    df['signal'] = 0
    
    # Calculate rolling statistics
    for i in range(lookback_window, len(df)):
        # Get lookback window data
        window_data = df.iloc[i-lookback_window:i]
        
        # Calculate hedge ratio via linear regression
        if len(window_data) >= 2 and window_data['p2'].std() > 1e-8:
            beta = np.polyfit(window_data["p2"], window_data["p1"], 1)[0]
            df.loc[i, 'beta'] = beta
            
            # Compute current spread
            current_spread = df.loc[i, 'p1'] - beta * df.loc[i, 'p2']
            df.loc[i, 'spread'] = current_spread
            
            # Calculate spread statistics for normalization
            window_spreads = window_data['p1'] - beta * window_data['p2']
            spread_mean = window_spreads.mean()
            spread_std = window_spreads.std()
            
            df.loc[i, 'spread_mean'] = spread_mean
            df.loc[i, 'spread_std'] = spread_std
            
            # Generate signals based on normalized spread
            if spread_std > 1e-8:
                normalized_spread = (current_spread - spread_mean) / spread_std
                
                # Signal generation
                if normalized_spread > threshold:
                    # Spread too high: sell spread (sell asset1, buy asset2)
                    df.loc[i, 'signal'] = -1
                elif normalized_spread < -threshold:
                    # Spread too low: buy spread (buy asset1, sell asset2)
                    df.loc[i, 'signal'] = 1
                elif abs(normalized_spread) < threshold * 0.5:  # Exit when spread normalizes
                    df.loc[i, 'signal'] = 0
    
    # Clean signals - only signal on changes
    df['signal_change'] = df['signal'].diff().fillna(0)
    
    # Initialize trading system components
    oms1 = OrderManagementSystem()
    oms2 = OrderManagementSystem()
    book1 = LimitOrderBook(symbol1)
    book2 = LimitOrderBook(symbol2)
    oms1.matching_engine = book1
    oms2.matching_engine = book2
    tracker = PositionTracker(starting_cash)
    trades_list = []
    
    # Track positions separately for each asset
    position1 = 0  # Position in asset 1
    position2 = 0  # Position in asset 2
    previous_signal = 0
    total_transaction_costs = 0
    
    for idx, row in df.iterrows():
        current_signal = row['signal']
        timestamp = row['timestamp']
        price1 = row['p1']
        price2 = row['p2']
        
        # Skip if no signal change or if we have NaN values
        if (current_signal == previous_signal or 
            pd.isna(price1) or pd.isna(price2) or 
            pd.isna(row['spread']) or pd.isna(row['beta'])):
            continue
        
        # Determine trade actions
        trades_to_execute = []
        
        if current_signal == 1 and previous_signal != 1:
            # Buy spread: buy asset1, sell asset2
            # Close any existing opposite positions first
            if position1 < 0:
                trades_to_execute.append((symbol1, "buy", abs(position1), price1))
            if position2 > 0:
                trades_to_execute.append((symbol2, "sell", position2, price2))
            
            # Enter new position
            trades_to_execute.append((symbol1, "buy", position_size, price1))
            trades_to_execute.append((symbol2, "sell", position_size, price2))
            
        elif current_signal == -1 and previous_signal != -1:
            # Sell spread: sell asset1, buy asset2
            # Close any existing opposite positions first
            if position1 > 0:
                trades_to_execute.append((symbol1, "sell", position1, price1))
            if position2 < 0:
                trades_to_execute.append((symbol2, "buy", abs(position2), price2))
            
            # Enter new position
            trades_to_execute.append((symbol1, "sell", position_size, price1))
            trades_to_execute.append((symbol2, "buy", position_size, price2))
            
        elif current_signal == 0 and previous_signal != 0:
            # Exit all positions
            if position1 != 0:
                side1 = "sell" if position1 > 0 else "buy"
                trades_to_execute.append((symbol1, side1, abs(position1), price1))
            
            if position2 != 0:
                side2 = "sell" if position2 > 0 else "buy"
                trades_to_execute.append((symbol2, side2, abs(position2), price2))
        
        # Execute all trades
        for symbol, side, quantity, price in trades_to_execute:
            if quantity <= 0:
                continue
                
            # Check cash availability for buy orders
            if side == "buy":
                required_cash = quantity * price * (1 + transaction_cost)
                if required_cash > tracker.cash:
                    # Adjust quantity to available cash
                    max_quantity = int(tracker.cash / (price * (1 + transaction_cost)))
                    if max_quantity <= 0:
                        continue
                    quantity = max_quantity
            
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
            
            # Submit to appropriate OMS
            if symbol == symbol1:
                oms1.new_order(order)
            else:
                oms2.new_order(order)
            
            # Calculate transaction cost
            trade_cost = quantity * price * transaction_cost
            total_transaction_costs += trade_cost
            
            # Create execution report
            execution_report = {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "filled_qty": quantity,
                "price": price,
                "timestamp": timestamp
            }
            
            # Update position tracker
            tracker.update(execution_report)
            
            # Deduct transaction cost from cash
            tracker.cash -= trade_cost
            
            # Update position tracking
            if symbol == symbol1:
                if side == "buy":
                    position1 += quantity
                else:
                    position1 -= quantity
            else:  # symbol2
                if side == "buy":
                    position2 += quantity
                else:
                    position2 -= quantity
            
            # Record trade
            trade_record = {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "signal": current_signal,
                "spread": row['spread'],
                "beta": row['beta'],
                "transaction_cost": trade_cost,
                "leg": "asset1" if symbol == symbol1 else "asset2"
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
            'final_cash': starting_cash,
            'transaction_costs': 0.0,
            'num_pairs_traded': 0
        }
    else:
        # Compute equity curve including transaction costs
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
        final_value = equity_curve.iloc[-1] - total_transaction_costs  # Subtract remaining costs
        total_return = (final_value - starting_cash) / starting_cash if starting_cash != 0 else 0.0
        
        # Count pairs traded (trades come in pairs for arbitrage)
        num_pairs = len(trades_list) // 2
        
        metrics_dict = {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(trades_list),
            'final_cash': final_value,
            'transaction_costs': total_transaction_costs,
            'num_pairs_traded': num_pairs
        }
    
    return df, trades_list, metrics_dict


def run_backtest_single_interface(combined_history: pd.DataFrame, risk_params: Dict, symbol_pair: str) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """
    Wrapper function for single DataFrame interface compatibility with Streamlit
    
    Args:
        combined_history: DataFrame with columns ['timestamp', 'price1', 'price2'] or similar
        risk_params: Same as main run_backtest function
        symbol_pair: String like "AAPL-MSFT" or comma-separated
        
    Returns:
        Same as main run_backtest function
    """
    
    # Parse symbol pair
    if '-' in symbol_pair:
        symbol1, symbol2 = symbol_pair.split('-', 1)
    elif ',' in symbol_pair:
        symbol1, symbol2 = symbol_pair.split(',', 1)
    else:
        # Default fallback
        symbol1, symbol2 = "ASSET1", "ASSET2"
    
    # Check if we have the right columns
    price_cols = [col for col in combined_history.columns if 'price' in col.lower()]
    
    if len(price_cols) >= 2:
        # Use first two price columns
        hist1 = pd.DataFrame({
            'timestamp': combined_history['timestamp'],
            'last_price': combined_history[price_cols[0]]
        })
        hist2 = pd.DataFrame({
            'timestamp': combined_history['timestamp'], 
            'last_price': combined_history[price_cols[1]]
        })
    else:
        raise ValueError("Combined history must have at least 2 price columns for arbitrage strategy")
    
    return run_backtest(hist1, hist2, risk_params, symbol1, symbol2)


if __name__ == "__main__":
    # Test the strategy with sample correlated data
    
    print("🔄 Testing Cross-Asset Arbitrage Strategy")
    print("=" * 50)
    
    # Generate sample correlated assets
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    
    # Asset 1: Base asset with trend
    price1_base = 100
    returns1 = np.random.normal(0.0005, 0.02, len(dates))
    prices1 = [price1_base]
    for r in returns1[1:]:
        prices1.append(prices1[-1] * (1 + r))
    
    # Asset 2: Correlated with Asset 1 but with some noise
    correlation = 0.8
    noise_factor = 0.3
    
    prices2 = []
    price2_base = 50  # Different base price
    hedge_ratio = 0.5  # True relationship
    
    for i, p1 in enumerate(prices1):
        if i == 0:
            prices2.append(price2_base)
        else:
            # Correlated movement with some noise
            p1_change = (p1 - prices1[i-1]) / prices1[i-1]
            correlated_change = correlation * p1_change + (1 - correlation) * np.random.normal(0, 0.01)
            noise = np.random.normal(0, noise_factor * 0.01)
            new_price = prices2[-1] * (1 + correlated_change + noise)
            prices2.append(max(new_price, 1))  # Prevent negative prices
    
    # Create sample data
    history1 = pd.DataFrame({
        'timestamp': dates,
        'last_price': prices1
    })
    
    history2 = pd.DataFrame({
        'timestamp': dates,
        'last_price': prices2
    })
    
    # Test parameters
    test_params = {
        'starting_cash': 100000.0,
        'position_size': 100,
        'threshold': 2.0,  # 2 standard deviations
        'transaction_cost': 0.001,  # 0.1%
        'lookback_window': 30
    }
    
    try:
        # Run backtest
        signals, trades, metrics = run_backtest(history1, history2, test_params, 'ASSET1', 'ASSET2')
        
        print("Cross-Asset Arbitrage Strategy Test Results:")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Number of Trades: {metrics['num_trades']}")
        print(f"Number of Pairs Traded: {metrics['num_pairs_traded']}")
        print(f"Final Cash: ${metrics['final_cash']:,.2f}")
        print(f"Transaction Costs: ${metrics['transaction_costs']:,.2f}")
        
        if trades:
            print(f"\nFirst few trades:")
            for i, trade in enumerate(trades[:6]):
                print(f"  {i+1}: {trade}")
        
        # Show signal distribution
        signal_counts = signals['signal'].value_counts().sort_index()
        print(f"\nSignal distribution: {dict(signal_counts)}")
        
        # Show spread statistics
        valid_spreads = signals['spread'].dropna()
        if len(valid_spreads) > 0:
            print(f"\nSpread Statistics:")
            print(f"Average spread: {valid_spreads.mean():.4f}")
            print(f"Spread std dev: {valid_spreads.std():.4f}")
            print(f"Min spread: {valid_spreads.min():.4f}")
            print(f"Max spread: {valid_spreads.max():.4f}")
        
        # Show hedge ratio statistics
        valid_betas = signals['beta'].dropna()
        if len(valid_betas) > 0:
            print(f"\nHedge Ratio (Beta) Statistics:")
            print(f"Average beta: {valid_betas.mean():.4f}")
            print(f"Beta std dev: {valid_betas.std():.4f}")
        
    except Exception as e:
        print(f"❌ Error during backtesting: {e}")
        import traceback
        traceback.print_exc()
