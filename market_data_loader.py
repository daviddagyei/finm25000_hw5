import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Optional, Tuple, Dict, Union


class MarketDataLoader:
    """
    A market data loader for fetching OHLCV data for equities, ETFs, FX, crypto, bonds/futures,
    plus options chains, either by period or explicit start/end dates.
    """
    
    def __init__(self, interval: str = "1d", period: str = "1y"):
        """
        Initialize the MarketDataLoader.
        
        Args:
            interval: Data interval (e.g., '1d', '1h', '5m')
            period: Period for data fetching (e.g., '1y', '6mo', '1mo')
        """
        self.interval = interval
        self.period = period
        self._history_cache = {}  
        self._options_cache = {}
        self._asset_type_cache = {}  
    
    def _rename_and_tz(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to standard format and ensure UTC timezone-aware DatetimeIndex.
        
        Args:
            df: Raw DataFrame from yfinance
            
        Returns:
            DataFrame with standardized columns and UTC timezone
        """
        if df.empty:
            return df
            
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'last_price',
            'Adj Close': 'last_price',
            'Volume': 'volume'
        }
        
        df_renamed = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        required_cols = ['open', 'high', 'low', 'last_price', 'volume']
        for col in required_cols:
            if col not in df_renamed.columns:
                df_renamed[col] = np.nan
        
        if not df_renamed.index.tz:
            df_renamed.index = df_renamed.index.tz_localize('UTC')
        elif df_renamed.index.tz != pytz.UTC:
            df_renamed.index = df_renamed.index.tz_convert('UTC')
            
        return df_renamed
    
    def _load_period(self, symbol: str) -> pd.DataFrame:
        """
        Load historical data for a symbol using the configured period.
        
        Args:
            symbol: Symbol to fetch data for
            
        Returns:
            DataFrame with historical OHLCV data
        """
        cache_key = f"{symbol}_{self.period}_{self.interval}"
        
        if cache_key in self._history_cache:
            return self._history_cache[cache_key]
        
        try:
            df = yf.download(
                symbol, 
                period=self.period, 
                interval=self.interval, 
                auto_adjust=True,
                progress=False
            )
            
            df_processed = self._rename_and_tz(df)
            self._history_cache[cache_key] = df_processed
            return df_processed
            
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_history(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical data for a symbol, either by date range or configured period.
        
        Args:
            symbol: Symbol to fetch data for
            start: Start date (YYYY-MM-DD format) - optional
            end: End date (YYYY-MM-DD format) - optional
            
        Returns:
            DataFrame with historical OHLCV data
        """
        if start is not None and end is not None:
            cache_key = f"{symbol}_{start}_{end}_{self.interval}"
            
            if cache_key in self._history_cache:
                return self._history_cache[cache_key]
            
            try:
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    interval=self.interval,
                    auto_adjust=True,
                    progress=False
                )
                
                df_processed = self._rename_and_tz(df)
                self._history_cache[cache_key] = df_processed
                return df_processed
                
            except Exception as e:
                print(f"Error loading data for {symbol} from {start} to {end}: {e}")
                return pd.DataFrame()
        else:
            return self._load_period(symbol)
    
    def _locate_timestamp(self, df: pd.DataFrame, ts: Union[str, datetime, pd.Timestamp]) -> int:
        """
        Locate the index position for a given timestamp in the DataFrame.
        
        Args:
            df: DataFrame with DatetimeIndex
            ts: Timestamp to locate
            
        Returns:
            Index position (integer)
        """
        if df.empty:
            return -1
            
        if isinstance(ts, str):
            ts = pd.Timestamp(ts)
        elif isinstance(ts, datetime):
            ts = pd.Timestamp(ts)
            
        if ts.tz is None:
            ts = ts.tz_localize('UTC')
        elif ts.tz != df.index.tz:
            ts = ts.tz_convert(df.index.tz)
        
        if ts in df.index:
            return df.index.get_loc(ts)
        
        indexer = df.index.get_indexer([ts], method="ffill")
        return indexer[0] if indexer[0] != -1 else 0
    
    def _scalar_to_float(self, x) -> float:
        """
        Convert pandas scalar to Python float primitive.
        
        Args:
            x: Pandas scalar value or Series
            
        Returns:
            Python float
        """
        # Handle Series by taking the first element
        if hasattr(x, 'iloc') and len(x) > 0:
            x = x.iloc[0]
        elif hasattr(x, 'values') and len(x.values) > 0:
            x = x.values[0]
            
        if pd.isna(x):
            return float('nan')
        if hasattr(x, 'item'):
            return float(x.item())
        return float(x)
    
    def _scalar_to_int(self, x) -> int:
        """
        Convert pandas scalar to Python int primitive.
        
        Args:
            x: Pandas scalar value or Series
            
        Returns:
            Python int
        """
        # Handle Series by taking the first element
        if hasattr(x, 'iloc') and len(x) > 0:
            x = x.iloc[0]
        elif hasattr(x, 'values') and len(x.values) > 0:
            x = x.values[0]
            
        if pd.isna(x):
            return 0
        if hasattr(x, 'item'):
            return int(x.item())
        return int(x)
    
    def get_price(self, symbol: str, timestamp: Union[str, datetime, pd.Timestamp]) -> float:
        """
        Get the price for a symbol at a specific timestamp.
        
        Args:
            symbol: Symbol to get price for
            timestamp: Timestamp to get price at
            
        Returns:
            Price as float
        """
        df = self.get_history(symbol)
        if df.empty:
            return float('nan')
            
        idx = self._locate_timestamp(df, timestamp)
        if idx == -1:
            return float('nan')
            
        return self._scalar_to_float(df.iloc[idx]['last_price'])
    
    def _get_asset_type(self, symbol: str) -> str:
        """
        Get the asset type for a symbol using yfinance ticker info.
        
        Args:
            symbol: Symbol to classify
            
        Returns:
            Asset type string ('equity', 'etf', 'currency', 'crypto', 'index', 'commodity', 'bond', 'unknown')
        """
        cache_key = f"{symbol}_asset_type"
        
        if not hasattr(self, '_asset_type_cache'):
            self._asset_type_cache = {}
            
        if cache_key in self._asset_type_cache:
            return self._asset_type_cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            quote_type = info.get('quoteType', '').lower()
            
            if quote_type == 'equity':
                asset_type = 'equity'
            elif quote_type == 'etf':
                asset_type = 'etf'
            elif quote_type == 'currency':
                asset_type = 'currency'
            elif quote_type == 'cryptocurrency':
                asset_type = 'crypto'
            elif quote_type == 'index':
                asset_type = 'index'
            elif quote_type in ['commodity', 'future']:
                asset_type = 'commodity'
            elif quote_type == 'bond':
                asset_type = 'bond'
            else:
                symbol_upper = symbol.upper()
                
                if ('=' in symbol_upper or 
                    symbol_upper in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'] or
                    (len(symbol_upper) == 6 and 
                     symbol_upper[:3] in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD'] and
                     symbol_upper[3:] in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD'])):
                    asset_type = 'currency'
                
                elif ('BTC' in symbol_upper or 'ETH' in symbol_upper or 'DOGE' in symbol_upper or 
                      'ADA' in symbol_upper or 'DOT' in symbol_upper or 'LINK' in symbol_upper or
                      symbol_upper.endswith('-USD') or symbol_upper.endswith('-USDT')):
                    asset_type = 'crypto'
                
                elif (symbol_upper.startswith('^') or 
                      'SPX' in symbol_upper or 'NDX' in symbol_upper or 'DJI' in symbol_upper or 'RUT' in symbol_upper):
                    asset_type = 'index'
                
                elif (symbol_upper.endswith('.TO') or symbol_upper.endswith('.L') or 
                      symbol_upper.endswith('.PA') or symbol_upper.endswith('.DE') or
                      '.' in symbol_upper):
                    asset_type = 'equity'
                
                else:
                    asset_type = 'equity'
            
            self._asset_type_cache[cache_key] = asset_type
            return asset_type
            
        except Exception as e:
            print(f"Warning: Could not determine asset type for {symbol}: {e}")
            symbol_upper = symbol.upper()
            if '=' in symbol_upper or len(symbol_upper) == 6:
                return 'currency'
            elif 'BTC' in symbol_upper or 'ETH' in symbol_upper:
                return 'crypto'
            else:
                return 'equity'

    def _get_spread_for_asset_type(self, asset_type: str, symbol: str) -> float:
        """
        Get appropriate spread percentage based on asset type and specific symbol characteristics.
        
        Args:
            asset_type: The asset type classification
            symbol: The symbol for additional context
            
        Returns:
            Spread percentage as decimal (e.g., 0.001 = 0.1%)
        """
        symbol_upper = symbol.upper()
        
        if asset_type == 'currency':
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
            if any(pair in symbol_upper for pair in major_pairs):
                return 0.0001  
            else:
                return 0.0002  
        elif asset_type == 'crypto':
            if 'BTC' in symbol_upper or 'ETH' in symbol_upper:
                return 0.001   
            else:
                return 0.005   
                
        elif asset_type == 'etf':
            return 0.0002     
            
        elif asset_type == 'equity':
            if ('.' not in symbol_upper or 
                symbol_upper.endswith('.L') or symbol_upper.endswith('.TO')):
                return 0.001  
            else:
                return 0.0005
                
        elif asset_type == 'index':
            return 0.0001    
            
        elif asset_type == 'commodity':
            return 0.002     
            
        elif asset_type == 'bond':
            return 0.0005   
            
        else: 
            return 0.001     

    def get_bid_ask(self, symbol: str, timestamp: Union[str, datetime, pd.Timestamp]) -> Tuple[float, float]:
        """
        Get bid and ask prices for a symbol at a specific timestamp.
        Uses yfinance asset type classification for more accurate spread estimation.
        
        Args:
            symbol: Symbol to get bid/ask for
            timestamp: Timestamp to get bid/ask at
            
        Returns:
            Tuple of (bid, ask) prices
        """
        price = self.get_price(symbol, timestamp)
        if np.isnan(price):
            return (float('nan'), float('nan'))
        
        asset_type = self._get_asset_type(symbol)
        
        spread_pct = self._get_spread_for_asset_type(asset_type, symbol)
        
        half_spread = price * spread_pct / 2
        bid = price - half_spread
        ask = price + half_spread
        
        return (self._scalar_to_float(bid), self._scalar_to_float(ask))
    
    def get_volume(self, symbol: str, start: Union[str, datetime, pd.Timestamp], 
                   end: Union[str, datetime, pd.Timestamp]) -> int:
        """
        Get total volume for a symbol between start and end timestamps.
        
        Args:
            symbol: Symbol to get volume for
            start: Start timestamp
            end: End timestamp
            
        Returns:
            Total volume as int
        """
        df = self.get_history(symbol)
        if df.empty:
            return 0
        
        if isinstance(start, str):
            start = pd.Timestamp(start)
        elif isinstance(start, datetime):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)
        elif isinstance(end, datetime):
            end = pd.Timestamp(end)
            
        if start.tz is None:
            start = start.tz_localize('UTC')
        if end.tz is None:
            end = end.tz_localize('UTC')
            
        if start.tz != df.index.tz:
            start = start.tz_convert(df.index.tz)
        if end.tz != df.index.tz:
            end = end.tz_convert(df.index.tz)
        
        mask = (df.index >= start) & (df.index <= end)
        volume_sum = df.loc[mask, 'volume'].sum()
        
        return self._scalar_to_int(volume_sum)
    
    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Underlying symbol
            expiry: Specific expiry date (YYYY-MM-DD) - if None, uses nearest expiry
            
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames
        """
        cache_key = f"{symbol}_options_{expiry}"
        
        if cache_key in self._options_cache:
            return self._options_cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiry dates
            expiry_dates = ticker.options
            if not expiry_dates:
                return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}
            
            # Choose expiry date
            if expiry is None:
                # Use nearest expiry
                target_expiry = expiry_dates[0]
            else:
                # Find closest match to requested expiry
                expiry_dt = pd.Timestamp(expiry).date()
                available_dates = [pd.Timestamp(exp).date() for exp in expiry_dates]
                closest_idx = min(range(len(available_dates)), 
                                key=lambda i: abs((available_dates[i] - expiry_dt).days))
                target_expiry = expiry_dates[closest_idx]
            
            # Get options data
            options = ticker.option_chain(target_expiry)
            
            result = {
                "calls": options.calls if hasattr(options, 'calls') else pd.DataFrame(),
                "puts": options.puts if hasattr(options, 'puts') else pd.DataFrame()
            }
            
            self._options_cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"Error loading options for {symbol}: {e}")
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}