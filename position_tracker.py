import pandas as pd
from typing import List, Dict


class PositionTracker:
    """
    Tracks positions and cash, records trades, and computes P&L.
    """

    def __init__(self, starting_cash: float = 0.0):
        # Net position per symbol (symbol -> shares/contracts)
        self.positions: Dict[str, int] = {}
        # Cash balance
        self.cash: float = starting_cash
        # List of trade records (one dict per fill)
        self.blotter: List[Dict] = []

    def update(self, report: Dict) -> None:
        """
        Process one execution report with keys:
          - order_id
          - symbol
          - side ('buy' or 'sell')
          - filled_qty (int)
          - price (float)
          - timestamp (datetime)
        Updates positions, cash, and appends to the blotter list.
        """
        symbol    = report["symbol"]
        qty       = report["filled_qty"]
        price     = report["price"]
        side      = report["side"]
        timestamp = report["timestamp"]

        # 1) Update net position
        # A 'buy' increases your holding, 'sell' decreases it
        delta = qty if side == "buy" else -qty
        self.positions[symbol] = self.positions.get(symbol, 0) + delta

        # 2) Update cash
        # Cash flows negative for buys (cash out), positive for sells
        cash_flow = -qty * price if side == "buy" else qty * price
        self.cash += cash_flow

        # 3) Record in blotter
        # We'll compute realized P&L later in summary
        self.blotter.append({
            "timestamp":   timestamp,
            "symbol":      symbol,
            "side":        side,
            "quantity":    qty,
            "price":       price,
            "cash_flow":   cash_flow
        })

    def get_blotter(self) -> pd.DataFrame:
        """
        Return a DataFrame of all fills with columns:
          timestamp, symbol, side, quantity, price, cash_flow
        """
        return pd.DataFrame(self.blotter)

    def get_pnl_summary(self, current_prices: Dict[str, float] = None) -> Dict:
        """
        Returns a dict with:
          - 'realized_pnl': float
          - 'unrealized_pnl': float (0 if no current_prices given)
          - 'total_pnl': sum of realized + unrealized
          - 'current_cash': self.cash
          - 'positions': copy of self.positions dict
        """
        blotter_df = self.get_blotter()
        realized_pnl = float(blotter_df["cash_flow"].sum()) if not blotter_df.empty else 0.0

        unrealized_pnl = 0.0
        if current_prices:
            for sym, pos in self.positions.items():
                price = current_prices.get(sym, 0.0)
                unrealized_pnl += pos * price

        total_pnl = realized_pnl + unrealized_pnl

        return {
            "realized_pnl":    realized_pnl,
            "unrealized_pnl":  unrealized_pnl,
            "total_pnl":       total_pnl,
            "current_cash":    self.cash,
            "positions":       dict(self.positions)
        }
