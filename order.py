from dataclasses import dataclass
from datetime import datetime


@dataclass
class Order:
    """
    Represents a single trade instruction.
    """
    id: str                     # unique identifier (e.g. UUID or string)
    symbol: str                 # ticker or asset code (e.g. "AAPL", "EURUSD=X")
    side: str                   # "buy" or "sell"
    quantity: int               # must be > 0
    type: str                   # "market", "limit", or "stop"
    price: float = None         # limit/stop price, None for market orders
    timestamp: datetime = None  # when the order was created
