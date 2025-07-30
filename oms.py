from order import Order
from datetime import datetime, timezone
from typing import Dict, Optional


class OrderManagementSystem:
    """
    Validates, tracks, and optionally routes orders.
    """
    def __init__(self, matching_engine=None):
        self._orders: Dict[str, Order]  = {}
        self._statuses: Dict[str, str]  = {}
        self.matching_engine = matching_engine

    def new_order(self, order: Order) -> dict:
        if order.side not in ("buy", "sell"):
            raise ValueError("Side must be 'buy' or 'sell'")
        if order.quantity <= 0:
            raise ValueError("Quantity must be > 0")
        if order.type not in ("market", "limit", "stop"):
            raise ValueError("Type must be 'market', 'limit', or 'stop'")
        if order.type in ("limit", "stop") and order.price is None:
            raise ValueError("Limit/stop orders require a price")

        now = datetime.now(timezone.utc)
        order.timestamp = order.timestamp or now

        self._orders[order.id]   = order
        self._statuses[order.id] = "accepted"

        if self.matching_engine:
            self.matching_engine.add_order(order)

        return {
            "order_id": order.id,
            "status":   "accepted",
            "timestamp": order.timestamp
        }

    def cancel_order(self, order_id: str) -> dict:
        if order_id not in self._orders:
            raise KeyError(f"Order {order_id} not found")
        current = self._statuses[order_id]
        if current in ("canceled", "filled"):
            raise ValueError(f"Cannot cancel order in status {current}")
        self._statuses[order_id] = "canceled"
        return {
            "order_id": order_id,
            "status":   "canceled",
            "timestamp": datetime.now(timezone.utc)
        }

    def amend_order(
        self,
        order_id:   str,
        new_qty:    Optional[int]   = None,
        new_price:  Optional[float] = None
    ) -> dict:
        if order_id not in self._orders:
            raise KeyError(f"Order {order_id} not found")
        if self._statuses[order_id] != "accepted":
            raise ValueError("Only accepted orders can be amended")

        order = self._orders[order_id]
        if new_qty is not None:
            if new_qty <= 0:
                raise ValueError("Quantity must be > 0")
            order.quantity = new_qty
        if new_price is not None:
            if order.type not in ("limit", "stop"):
                raise ValueError("Only limit/stop orders can change price")
            order.price = new_price

        order.timestamp = datetime.now(timezone.utc)
        return {
            "order_id": order_id,
            "status":   "amended",
            "timestamp": order.timestamp
        }
