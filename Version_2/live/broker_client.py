"""
live/broker_client.py

Broker integration, Alpaca paper trading first. The interface (BrokerClient
ABC) is deliberately broker-and-mode-agnostic: submit_order / cancel_order /
get_position / get_all_positions / get_account_equity don't know or care
whether they're talking to a paper or a real account -- that's purely a
choice of which API keys AlpacaBrokerClient is constructed with. Going from
paper to live trading is a config change (paper=False + live keys), not a
rewrite of anything that calls this module.

Safety note: paper and live credentials are read from DIFFERENT env var
names on purpose (TRADING_ALPACA_PAPER_* vs TRADING_ALPACA_LIVE_*), so a
stale env var or a copy-paste mistake can't silently point a "paper" run at
a real brokerage account. AlpacaBrokerClient refuses to construct in live
mode unless the live env vars are explicitly set -- there is no fallback to
paper credentials.

Caveat: this is written against alpaca-py's documented request/response
shapes, not verified against a live install in this environment (no network
access to Alpaca's API here). Smoke-test submit_order/get_position against
the actual paper endpoint before trusting this against real capital.
"""

import enum
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce as AlpacaTimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"


class BrokerAPIError(Exception):
    """
    Raised after _with_retries() exhausts its attempts on a broker API call.
    live/reconciliation.py is expected to catch this and feed it to
    risk/kill_switch.py's KillSwitch.record_broker_error() -- that's the
    ONLY place a broker connectivity failure should turn into a halt
    decision, not scattered try/excepts elsewhere.
    """


class BrokerOrderRejected(Exception):
    """
    Raised when the broker definitively REJECTS an order (a 4xx response:
    invalid qty, insufficient buying power, unknown symbol, bad limit
    price, ...). Distinct from BrokerAPIError on purpose:

      - BrokerAPIError  -> transport/5xx trouble; retryable, and feeds
                           the kill switch's broker-error streak.
      - BrokerOrderRejected -> a business-level "no" from the broker;
                           retrying cannot make it succeed and would just
                           burn API calls (or, worse, double-submit if the
                           first attempt actually landed server-side).

    Callers should log the rejection and skip the order for that cycle --
    NOT feed it to the broker-error-streak counter, since the connection
    is fine and the rejection carries real information.
    """


@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    time_in_force: str = "day"
    client_order_id: Optional[str] = None


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    status: str
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None


@dataclass
class PositionInfo:
    symbol: str
    qty: float          # signed: positive long, negative short
    avg_entry_price: float
    market_value: float
    unrealized_pl: float


class BrokerClient:
    """Abstract broker interface. Every concrete broker (Alpaca now, anything else later) implements this."""

    def submit_order(self, request: OrderRequest) -> OrderResult:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        raise NotImplementedError

    def get_all_positions(self) -> Dict[str, PositionInfo]:
        raise NotImplementedError

    def get_account_equity(self) -> float:
        raise NotImplementedError

    def is_connected(self) -> bool:
        raise NotImplementedError


def _with_retries(fn, max_retries: int = 3, base_delay: float = 0.5, retry_exceptions=(APIError, ConnectionError, TimeoutError)):
    """
    Transport-level resilience only (a flaky connection, a 5xx, a timeout) --
    exponential backoff, then give up and raise BrokerAPIError. This is NOT
    where "the broker is telling us something meaningful is wrong" gets
    decided; that's reconciliation.py's job, one layer up.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn()
        except retry_exceptions as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise BrokerAPIError(f"broker API call failed after {max_retries} attempts: {last_exc}") from last_exc


class AlpacaBrokerClient(BrokerClient):
    def __init__(self, paper: bool = True, max_retries: int = 3):
        self.paper = paper
        self.max_retries = max_retries

        if paper:
            api_key = os.environ.get("TRADING_ALPACA_PAPER_KEY")
            api_secret = os.environ.get("TRADING_ALPACA_PAPER_SECRET")
            if not api_key or not api_secret:
                raise RuntimeError(
                    "paper=True but TRADING_ALPACA_PAPER_KEY / TRADING_ALPACA_PAPER_SECRET "
                    "are not set -- refusing to fall back to any other credentials."
                )
        else:
            api_key = os.environ.get("TRADING_ALPACA_LIVE_KEY")
            api_secret = os.environ.get("TRADING_ALPACA_LIVE_SECRET")
            if not api_key or not api_secret:
                raise RuntimeError(
                    "paper=False (LIVE TRADING) but TRADING_ALPACA_LIVE_KEY / "
                    "TRADING_ALPACA_LIVE_SECRET are not set -- refusing to silently fall "
                    "back to paper credentials for a live-mode client."
                )

        self._client = TradingClient(api_key, api_secret, paper=paper)

    def submit_order(self, request: OrderRequest) -> OrderResult:
        side = AlpacaOrderSide.BUY if request.side == OrderSide.BUY else AlpacaOrderSide.SELL
        tif = AlpacaTimeInForce.DAY

        # Same client_order_id on every retry attempt: if attempt N actually
        # landed server-side but the response was lost, the retry is
        # deduplicated by the broker instead of placing a second order.
        client_order_id = request.client_order_id or f"ad-{uuid.uuid4().hex}"

        if request.order_type == OrderType.MARKET:
            order_req = MarketOrderRequest(
                symbol=request.symbol, qty=request.qty, side=side,
                time_in_force=tif, client_order_id=client_order_id,
            )
        else:
            if request.limit_price is None:
                raise ValueError("limit_price is required for a LIMIT order")
            order_req = LimitOrderRequest(
                symbol=request.symbol, qty=request.qty, side=side,
                time_in_force=tif, limit_price=request.limit_price,
                client_order_id=client_order_id,
            )

        def _submit():
            try:
                return self._client.submit_order(order_req)
            except APIError as e:
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500:
                    # Business-level rejection (bad qty, no buying power, ...).
                    # NOT retryable -- see BrokerOrderRejected's docstring.
                    raise BrokerOrderRejected(
                        f"order rejected by broker (status {status}): {e}"
                    ) from e
                raise

        order = _with_retries(_submit, max_retries=self.max_retries)
        return _to_order_result(order)

    def cancel_order(self, order_id: str) -> bool:
        def _cancel():
            self._client.cancel_order_by_id(order_id)
            return True

        try:
            return _with_retries(_cancel, max_retries=self.max_retries)
        except BrokerAPIError:
            return False

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        def _get():
            try:
                return self._client.get_open_position(symbol)
            except APIError as e:
                # Alpaca raises an APIError (not a distinct exception type)
                # for "no position exists" -- treat that as None, not a
                # retryable failure.
                if "position does not exist" in str(e).lower():
                    return None
                raise

        pos = _with_retries(_get, max_retries=self.max_retries)
        return _to_position_info(pos) if pos is not None else None

    def get_all_positions(self) -> Dict[str, PositionInfo]:
        positions = _with_retries(lambda: self._client.get_all_positions(), max_retries=self.max_retries)
        return {p.symbol: _to_position_info(p) for p in positions}

    def get_account_equity(self) -> float:
        account = _with_retries(lambda: self._client.get_account(), max_retries=self.max_retries)
        return float(account.equity)

    def is_connected(self) -> bool:
        try:
            self._client.get_account()
            return True
        except Exception:
            return False


def _to_order_result(order) -> OrderResult:
    return OrderResult(
        order_id=str(order.id),
        symbol=order.symbol,
        side=OrderSide.BUY if str(order.side).lower().endswith("buy") else OrderSide.SELL,
        qty=float(order.qty) if order.qty is not None else 0.0,
        status=str(order.status),
        filled_qty=float(order.filled_qty) if order.filled_qty is not None else 0.0,
        filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price is not None else None,
    )


def _to_position_info(pos) -> PositionInfo:
    # Some Alpaca SDK versions report qty as unsigned + a separate `side`
    # field; guard both conventions rather than assuming one.
    qty = float(pos.qty)
    side = getattr(pos, "side", None)
    if side is not None and str(side).lower().endswith("short") and qty > 0:
        qty = -qty
    return PositionInfo(
        symbol=pos.symbol,
        qty=qty,
        avg_entry_price=float(pos.avg_entry_price),
        market_value=float(pos.market_value),
        unrealized_pl=float(pos.unrealized_pl),
    )