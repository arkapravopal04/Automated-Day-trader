"""
this is for the live training



upstox_live_runner.py — loads your trained PPO agent and runs it on
live 5m NSE candles via Upstox paper trading.

Run:
    python upstox_live_runner.py

Stop with Ctrl-C — prints a full summary on exit.

Before running:
    1. Paste your daily access token in upstox_data.py → ACCESS_TOKEN
    2. Make sure your model is trained: models/best_model.pkl must exist
    3. pip install upstox-python-sdk
"""

import os
import sys
import time
import numpy as np

from engine import Tensor
from Neural_Nets import (LSTM, Conv2D, Flatten, MultiHeadAttention,
                         FusionLayers, RegimeDetector)
from agent import PPOAgent
from models_utils import load_model
from risk import RiskManager
from live_paper import UpstoxPaperBroker
from live_data import LiveCandleStream, load_data, transform_data, build_windows

# ── config ────────────────────────────────────────────────────────────────────
BASE_PATH       = "/kaggle/working" if os.path.exists("/kaggle") else "."
BEST_MODEL_PATH = f"{BASE_PATH}/models/best_model.pkl"
CHECKPOINT_PATH = f"{BASE_PATH}/models/checkpoint.pkl"
TRADE_LOG_PATH  = f"{BASE_PATH}/logs/upstox_paper_trades.csv"

# ticker to trade live — must exist in INSTRUMENT_KEYS in upstox_data.py
TICKER          = "RELIANCE"

# paper balance in ₹ — use something realistic for position sizing
INITIAL_BALANCE = 50000.0

WINDOW_SIZE     = 20     # must match training
POLL_SECONDS    = 5      # how often to check for a new completed bar

CNN_FLAT_SIZE   = 288
FUSED_STATE_SIZE= 75

# NSE market hours (IST) — agent only runs during these hours
MARKET_OPEN_H   = 9
MARKET_OPEN_M   = 15
MARKET_CLOSE_H  = 15
MARKET_CLOSE_M  = 25   # stop 5m before close to avoid end-of-day noise

GREEN  = "\033[92m"
RED    = "\033[91m"
RESET  = "\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# build agent — identical architecture to train.py
# ─────────────────────────────────────────────────────────────────────────────

def build_agent() -> PPOAgent:
    lstm      = LSTM(input_size=5, hidden_size=64, num_layers=2)
    attention = MultiHeadAttention(hidden_size=64, num_heads=4)
    cnn       = Conv2D(in_channels=1, out_channels=16, kernel_size=(3, 5))
    flatten   = Flatten()
    regime    = RegimeDetector(input_size=5, hidden_size=32)
    fusion    = FusionLayers(
        lstm_hidden_size=64,
        cnn_out_channels=CNN_FLAT_SIZE,
        nlp_hidden_size=64,
        hidden_size=64,
        risk_size=8,
    )
    return PPOAgent(
        state_size=FUSED_STATE_SIZE, action_size=2,
        lstm=lstm, attention=attention, cnn=cnn,
        flatten=flatten, regime=regime, fusion=fusion,
    )


# ─────────────────────────────────────────────────────────────────────────────
# state extraction — mirrors env._get_state()
# ─────────────────────────────────────────────────────────────────────────────

def extract_state(window: np.ndarray,
                  broker: UpstoxPaperBroker,
                  risk_manager: RiskManager,
                  agent: PPOAgent,
                  current_price: float,
                  nlp_vec: Tensor) -> Tensor:

    sample = Tensor(window.astype(np.float64))

    hidden_states, _ = agent.lstm(sample)
    lstm_out          = agent.attention(hidden_states)

    cnn_input = Tensor(window.reshape(1, WINDOW_SIZE, 5).astype(np.float64))
    cnn_raw   = agent.cnn(cnn_input)
    cnn_out   = agent.flatten(cnn_raw)

    regime_out    = agent.regime(sample)
    risk_features = risk_manager.get_risk_features()
    risk_tensor   = Tensor(risk_features.reshape(1, -1))

    l_f = lstm_out[-1].reshape(1, -1)
    c_f = cnn_out.reshape(1, -1)
    n_f = Tensor(nlp_vec.data.reshape(1, -1))
    r_f = regime_out.reshape(1, -1)

    fused  = agent.fusion(l_f, c_f, n_f, r_f, risk_tensor)
    f_flat = fused.reshape(64)

    nw             = broker.net_worth(current_price)
    unrealised_pnl = 0.0
    if broker.position > 0 and broker.entry_price > 0:
        unrealised_pnl = (current_price - broker.entry_price) / (broker.entry_price + 1e-8)

    portfolio = Tensor(np.array([
        broker.position / (INITIAL_BALANCE + 1e-6),
        broker.balance  / (INITIAL_BALANCE + 1e-6),
        unrealised_pnl,
    ], dtype=np.float64))

    return f_flat.concat(portfolio).concat(Tensor(risk_features))


# ─────────────────────────────────────────────────────────────────────────────
# market hours check
# ─────────────────────────────────────────────────────────────────────────────

def _is_market_open() -> bool:
    """Returns True if current IST time is within NSE trading hours."""
    from datetime import datetime, timezone, timedelta
    IST  = timezone(timedelta(hours=5, minutes=30))
    now  = datetime.now(IST)
    # skip weekends
    if now.weekday() >= 5:
        return False
    after_open  = (now.hour, now.minute) >= (MARKET_OPEN_H,  MARKET_OPEN_M)
    before_close= (now.hour, now.minute) <= (MARKET_CLOSE_H, MARKET_CLOSE_M)
    return after_open and before_close


def _time_to_open_seconds() -> int:
    """Seconds until next market open (rough estimate)."""
    from datetime import datetime, timezone, timedelta
    IST   = timezone(timedelta(hours=5, minutes=30))
    now   = datetime.now(IST)
    today_open = now.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M,
                             second=0, microsecond=0)
    if now < today_open and now.weekday() < 5:
        return int((today_open - now).total_seconds())
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═'*55}")
    print(f"  PPO Paper Trader — {TICKER}  NSE  5m candles")
    print(f"  Balance: ₹{INITIAL_BALANCE:,.0f}  |  Window: {WINDOW_SIZE}")
    print(f"{'═'*55}\n")

    # ── load model ────────────────────────────────────────────────────────────
    agent = build_agent()
    model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else CHECKPOINT_PATH
    if os.path.exists(model_path):
        load_model(agent, model_path)
        print(f"[runner] model loaded ← {model_path}")
    else:
        print(f"[runner] WARNING — no saved model found. Train first!")

    for drop in (agent.actor_drop1, agent.actor_drop2,
                 agent.critic_drop1, agent.critic_drop2):
        drop.training = False

    # ── components ────────────────────────────────────────────────────────────
    broker = UpstoxPaperBroker(
        initial_balance   = INITIAL_BALANCE,
        ticker            = TICKER,
        log_path          = TRADE_LOG_PATH,
        trade_threshold   = 0.515,
        neutral_zone      = 0.10,
        cooldown_bars     = 8,
        max_drawdown_limit= 0.20,
    )

    risk_manager = RiskManager(
        initial_balance      = INITIAL_BALANCE,
        max_drawdown_limit   = 0.20,
        kelly_fraction_cap   = 0.25,
        atr_multiplier       = 2.0,
        black_swan_threshold = 6.0,
        vol_window           = 50,
    )

    nlp_vec = Tensor(np.zeros((1, 64), dtype=np.float64))

    # ── wait for market open ──────────────────────────────────────────────────
    if not _is_market_open():
        secs = _time_to_open_seconds()
        if secs > 0:
            print(f"[runner] market closed — opening in ~{secs//60}m {secs%60}s")
            print(f"[runner] seeding candle buffer while waiting …")
        else:
            print(f"[runner] market closed (weekend or after hours)")
            print(f"[runner] starting stream anyway for testing …")

    # ── start stream ──────────────────────────────────────────────────────────
    stream = LiveCandleStream(TICKER, buffer_size=WINDOW_SIZE + 10)
    stream.start()
    print(f"[runner] stream started — waiting for buffer to fill …")
    time.sleep(5)

    # ── tracking ──────────────────────────────────────────────────────────────
    last_buffer_count = 0
    step              = 0
    prev_price        = None

    print(f"[runner] entering live loop — Ctrl-C to stop\n")

    try:
        while True:
            # outside market hours — pause but keep stream alive
            if not _is_market_open():
                # force close any open position at end of day
                if broker.position > 0 and prev_price:
                    print("\n[runner] market closing — forcing flat")
                    broker._close_position(prev_price, reason="eod")
                sys.stdout.write("\r[runner] market closed — waiting …" + " "*20)
                sys.stdout.flush()
                time.sleep(30)
                continue

            window, current_price = stream.get_window(WINDOW_SIZE)

            if window is None:
                time.sleep(POLL_SECONDS)
                continue

            current_count = len(stream._raw_buffer)
            if current_count == last_buffer_count:
                time.sleep(POLL_SECONDS)
                continue
            last_buffer_count = current_count
            step += 1

            # update risk manager
            if prev_price is not None:
                nw = broker.net_worth(current_price)
                risk_manager.update_market(current_price, prev_price, nw)

            # build state
            try:
                state = extract_state(window, broker, risk_manager,
                                      agent, current_price, nlp_vec)
            except Exception as e:
                print(f"\n[runner] state error: {e} — skipping")
                prev_price = current_price
                continue

            # get action
            action = agent.select_action(state)
            agent.states.clear()
            agent.actions.clear()
            agent.log_probs.clear()
            agent.values.clear()

            # risk filter
            action = risk_manager.adjust_action(action, current_price)

            # execute
            result = broker.execute(action, current_price)

            # update risk after trade
            if result["trade_occurred"]:
                if result["side"] == "SELL":
                    risk_manager.record_trade(result["pnl"])
                elif result["side"] == "BUY":
                    risk_manager.on_open_position(current_price, float(action[0]))

            # print step
            nw       = result["net_worth"]
            nw_color = GREEN if nw >= INITIAL_BALANCE else RED
            pos      = result["position"]
            shares   = result.get("shares_held", 0)

            sys.stdout.write(
                f"\r[{step:5d}] "
                f"₹{current_price:8.2f}  "
                f"NW: {nw_color}₹{nw:10.2f}{RESET}  "
                f"Pos: {shares:4d} shares (₹{pos:8.2f})  "
                f"Trades: {result['n_trades']:3d}  "
                f"WR: {result['win_rate']:.1%}  "
                f"DD: {result['drawdown']:.1%}"
                + " " * 5
            )
            sys.stdout.flush()

            if result["trade_occurred"]:
                sys.stdout.write("\n")
                sys.stdout.flush()

            prev_price = current_price

    except KeyboardInterrupt:
        print("\n\n[runner] stopped by user")

    finally:
        stream.stop()
        price = prev_price or INITIAL_BALANCE
        print(broker.summary(price))


if __name__ == "__main__":
    main()