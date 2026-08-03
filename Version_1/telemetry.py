"""
telemetry.py — Live visualization of PPO Agent training and testing.
Updated to support 2D Action Space (Direction and Sizing/Leverage).
"""

import sys
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console
from rich import box
from collections import deque
import time
import numpy as np

# Force Rich to use the original un-redirected standard output stream
# so it bypasses print redirection and renders directly to the terminal screen.
console_file = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
console = Console(file=console_file)

# Currency units config
currency_units = "$"

class Telemetry:
    def __init__(self, max_history=20):
        self.max_history = max_history
        self.episode_log = deque(maxlen=max_history)
        self.live = None

        # current episode state
        self.ticker = "-"
        self.episode = 0
        self.step = 0
        self.total_steps = 0
        self.net_worth = 0.0
        self.balance = 0.0
        self.position = 0.0
        self.price = 0.0
        self.std = 0.0
        self.dir_mean = 0.0  # policy signal internal tracker
        self.num_trades = 0
        self.winning_trades = 0
        self.total_reward = 0.0
        self.max_drawdown = 0.0
        self.peak_worth = 0.0
        self.initial_balance = 10000.0

        # Action-space metrics
        self.last_action_dir = 0.0   # Long/Short/Hold raw or discrete representation
        self.last_action_size = 0.0  # Leverage or trade sizing [0.0 - 1.0+]
        self.rolling_actions = deque(maxlen=100) # capture last 100 steps of actions

        # reward breakdown (last step)
        self.r_trade = 0.0
        self.r_step = 0.0
        self.r_stress = 0.0
        self.r_milestone = 0.0
        self.r_terminal = 0.0
        self.r_total_step = 0.0
        self.r_hold_loser = 0.0
        self.r_premature_close = 0.0

        # grad norms (last update)
        self.grad_norm_head = 0.0
        self.grad_norm_extractor = 0.0
        self.grad_norm_fusion = 0.0
        self.milestones_crossed = set()

        self._start_time = time.time()

    def update_step(self, ticker, episode, step, total_steps,
                    net_worth, balance, position, price, std,
                    num_trades, winning_trades, total_reward,
                    r_trade=0.0, r_step=0.0, r_hold_loser=0.0, r_stress=0.0,
                    r_premature_close=0.0, r_milestone=0.0, r_terminal=0.0, r_total=0.0,
                    milestones_crossed=None, dir_mean=0.0,
                    action_direction=0.0, action_size=0.0):
        self.ticker = ticker
        self.episode = episode
        self.step = step
        self.total_steps = total_steps
        self.net_worth = net_worth
        self.balance = balance
        self.position = position
        self.price = price
        self.std = std
        self.dir_mean = dir_mean
        
        # Action space recording
        self.last_action_dir = action_direction
        self.last_action_size = action_size
        self.rolling_actions.append((action_direction, action_size))
        
        self.num_trades = num_trades
        self.winning_trades = winning_trades
        self.total_reward = total_reward
        self.peak_worth = max(self.peak_worth, net_worth)
        self.max_drawdown = (self.peak_worth - net_worth) / (self.peak_worth + 1e-8)
        self.r_trade = r_trade
        self.r_step = r_step
        self.r_stress = r_stress
        self.r_premature_close = r_premature_close
        self.r_milestone = r_milestone
        self.r_terminal = r_terminal
        self.r_total_step = r_total
        self.r_hold_loser = r_hold_loser
        
        if milestones_crossed:
            self.milestones_crossed = milestones_crossed
        if self.live:
            self.live.update(self._build_layout())

    def update_grad_norms(self, head_norm, extractor_norm, fusion_norm=0.0):
        self.grad_norm_head = head_norm
        self.grad_norm_extractor = extractor_norm
        self.grad_norm_fusion = fusion_norm

    def log_episode(self, episode, ticker, final_balance, total_reward,
                    num_trades, win_rate, max_drawdown, std, bankrupt,
                    dir_mean=0.0, sharpe=0.0, sortino=0.0,
                    benchmark_return=0.0, alpha_vs_bh=0.0):
        
        # Calculate final episode action stats
        avg_size = 0.0
        if self.rolling_actions:
            avg_size = float(np.mean([a[1] for a in self.rolling_actions]))

        self.episode_log.append({
            'ep':               episode,
            'ticker':           ticker,
            'balance':          final_balance,
            'reward':           total_reward,
            'trades':           num_trades,
            'wr':               win_rate,
            'dd':               max_drawdown,
            'std':              std,
            'bankrupt':         bankrupt,
            'dir_mean':         dir_mean,
            'avg_size':         avg_size,
            'sharpe':           sharpe,
            'sortino':          sortino,
            'benchmark_return': benchmark_return,
            'alpha_vs_bh':      alpha_vs_bh,
        })
        self.peak_worth = 0.0  # reset peak
        self.rolling_actions.clear()

    def _color_val(self, val, good_threshold=0, reverse=False):
        if reverse:
            return "green" if val <= good_threshold else "red"
        return "green" if val >= good_threshold else "red"

    def _build_live_panel(self):
        wr = self.winning_trades / self.num_trades if self.num_trades > 0 else 0.0
        progress = self.step / max(self.total_steps, 1)
        bar_len  = 20
        filled   = int(bar_len * progress)
        progress_bar = f"[{'█' * filled}{'░' * (bar_len - filled)}] {progress:.1%}"

        nw_color  = self._color_val(self.net_worth, self.initial_balance)
        wr_color  = self._color_val(wr, 0.52)
        pos_sign  = "L" if self.position > 0 else ("S" if self.position < 0 else "-")
        pos_color = "green" if self.position >= 0 else "red"
        elapsed   = int(time.time() - self._start_time)

        milestones_str = ", ".join([f"{currency_units}{m:,}" for m in sorted(self.milestones_crossed)]) or "none"

        # Action space visualization
        size_bar_len = 10
        size_filled = min(int(size_bar_len * self.last_action_size), size_bar_len)
        size_bar = f"[{'■' * size_filled}{' ' * (size_bar_len - size_filled)}] {self.last_action_size:.2%}"

        # Classify the last action direction
        if self.last_action_dir > 0.3:
            act_dir_str = f"[green]LONG ({self.last_action_dir:+.2f})[/green]"
        elif self.last_action_dir < -0.3:
            act_dir_str = f"[red]SHORT ({self.last_action_dir:+.2f})[/red]"
        else:
            act_dir_str = f"[yellow]HOLD ({self.last_action_dir:+.2f})[/yellow]"

        # Color-highlight directional bias
        if abs(self.dir_mean) > 0.8:
            dir_color = "bold red"
        elif abs(self.dir_mean) > 0.3:
            dir_color = "yellow"
        else:
            dir_color = "green"

        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold cyan", width=22)
        t.add_column(width=30)

        t.add_row("Ticker",        f"[bold]{self.ticker}[/bold] | Ep {self.episode}")
        t.add_row("Progress",       progress_bar)
        t.add_row("Step",          f"{self.step} / {self.total_steps}")
        t.add_row("Net Worth",     f"[{nw_color}]{currency_units}{self.net_worth:,.2f}[/{nw_color}]")
        t.add_row("Balance",       f"{currency_units}{self.balance:,.2f}")
        t.add_row("Position",      f"[{pos_color}]{pos_sign} {currency_units}{abs(self.position):,.2f}[/{pos_color}]")
        t.add_row("Price",         f"{currency_units}{self.price:.2f}")
        t.add_row("Trades",        f"{self.num_trades} | WR [{wr_color}]{wr:.1%}[/{wr_color}]")
        t.add_row("Drawdown",      f"[{'red' if self.max_drawdown > 0.15 else 'yellow'}]{self.max_drawdown:.1%}[/{'red' if self.max_drawdown > 0.15 else 'yellow'}]")
        t.add_row("Total Reward",  f"[{'green' if self.total_reward >= 0 else 'red'}]{self.total_reward:.2f}[/{'green' if self.total_reward >= 0 else 'red'}]")
        t.add_row("Std (Explore)", f"{self.std:.4f}")
        t.add_row("Last Action Dir", act_dir_str)
        t.add_row("Last Action Size", size_bar)
        t.add_row("Dir Mean (Bias)", f"[{dir_color}]{self.dir_mean:+.4f}[/{dir_color}]")
        t.add_row("Milestones",    milestones_str)
        t.add_row("Elapsed",       f"{elapsed // 60}m {elapsed % 60}s")

        return Panel(t, title="[bold white]Live Episode[/bold white]", border_style="blue", box=box.ROUNDED)

    def _build_reward_panel(self):
        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold cyan", width=18)
        t.add_column(width=12)

        def fmt(v):
            color = "green" if v > 0 else ("red" if v < 0 else "white")
            return f"[{color}]{v:+.4f}[/{color}]"

        t.add_row("Trade",      fmt(self.r_trade))
        t.add_row("Step",       fmt(self.r_step))
        t.add_row("Hold Loser", fmt(self.r_hold_loser))
        t.add_row("Stress",     fmt(self.r_stress))
        t.add_row("Prem Close", fmt(self.r_premature_close))
        t.add_row("Milestone",  fmt(self.r_milestone))
        t.add_row("Terminal",   fmt(self.r_terminal))
        t.add_row("─" * 16,     "─" * 10)
        t.add_row("Step Total", fmt(self.r_total_step))

        return Panel(t, title="[bold white]Last Step Reward[/bold white]", border_style="yellow", box=box.ROUNDED)

    def _build_action_distribution_panel(self):
        # Calculate metrics from rolling window (last 100 actions)
        longs, shorts, holds = 0, 0, 0
        avg_size = 0.0
        
        if self.rolling_actions:
            sizes = []
            for d, s in self.rolling_actions:
                sizes.append(s)
                if d > 0.3:
                    longs += 1
                elif d < -0.3:
                    shorts += 1
                else:
                    holds += 1
            total = len(self.rolling_actions)
            long_pct = longs / total
            short_pct = shorts / total
            hold_pct = holds / total
            avg_size = float(np.mean(sizes))
        else:
            long_pct = short_pct = hold_pct = 0.0

        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold cyan", width=18)
        t.add_column(width=12)

        t.add_row("Long Freq",  f"[green]{long_pct:.1%}[/green]")
        t.add_row("Short Freq", f"[red]{short_pct:.1%}[/red]")
        t.add_row("Hold Freq",  f"[yellow]{hold_pct:.1%}[/yellow]")
        t.add_row("─" * 16,     "─" * 10)
        t.add_row("Avg Sizing", f"[white]{avg_size:.2%}[/white]")
        t.add_row("Head Norm",  f"{self.grad_norm_head:.4f}")
        t.add_row("Ext Norm",   f"{self.grad_norm_extractor:.4f}")
        t.add_row("Fus Norm",   f"{self.grad_norm_fusion:.4f}")

        return Panel(t, title="[bold white]Policy Insights[/bold white]", border_style="magenta", box=box.ROUNDED)

    def _build_history_panel(self):
        t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        t.add_column("Ep",      width=4)
        t.add_column("Ticker",  width=8)
        t.add_column("Balance", width=11)
        t.add_column("Reward",  width=8)
        t.add_column("Trades",  width=7)
        t.add_column("WR",      width=6)
        t.add_column("DD",      width=6)
        t.add_column("Sharpe",  width=7)
        t.add_column("Sortino", width=7)
        t.add_column("AvgSize", width=8)
        t.add_column("AvgDir",  width=8)
        t.add_column("Status",  width=8)

        for r in reversed(self.episode_log):
            bal_color = "green" if r['balance'] >= self.initial_balance else "red"
            wr_color  = "green" if r['wr'] >= 0.52 else "red"
            status    = "[red]BUST[/red]" if r['bankrupt'] else "[green]OK[/green]"

            dm_val   = r.get('dir_mean', 0.0)
            dm_color = "bold red" if abs(dm_val) > 0.5 else ("yellow" if abs(dm_val) > 0.2 else "green")

            sh       = r.get('sharpe', 0.0)
            sh_color = "green" if sh >= 0.5 else ("yellow" if sh >= 0 else "red")

            so       = r.get('sortino', 0.0)
            so_color = "green" if so >= 0.5 else ("yellow" if so >= 0 else "red")

            avg_sz   = r.get('avg_size', 0.0)

            t.add_row(
                str(r['ep']),
                r['ticker'].replace(".NS", ""),
                f"[{bal_color}]{currency_units}{r['balance']:,.0f}[/{bal_color}]",
                f"[{'green' if r['reward'] >= 0 else 'red'}]{r['reward']:.1f}[/{'green' if r['reward'] >= 0 else 'red'}]",
                str(r['trades']),
                f"[{wr_color}]{r['wr']:.1%}[/{wr_color}]",
                f"{r['dd']:.1%}",
                f"[{sh_color}]{sh:+.2f}[/{sh_color}]",
                f"[{so_color}]{so:+.2f}[/{so_color}]",
                f"{avg_sz:.1%}",
                f"[{dm_color}]{dm_val:+.3f}[/{dm_color}]",
                status,
            )

        return Panel(t, title="[bold white]Episode History[/bold white]", border_style="cyan", box=box.ROUNDED)

    def _build_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="top",    size=20),
            Layout(name="bottom", size=14),
        )
        layout["top"].split_row(
            Layout(self._build_live_panel(),   name="live",   ratio=2),
            Layout(self._build_reward_panel(), name="reward", ratio=1),
            Layout(self._build_action_distribution_panel(), name="grads",  ratio=1),
        )
        layout["bottom"].update(self._build_history_panel())
        return layout

    def start(self):
        self.live = Live(self._build_layout(), refresh_per_second=4,
                         screen=False, console=console)
        self.live.start()

    def stop(self):
        if self.live:
            self.live.stop()