'''
Live telemetry dashboard. Upgraded to support vectorized rollouts,
displaying a unified Command Center mapping all active parallel environments.
Optimized for 4-6 core multitasking environments.
'''

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich import box
from collections import deque
import time

import sys

console = Console(file=sys.__stdout__)
currency_units = "$"

class VectorizedTelemetry:
    def __init__(self, max_history=20):
        self.max_history = max_history
        self.iteration_log = deque(maxlen=max_history)
        self.live = None
        self.initial_balance = 10000.0

        # Live batch data tracking
        self.batch_data = {}
        self.avg_rewards = {}
        
        self.grad_norm_head = 0.0
        self.grad_norm_extractor = 0.0
        self.grad_norm_fusion = 0.0

        self._start_time = time.time()
        self._current_episode = 0
        self._total_episodes = 0

    def set_episode_info(self, current_episode: int, total_episodes: int):
        """Called once per episode so the header can display progress."""
        self._current_episode = current_episode
        self._total_episodes = total_episodes

    def update_step(self, ticker_data_list):
        """Updates the live grid mapping for all vectorized environments."""
        for data in ticker_data_list:
            self.batch_data[data['ticker']] = data
        if self.live:
            self.live.update(self._build_layout())

    def update_rewards(self, avg_reward_breakdown):
        self.avg_rewards = avg_reward_breakdown

    def update_grad_norms(self, head_norm, extractor_norm, fusion_norm=0.0):
        self.grad_norm_head = head_norm
        self.grad_norm_extractor = extractor_norm
        self.grad_norm_fusion = fusion_norm

    def log_iteration(self, iteration, avg_nw, growth, bh, alpha, sharpe, sortino, wr, trades, is_best):
        """Logs the aggregated end-of-iteration metrics."""
        self.iteration_log.append({
            'iter': iteration,
            'avg_nw': avg_nw,
            'growth': growth,
            'bh': bh,
            'alpha': alpha,
            'sharpe': sharpe,
            'sortino': sortino,
            'wr': wr,
            'trades': trades,
            'is_best': is_best
        })

    def _color_val(self, val, threshold=0):
        return "green" if val >= threshold else "red"

    def _build_batch_table(self):
        t = Table(box=box.SIMPLE, expand=True, header_style="bold cyan")
        t.add_column("Ticker")
        t.add_column("Net Worth", justify="right")
        t.add_column("Position", justify="right")
        t.add_column("Price", justify="right")
        t.add_column("Reward", justify="right")
        t.add_column("Trades (WR)", justify="right")
        t.add_column("Dir Bias", justify="right")

        for ticker, d in sorted(self.batch_data.items()):
            nw_color = self._color_val(d['net_worth'], self.initial_balance)
            pos_color = "green" if d['position'] >= 0 else "red"
            pos_sign = "L" if d['position'] > 0 else ("S" if d['position'] < 0 else "-")
            rew_color = self._color_val(d['reward'], 0)
            
            dm = d.get('dir_mean', 0.0)
            dm_color = "bold red" if abs(dm) > 0.5 else ("yellow" if abs(dm) > 0.2 else "green")

            t.add_row(
                f"[bold]{ticker}[/]",
                f"[{nw_color}]{currency_units}{d['net_worth']:,.2f}[/]",
                f"[{pos_color}]{pos_sign} {currency_units}{abs(d['position']):,.2f}[/]",
                f"{currency_units}{d['price']:.2f}",
                f"[{rew_color}]{d['reward']:+.3f}[/]",
                f"{d['trades']} ({d['win_rate']:.0%})",
                f"[{dm_color}]{dm:+.3f}[/]"
            )

        first_val = next(iter(self.batch_data.values()), None)
        step  = first_val['step']        if first_val else 0
        total = first_val['total_steps'] if first_val else 0
        elapsed = int(time.time() - self._start_time)

        ep_str = (f"Episode {self._current_episode}/{self._total_episodes} | "
                  if self._total_episodes > 0 else "")
        title = (f"[bold white]Parallel Vector Environments | {ep_str}"
                 f"Step {step}/{total} | {elapsed//60}m {elapsed%60}s[/]")
        return Panel(t, title=title, border_style="blue", box=box.ROUNDED)

    def _build_reward_panel(self):
        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold cyan", width=14)
        t.add_column(width=10)

        def fmt(v):
            color = "green" if v > 0 else ("red" if v < 0 else "white")
            return f"[{color}]{v:+.3f}[/{color}]"

        for k, v in self.avg_rewards.items():
            if k != 'total':
                t.add_row(k.capitalize(), fmt(v))
        
        t.add_row("─" * 12, "─" * 8)
        t.add_row("Batch Avg", fmt(self.avg_rewards.get('total', 0.0)))

        return Panel(t, title="[bold white]Avg Rewards[/]", border_style="yellow", box=box.ROUNDED)

    def _build_grad_panel(self):
        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold cyan", width=14)
        t.add_column(width=10)

        def grad_color(v):
            if v > 0.8: return "red"
            if v > 0.4: return "yellow"
            return "green"

        hn, en, fn = self.grad_norm_head, self.grad_norm_extractor, self.grad_norm_fusion
        t.add_row("Head",      f"[{grad_color(hn)}]{hn:.4f}[/]")
        t.add_row("Extractor", f"[{grad_color(en)}]{en:.4f}[/]")
        t.add_row("Fusion",    f"[{grad_color(fn)}]{fn:.4f}[/]")

        return Panel(t, title="[bold white]Grad Norms[/]", border_style="magenta", box=box.ROUNDED)

    def _build_history_panel(self):
        t = Table(box=box.SIMPLE, expand=True, header_style="bold cyan")
        t.add_column("Ep",   width=5)
        t.add_column("Avg Balance",   justify="right")
        t.add_column("Growth",        justify="right")
        t.add_column("B&H",           justify="right")
        t.add_column("Alpha",         justify="right")
        t.add_column("Sharpe",        justify="right")
        t.add_column("Sortino",       justify="right")
        t.add_column("Avg Trades",    justify="right")
        t.add_column("Avg WR",        justify="right")
        t.add_column("Best?",         justify="center")

        for r in reversed(self.iteration_log):
            bal_color   = self._color_val(r['avg_nw'], self.initial_balance)
            alpha_color = self._color_val(r['alpha'], 0)
            sh_color    = "green" if r['sharpe']  >= 0.5 else ("yellow" if r['sharpe']  >= 0 else "red")
            so_color    = "green" if r['sortino'] >= 0.5 else ("yellow" if r['sortino'] >= 0 else "red")
            bh_color    = self._color_val(r['bh'], 0)
            star = "[bold yellow]★[/]" if r['is_best'] else ""

            t.add_row(
                str(r['iter']),
                f"[{bal_color}]{currency_units}{r['avg_nw']:,.0f}[/]",
                f"{r['growth']:+.2%}",
                f"[{bh_color}]{r['bh']:+.2%}[/]",
                f"[{alpha_color}]{r['alpha']:+.2%}[/]",
                f"[{sh_color}]{r['sharpe']:+.2f}[/]",
                f"[{so_color}]{r['sortino']:+.2f}[/]",
                f"{r['trades']:.1f}",
                f"{r['wr']:.1%}",
                star
            )

        return Panel(t, title="[bold white]Episode History[/]", border_style="cyan", box=box.ROUNDED)

    def _build_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="top",    ratio=2),
            Layout(name="bottom", ratio=1),
        )
        layout["top"].split_row(
            Layout(self._build_batch_table(), name="batch_grid", ratio=3),
            Layout(name="side_panels", ratio=1)
        )
        layout["side_panels"].split_column(
            Layout(self._build_reward_panel()),
            Layout(self._build_grad_panel())
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