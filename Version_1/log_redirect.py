# log_redirect.py
import sys
import os
import re
import warnings
import builtins

# Color Definitions
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Save the absolute raw system terminal streams before any third-party library or script overrides them
if '_real_terminal_stdout' not in globals():
    _real_terminal_stdout = sys.__stdout__
    _real_terminal_stderr = sys.__stderr__
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    _original_print = builtins.print

_log_file_handle = None
_stdout_proxy = None
_stderr_proxy = None
_original_showwarning = None

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_run.log")
COUNTER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_counter.txt")

class TelemetryFirewallStream:
    """
    Acts as a proxy stream that intercepts standard outputs/errors:
    - Writes every single clean message directly to training_run.log.
    - Forces immediate operating system-level sync (os.fsync) to bypass cache write delays.
    - Allows only rich telemetry tables, layouts, and training updates to show in the terminal.
    - Beautifully colorizes financial gains in GREEN, losses in RED, and [RISK] alerts in BLUE.
    """
    def __init__(self, log_file_handle, real_terminal, is_stderr=False):
        self.log_file = log_file_handle
        self.real_terminal = real_terminal
        self.is_stderr = is_stderr

    def write(self, message):
        if not message:
            return

        # 1. Strip ANSI escape characters and write cleanly to the log file
        clean_msg = re.sub(r'\x1b\[[0-9;]*m', '', message)
        self.log_file.write(clean_msg)
        self.log_file.flush()  # Push Python internal buffers to the OS
        
        # Force the OS to physically write/sync the file descriptor immediately to disk
        try:
            os.fsync(self.log_file.fileno())
        except Exception:
            pass

        # 2. Handle terminal routing logic
        if self.is_stderr:
            # Let tracebacks, critical execution errors, and custom failures print directly to terminal
            is_critical = any(keyword in message for keyword in ["Traceback", "Error", "Exception", "FAILED", "Critical"])
            if is_critical:
                self.real_terminal.write(f"{RED}{message}{RESET}")
                self.real_terminal.flush()
        else:
            # Check if this message contains the telemetry, layout boxes, or agent statistics
            is_telemetry = (
                any(char in message for char in ["═", "─", "╢", "╟", "║", "│", "┌", "┐", "└", "┘", "┼", "┬", "┴"]) or
                any(keyword in message for keyword in [
                    "EPISODE", "Step", "Net Worth", "Alpha", "Sharpe", "Sortino", "Win Rate", 
                    "Trades", "Gradient", "AAPL", "SPY", "IWM", "XLE", "XBI", "NVDA", "QQQ", 
                    "INITIALIZING VECTORIZED PPO", "Telemetry", "Benchmark"
                ]) or
                "\033" in message or "\x1b" in message
            )

            if is_telemetry:
                colored_message = self.colorize(message)
                self.real_terminal.write(colored_message)
                self.real_terminal.flush()

    def flush(self):
        self.log_file.flush()
        try:
            os.fsync(self.log_file.fileno())
        except Exception:
            pass
        self.real_terminal.flush()

    def colorize(self, text: str) -> str:
        """Dynamically applies high-fidelity colors based on financial outcomes and risk alerts."""
        # 1. Format [RISK] messages in bright blue
        if "[RISK]" in text:
            text = text.replace("[RISK]", f"{BLUE}[RISK]{RESET}")

        # 2. Highlight negative values/Losses in bright RED
        text = re.sub(
            r'(-\s*\$\s*\d+(?:\,\d{3})*(?:\.\d+)?)', 
            rf"{RED}\1{RESET}", 
            text
        )
        text = re.sub(
            r'(-\s*\d+(?:\.\d+)?\s*%)', 
            rf"{RED}\1{RESET}", 
            text
        )

        # 3. Highlight positive values/Profits in bright GREEN
        text = re.sub(
            r'(\+\s*\$\s*\d+(?:\,\d{3})*(?:\.\d+)?)', 
            rf"{GREEN}\1{RESET}", 
            text
        )
        text = re.sub(
            r'(\+\s*\d+(?:\,\d{3})*(?:\.\d+)?\s*%)', 
            rf"{GREEN}\1{RESET}", 
            text
        )

        return text


def custom_print(*args, **kwargs):
    """
    Overridden built-in print statement that intercepts standard print() outputs 
    and forces them to go through our custom firewall proxy.
    """
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    file = kwargs.get('file', None)

    if file is None:
        file = sys.stdout

    message = sep.join(str(arg) for arg in args) + end

    # If the print is targeting standard output or standard error, force through proxies
    if file in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__, _stdout_proxy, _stderr_proxy):
        if file in (sys.stderr, _stderr_proxy):
            _stderr_proxy.write(message)
        else:
            _stdout_proxy.write(message)
    else:
        # Otherwise use the original built-in print
        _original_print(*args, **kwargs)


def reset_episode_log(episode_num):
    """
    Truncates and clears the log file to reset it for a fresh training episode.
    Updates active proxy streams on-the-fly.
    """
    global _log_file_handle, _stdout_proxy, _stderr_proxy
    if _log_file_handle:
        try:
            # Safely close old handle
            _log_file_handle.close()
        except Exception:
            pass

        try:
            # Re-open the log file in write mode ("w") which truncates/wipes it instantly
            _log_file_handle = open(LOG_PATH, "w", encoding="utf-8", buffering=1)
            
            # Write a fresh episode header to the top of the clean log
            _log_file_handle.write(f"{'='*70}\n[EPISODE {episode_num}] FRESH SESSION LOG INITIATED\n{'='*70}\n\n")
            _log_file_handle.flush()
            try:
                os.fsync(_log_file_handle.fileno())
            except Exception:
                pass
            
            # Bind the fresh handle to our running output proxies
            if _stdout_proxy:
                _stdout_proxy.log_file = _log_file_handle
            if _stderr_proxy:
                _stderr_proxy.log_file = _log_file_handle
        except Exception as e:
            _real_terminal_stdout.write(f"{RED}[log_redirect] Failed to reset episode log: {e}{RESET}\n")
            _real_terminal_stdout.flush()


def redirect_prints():
    global _log_file_handle, _stdout_proxy, _stderr_proxy, _original_showwarning
    try:
        # Read the persistent counter file
        run_count = 0
        if os.path.exists(COUNTER_PATH):
            try:
                with open(COUNTER_PATH, "r") as f:
                    run_count = int(f.read().strip())
            except Exception:
                run_count = 0

        run_count += 1

        # Check if we reached the limit to wipe and reset
        if run_count >= 5:
            mode = "w"  # Truncates (wipes) file
            run_count = 1  # Reset the cycle counter back to 1
        else:
            mode = "a"  # Appends to the existing file

        # Initialize the log file handle with the chosen mode
        _log_file_handle = open(LOG_PATH, mode, encoding="utf-8", buffering=1)
        
        # Add visual separators inside the log file
        if mode == "a":
            _log_file_handle.write(f"\n\n{'='*70}\n[RUN {run_count}/5] STARTING VECTORIZED RUN SESSION\n{'='*70}\n\n")
        else:
            _log_file_handle.write(f"{'='*70}\n[RUN 1/5] LOG CLEARED - FRESH 5-RUN CYCLE INITIATED\n{'='*70}\n\n")
        
        _log_file_handle.flush()
        try:
            os.fsync(_log_file_handle.fileno())
        except Exception:
            pass

        # Save the updated run counter
        with open(COUNTER_PATH, "w") as f:
            f.write(str(run_count))

        # Instantiate proxy controllers with raw terminal references
        _stdout_proxy = TelemetryFirewallStream(_log_file_handle, _real_terminal_stdout, is_stderr=False)
        _stderr_proxy = TelemetryFirewallStream(_log_file_handle, _real_terminal_stderr, is_stderr=True)
        
        # Lock down system streams globally
        sys.stdout = _stdout_proxy
        sys.stderr = _stderr_proxy
        sys.__stdout__ = _stdout_proxy
        sys.__stderr__ = _stderr_proxy
        
        # Hijack built-in print function globally
        builtins.print = custom_print
        
        # Intercept native Python warnings
        _original_showwarning = warnings.showwarning
        def _custom_showwarning(message, category, filename, lineno, file=None, line=None):
            clean_msg = f"[WARNING] {category.__name__}: {message} ({filename}:{lineno})\n"
            _log_file_handle.write(clean_msg)
            _log_file_handle.flush()
            try:
                os.fsync(_log_file_handle.fileno())
            except Exception:
                pass
        warnings.showwarning = _custom_showwarning
        
        _real_terminal_stdout.write(f"{BLUE}[log_redirect] Silent firewall activated. Running in cycle count {run_count}/5. Log file path: {LOG_PATH}{RESET}\n")
        _real_terminal_stdout.flush()
    except Exception as e:
        _real_terminal_stdout.write(f"{RED}[log_redirect] FAILED initialization: {e}{RESET}\n")
        _real_terminal_stdout.flush()
    return LOG_PATH


def restore_prints():
    global _log_file_handle, _stdout_proxy, _stderr_proxy, _original_showwarning
    
    # Restore standard streams
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr
    sys.__stdout__ = _real_terminal_stdout
    sys.__stderr__ = _real_terminal_stderr
    
    # Restore standard built-in print
    builtins.print = _original_print
    
    # Restore warnings
    if _original_showwarning is not None:
        warnings.showwarning = _original_showwarning
        _original_showwarning = None
        
    if _log_file_handle:
        try:
            _log_file_handle.close()
        except:
            pass
        _log_file_handle = None
    _stdout_proxy = None
    _stderr_proxy = None
