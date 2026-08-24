"""
Utility helpers for the Customer Churn Prediction project.

Provides path resolution, directory helpers, I/O utilities,
and attractive console output for terminal / VS Code / CMD.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import joblib

# ---------------------------------------------------------------------------
# Enable ANSI colors on Windows consoles (CMD / PowerShell / VS Code)
# ---------------------------------------------------------------------------
def _enable_windows_ansi() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL
    except Exception:
        pass


_enable_windows_ansi()

# Prefer UTF-8 so banners / checkmarks render in CMD & VS Code
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Detect whether fancy Unicode is safe
def _supports_unicode() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    return "utf" in enc


_UNICODE = _supports_unicode()


# Color codes (safe even if terminal ignores them)
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


# Symbols (ASCII fallback for older CMD encodings)
class S:
    CHECK = "✔" if _UNICODE else "[OK]"
    CROSS = "✖" if _UNICODE else "[X]"
    INFO = "ℹ" if _UNICODE else "[i]"
    WARN = "!" if _UNICODE else "[!]"
    ARROW = "▶" if _UNICODE else ">"
    STAR = "★" if _UNICODE else "*"
    BAR = "█" if _UNICODE else "#"
    H = "─" if _UNICODE else "-"
    H2 = "━" if _UNICODE else "="
    DH = "═" if _UNICODE else "="
    TL = "┌" if _UNICODE else "+"
    TR = "┐" if _UNICODE else "+"
    BL = "└" if _UNICODE else "+"
    BR = "┘" if _UNICODE else "+"
    VL = "│" if _UNICODE else "|"
    LJ = "├" if _UNICODE else "+"
    RJ = "┤" if _UNICODE else "+"


# ---------------------------------------------------------------------------
# Logging — quieter default so banner output stays clean
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("churn_utils")
logger.setLevel(logging.INFO)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def get_paths() -> dict[str, Path]:
    """Build a dictionary of commonly used project paths."""
    root = get_project_root()
    return {
        "root": root,
        "dataset": root / "dataset",
        "notebooks": root / "notebooks",
        "src": root / "src",
        "model": root / "model",
        "images": root / "images",
        "data_csv": root / "dataset" / "Telco-Customer-Churn.csv",
        "best_model": root / "model" / "best_model.pkl",
        "preprocessor": root / "model" / "preprocessor.pkl",
    }


def save_artifact(obj: Any, filepath: str | Path) -> Path:
    """Persist a Python object to disk with joblib."""
    path = Path(filepath)
    ensure_dir(path.parent)
    joblib.dump(obj, path)
    ok(f"Saved → {path.name}")
    return path.resolve()


def load_artifact(filepath: str | Path) -> Any:
    """Load a joblib-serialized object from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    obj = joblib.load(path)
    info(f"Loaded ← {path.name}")
    return obj


def set_plot_style() -> None:
    """Apply a consistent Matplotlib / Seaborn style for all figures."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 120,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
        }
    )


# ---------------------------------------------------------------------------
# Attractive console output
# ---------------------------------------------------------------------------
WIDTH = 72


def _line(char: str = "─") -> str:
    return char * WIDTH


def banner(title: str, subtitle: str = "") -> None:
    """Print a big project banner."""
    print()
    print(f"{C.CYAN}{C.BOLD}{S.DH * WIDTH}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}  {title.center(WIDTH - 4)}{C.RESET}")
    if subtitle:
        print(f"{C.DIM}  {subtitle.center(WIDTH - 4)}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}{S.DH * WIDTH}{C.RESET}")
    print()


def print_section(title: str, char: str = "=") -> None:
    """Print a clear section header."""
    print()
    print(f"{C.BLUE}{C.BOLD}{S.H2 * WIDTH}{C.RESET}")
    print(f"{C.BLUE}{C.BOLD}  {S.ARROW}  {title}{C.RESET}")
    print(f"{C.BLUE}{C.BOLD}{S.H2 * WIDTH}{C.RESET}")


def step(num: int, total: int, message: str) -> None:
    """Print a numbered progress step."""
    print(f"\n{C.MAGENTA}{C.BOLD}[{num}/{total}]{C.RESET} {C.WHITE}{message}{C.RESET}")


def ok(message: str) -> None:
    """Success line."""
    print(f"  {C.GREEN}{S.CHECK}{C.RESET}  {message}")


def info(message: str) -> None:
    """Info line."""
    print(f"  {C.CYAN}{S.INFO}{C.RESET}  {message}")


def warn(message: str) -> None:
    """Warning line."""
    print(f"  {C.YELLOW}{S.WARN}{C.RESET}  {message}")


def fail(message: str) -> None:
    """Error line."""
    print(f"  {C.RED}{S.CROSS}{C.RESET}  {message}")


def metric_row(name: str, value: float, best: bool = False) -> None:
    """Pretty metric line."""
    mark = f"{C.GREEN}{S.STAR}{C.RESET}" if best else " "
    print(f"  {mark} {name:<18} {C.BOLD}{value:.4f}{C.RESET}")


def print_table(headers: list[str], rows: list[list[str]], highlight_col: int = -1) -> None:
    """Print a simple aligned table."""
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(c)) for c in col) + 2 for col in cols]

    def fmt_row(row: list[str], header: bool = False) -> str:
        cells = []
        for i, cell in enumerate(row):
            text = str(cell).ljust(widths[i])
            if header:
                cells.append(f"{C.BOLD}{text}{C.RESET}")
            elif i == highlight_col:
                cells.append(f"{C.GREEN}{C.BOLD}{text}{C.RESET}")
            else:
                cells.append(text)
        return "  " + "".join(cells)

    print()
    print(fmt_row(headers, header=True))
    print("  " + S.H * sum(widths))
    for row in rows:
        print(fmt_row(row))
    print()


def box(title: str, lines: list[str]) -> None:
    """Print a highlighted result box."""
    inner = WIDTH - 2
    print()
    print(f"{C.GREEN}{C.BOLD}{S.TL}{S.H * inner}{S.TR}{C.RESET}")
    print(f"{C.GREEN}{C.BOLD}{S.VL}  {title.ljust(inner - 3)}{S.VL}{C.RESET}")
    print(f"{C.GREEN}{C.BOLD}{S.LJ}{S.H * inner}{S.RJ}{C.RESET}")
    for line in lines:
        print(f"{C.GREEN}{S.VL}{C.RESET}  {line.ljust(inner - 3)}{C.GREEN}{S.VL}{C.RESET}")
    print(f"{C.GREEN}{C.BOLD}{S.BL}{S.H * inner}{S.BR}{C.RESET}")
    print()


def finish(paths: dict) -> None:
    """Final success summary."""
    box(
        "PIPELINE COMPLETE",
        [
            f"Best model → {paths['best_model'].name}",
            f"Plots      → images/",
            f"Metrics    → model/metrics.csv",
            f"Folder     → {paths['root']}",
        ],
    )
    print(f"{C.DIM}Open images/ to view charts. Use predict.bat to score customers.{C.RESET}")
    print()
