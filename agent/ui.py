"""Отображение: rich-элементы терминального интерфейса агента."""

import time
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.core import UsageStats
from agent.settings import GIGACHAT_MODEL, MODEL, OLLAMA_MODEL, PROVIDER


HISTORY_DIR = Path("history")

PROVIDER_MODELS = {
    "open_router": MODEL,
    "ollama": OLLAMA_MODEL,
    "giga_chat": GIGACHAT_MODEL,
}

console = Console()

BANNER = r"""
     █████╗  ██████╗ ███████╗███╗   ██╗████████╗  ██████╗ ███████╗
    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝ ██╔════╝ ╚════██║
    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║    ███████╗    ██╔╝
    ██╔══██║██║   ██║██╔══╝  ██ ╚██╗██║   ██║    ██╔═══██╗  ██╔╝
    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║    ╚██████╔╝  ██║
    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝     ╚═════╝   ╚═╝


"""


def gradient_banner() -> None:
    """Печатает баннер с градиентом и строкой провайдер/модель."""
    colors = ["blue", "cyan", "cyan", "bright_cyan", "bright_magenta"]
    console.print()
    for i, line in enumerate(BANNER.splitlines()):
        console.print(Text(line, style=f"bold {colors[min(i, len(colors) - 1)]}"))
    console.print(Text(f"  agent-67 · {PROVIDER} · {PROVIDER_MODELS[PROVIDER]} · workspace/", style="dim italic"))
    console.print()


def format_usage(data: UsageStats, elapsed: float) -> Panel:
    """Строит панель статистики раунда: токены, кэш-бар, время ответа."""
    pct = data.cache_hit_pct or 0.0
    filled = round(pct / 10)
    bar = "▓" * filled + "░" * (10 - filled)
    body = (
        f"[dim]📊 Токены: input={data.prompt_tokens} "
        f"(кэш✅={data.cached} [{data.cache_hit_pct:.0f}%], write={data.cache_write}) | "
        f"output={data.completion_tokens}[/dim]\n\n"
        f"[dim]⚡ Cache[/dim] {bar} [green]{pct:.0f}%[/green]"
        f" ⏱  {time.strftime('%H:%M:%S')} - агент ответил за {elapsed:.0f} сек"
    )
    return Panel(body, title="📊 Раунд", border_style="dim", expand=False)


def print_conversation() -> None:
    """Список сохранённых бесед из history/."""
    if not HISTORY_DIR.exists():
        console.print(Text("Сохранённых бесед нет.", style="dim"))
        return
    ids = sorted(conv.stem for conv in HISTORY_DIR.glob("*.json"))
    if not ids:
        console.print(Text("Сохранённых бесед нет.", style="dim"))
        return
    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column(style="blue")
    for conv_id in ids:
        table.add_row(f"→ {conv_id}")
    console.print(table)


def assistant_bubble(answer: str) -> None:
    """Ответ модели - markdown в скруглённой панели."""
    console.print(
        Panel(
            Markdown(answer),
            border_style="bright_magenta",
            title="[bold magenta]agent-67[/bold magenta]",
            title_align="left",
        )
    )
    console.print()


def print_help() -> None:
    """Панель со списком команд."""
    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("/new", "начать новую беседу")
    table.add_row("/list", "список сохранённых бесед")
    table.add_row("/open <имя>", "продолжить беседу")
    table.add_row("/help", "эта справка")
    table.add_row("/exit", "выход (Ctrl+D, Ctrl+C)")
    console.print(Panel(table, title="Команды", border_style="dim", expand=False))


def status_line(conversation_id: str) -> None:
    """Печатает строку текущей беседы с подсказкой о командах."""
    console.print(
        Text(f" {conversation_id}", style="reverse cyan"),
        Text(" /help — команды ", style="dim"),
    )
