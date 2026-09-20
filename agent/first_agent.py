import asyncio
import time
from pathlib import Path

import httpx2
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from agent.composition import create_agent
from agent.core import UsageStats
from agent.settings import MODEL


console = Console()

HISTORY_DIR = Path("history")


def new_conversation_id() -> str:
    return time.strftime("conv_%Y%m%d_%H%M%S")


def format_usage(data: UsageStats) -> str:
    """Строка статистики токенов/кэша для отображения."""
    return (
        f"[dim]📊 Токены: input={data.prompt_tokens} "
        f"(кэш✅={data.cached} [{data.cache_hit_pct:.0f}%], write={data.cache_write}) | "
        f"output={data.completion_tokens}[/dim]"
    )


def print_conversation() -> None:
    """Список сохранённых бесед из history/."""
    if not HISTORY_DIR.exists():
        console.print("Сохранённых бесед нет.")
        return
    ids = sorted(conv.stem for conv in HISTORY_DIR.glob("*.json"))
    if not ids:
        console.print("Сохранённых бесед нет.")
        return
    for conv_id in ids:
        console.print(f"[blue] -> {conv_id}[/blue]")


async def main() -> None:
    """
    Простой чат в терминале

    Команды:
        /new            - начать новую беседу
        /list           - список сохранённых бесед
        /open <имя>     - продолжить беседу
        /exit / Ctrl+D   - выход
    """
    async with httpx2.AsyncClient(timeout=httpx2.Timeout(120)) as http_client:
        agent = create_agent(http_client, True, "workspace")
        conversation_id = new_conversation_id()
        console.print(
            Panel(
                f"[bold]agent-67[/bold] | модель: [cyan]{MODEL}[/cyan] | workspace/\n\n"
                f"Беседа: {conversation_id}. /new, /list, /open <имя>, /exit — выход."
            )
        )
        while True:
            try:
                console.print()
                stamp = time.strftime("%H:%M:%S")
                user_input = Prompt.ask(
                    f"[dim][{stamp}][/dim] [bold cyan]Беседа {conversation_id} >>> [/bold cyan]"
                ).strip()
                console.print()
            except EOFError:
                break
            except KeyboardInterrupt:
                console.print("\nДо свидания!")
                break

            if user_input.lower() in ("/exit", "/quit", "/выход", "/q"):
                console.print(Panel("До свидания!"))
                break

            if not user_input:
                continue

            if user_input == "/new":
                conversation_id = new_conversation_id()
                console.print(f"Новая беседа: {conversation_id}")
                continue
            if user_input == "/list":
                print_conversation()
                continue
            if user_input.startswith("/open"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    console.print("Использование: /open <имя_беседы> (см. /list)")
                    continue
                cid = parts[1].strip()
                if not (HISTORY_DIR / f"{cid}.json").exists():
                    console.print(f"Беседа не найдена: {cid} (см. /list)")
                    continue
                conversation_id = cid
                console.print(f"Продолжаем беседу: {conversation_id}")
                continue

            if user_input.startswith("/"):
                console.print("Неизвестная команда. Доступны: /new, /list, /open <имя>")
                continue
            try:
                started = time.perf_counter()
                with console.status("[magenta]Модель думает...[/magenta]", spinner="dots"):
                    answer, usage = await agent.run(user_input, conversation_id)
                elapsed = time.perf_counter() - started

            except httpx2.HTTPStatusError as e:
                status = e.response.status_code
                if status == 429:
                    console.print(
                        "[red]Лимит запросов модели (429), повторы не помогли. Подождите минуту или смените MODEL в .env[/red]"
                    )
                elif status == 401:
                    console.print("[red]Неверный API_KEY. Проверьте .env.[/red]")
                else:
                    console.print(f"[red]Ошибка API: {status}[/red]")
                continue
            except httpx2.HTTPError as e:
                console.print(f"[red]Сетевая ошибка: {type(e).__name__}: {e}[/red]")
                continue

            console.print(Rule("[bold magenta]Cache stats[/bold magenta]", style="magenta"))
            console.print(format_usage(usage))
            console.print(f"[dim]⏱ {time.strftime('%H:%M:%S')} — агент ответил за {elapsed:.0f} сек[/dim]")
            console.print(Panel(Markdown(answer), title="Ответ", border_style="cyan"))
            console.print()


if __name__ == "__main__":
    asyncio.run(main())
