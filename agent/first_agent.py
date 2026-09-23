"""Терминальный чат с агентом: цикл диалога, команды, обработка ошибок."""

import asyncio
import time

import httpx2
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from agent.composition import create_agent
from agent.ui import (
    HISTORY_DIR,
    assistant_bubble,
    console,
    format_usage,
    gradient_banner,
    print_conversation,
    print_help,
    status_line,
)


def ask_user(conversation_id: str) -> str:
    """Спрашивает ввод пользователя и эхом печатает его панелью 'Вы'"""
    console.print()
    stamp = time.strftime("%H:%M:%S")
    user_input: str = Prompt.ask(f"[dim][{stamp}][/dim] [bold cyan]Беседа {conversation_id} >>> [/bold cyan]").strip()
    console.print(
        Panel(
            user_input,
            title="[bold cyan]Вы[/bold cyan]",
            title_align="left",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()
    return user_input


def new_conversation_id() -> str:
    """Генерирует id беседы по времени."""
    return time.strftime("conv_%Y%m%d_%H%M%S")


def handle_command(user_input: str, conversation_id: str) -> str | None:
    """Обрабатывает команду /<команда>; возвращает id беседы, None - если введено обычное сообщение."""
    if user_input == "/new":
        cid = new_conversation_id()
        console.print(Rule(f"{cid}", style="cyan"))
        return cid
    if user_input in ("/help", "/?"):
        print_help()
        return conversation_id
    if user_input == "/list":
        print_conversation()
        return conversation_id
    if user_input.startswith("/open"):
        parts = user_input.split(maxsplit=1)
        if len(parts) < 2:
            console.print(Text("Использование: /open <имя_беседы> (см. /list)", style="yellow"))
            return conversation_id
        cid = parts[1].strip()
        if not (HISTORY_DIR / f"{cid}.json").exists():
            console.print(Text(f"Беседа не найдена: {cid} (см. /list)", style="yellow"))
            return conversation_id
        console.print(Rule(f"{cid}", style="cyan"))
        return cid
    if user_input.startswith("/"):
        console.print(Text("Неизвестная команда, см. /help", style="yellow"))
        return conversation_id
    return None


def print_api_error(e: httpx2.HTTPError) -> None:
    """Печатает ошибку API понятным текстом вместо трейсбека."""
    if isinstance(e, httpx2.HTTPStatusError):
        status = e.response.status_code
        if status == 429:
            console.print(
                Text("Лимит запросов (429), повторы не помогли. Подождите или смените MODEL в .env", style="red")
            )
        elif status == 401:
            console.print(Text("Неверный API_KEY. Проверьте .env.", style="red"))
        else:
            console.print(Text(f"Ошибка API: {status}", style="red"))
    else:
        console.print(Text(f"Сетевая ошибка: {type(e).__name__}: {e}", style="red"))


async def main() -> None:
    """
    Простой чат в терминале

    Команды:
        /new            - начать новую беседу
        /list           - список сохранённых бесед
        /open <имя>     - продолжить беседу
        /help           - справка по командам
        /exit / Ctrl+D  - выход
    """
    gradient_banner()

    async with httpx2.AsyncClient(timeout=httpx2.Timeout(120)) as http_client:
        agent = create_agent(http_client, True, "workspace")
        conversation_id = new_conversation_id()
        status_line(conversation_id)

        while True:
            try:
                user_input = ask_user(conversation_id)
            except EOFError:
                break
            except KeyboardInterrupt:
                console.print(Text("\nДо свидания!", style="dim"))
                break

            if user_input.lower() in ("/exit", "/quit", "/выход", "/q"):
                console.print(Text("До свидания! 👋", style="dim"))
                break

            if not user_input:
                continue

            handled = handle_command(user_input, conversation_id)
            if handled is not None:
                conversation_id = handled
                continue

            try:
                started = time.perf_counter()
                with console.status(Text("Модель думает…", style="magenta"), spinner="dots"):
                    answer, usage = await agent.run(user_input, conversation_id)
                elapsed = time.perf_counter() - started

            except httpx2.HTTPError as e:
                print_api_error(e)
                continue

            console.print(format_usage(usage, elapsed))
            console.print()
            assistant_bubble(answer)


if __name__ == "__main__":
    asyncio.run(main())
