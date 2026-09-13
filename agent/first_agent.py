import asyncio

import httpx2

from agent.composition import create_agent


async def main() -> None:
    """
    Простой чат в терминале

    Команда `exit` или Ctrl+D завершает работу.
    """
    async with httpx2.AsyncClient() as http_client:
        agent = create_agent(http_client, True)
        while True:
            try:
                user_input = input("INPUT >>> ").strip()  # noqa: ASYNC250
            except EOFError:
                break

            if user_input == "exit":
                break

            try:
                answer = await agent.run(user_input, "new_conversation_1")
            except httpx2.HTTPStatusError as e:
                status = e.response.status_code
                if status == 429:
                    print("Лимит запросов модели (429), повторы не помогли. Подождите минуту или смените MODEL в .env")
                elif status == 401:
                    print("Неверный API_KEY. Проверьте .env.")
                else:
                    print(f"Ошибка API: {status}")
                continue
            except httpx2.HTTPError as e:
                print(f"Сетевая ошибка: {e}")
                continue

            print(answer)


if __name__ == "__main__":
    asyncio.run(main())
