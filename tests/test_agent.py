import uuid

from agent.core import AgentLoop


async def test_tool_loop(build_agent, fake_llm_factory) -> None:
    """Полный цикл: LLM вызывает инструмент, получает результат, даёт финальный ответ."""
    llm = fake_llm_factory(
        [
            # раунд 1: модель просит вызвать инструмент
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"id": "1", "function": {"name": "upper", "arguments": '{"text": "привет"}'}}],
                },
                "finish_reason": "tool_calls",
            },
            # раунд 2: финальный ответ на основе результата инструмента
            {
                "message": {"content": "Модель увидела: ПРИВЕТ", "tool_calls": None},
                "finish_reason": "stop",
            },
        ]
    )

    answer = await build_agent(llm).run("привет", f"conv-{uuid.uuid4()}")

    assert answer == "Модель увидела: ПРИВЕТ"
    assert not llm.script  # оба сценария израсходованы, цикл не сделал лишних раундов


async def test_plain_answer_no_tools(build_agent, fake_llm_factory) -> None:
    """Модель отвечает сразу текстом — цикл завершается без вызова инструментов."""
    llm = fake_llm_factory(
        [
            {
                "message": {"content": "Просто ответ", "tool_calls": None},
                "finish_reason": "stop",
            }
        ]
    )

    answer = await build_agent(llm).run("вопрос", f"conv-{uuid.uuid4()}")

    assert answer == "Просто ответ"
    assert not llm.script


async def test_invalid_tool_arguments_json(build_agent, fake_llm_factory) -> None:
    """Невалидный JSON в arguments: модель получает сообщение об ошибке и отвечает заново."""
    llm = fake_llm_factory(
        [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"id": "1", "function": {"name": "upper", "arguments": "не json"}}],
                },
                "finish_reason": "tool_calls",
            },
            {
                "message": {"content": "Исправился", "tool_calls": None},
                "finish_reason": "stop",
            },
        ]
    )

    answer = await build_agent(llm).run("привет", f"conv-{uuid.uuid4()}")

    assert answer == "Исправился"
    # ошибка дошла до модели как результат инструмента
    tool_messages = [m for m in llm.last_messages if m["role"] == "tool"]
    assert any("невалидный JSON" in m["content"] for m in tool_messages)


async def test_content_filter_returns_text(build_agent, fake_llm_factory) -> None:
    """При content_filter с текстом цикл возвращает этот текст, а не продолжает работу."""
    llm = fake_llm_factory(
        [
            {
                "message": {"content": "Частичный ответ", "tool_calls": None},
                "finish_reason": "content_filter",
            }
        ]
    )

    answer = await build_agent(llm).run("вопрос", f"conv-{uuid.uuid4()}")

    assert answer == "Частичный ответ"
    assert not llm.script


async def test_content_filter_empty_text(build_agent, fake_llm_factory) -> None:
    """При content_filter без текста возвращается заглушка."""
    llm = fake_llm_factory(
        [
            {
                "message": {"content": "", "tool_calls": None},
                "finish_reason": "content_filter",
            }
        ]
    )

    answer = await build_agent(llm).run("вопрос", f"conv-{uuid.uuid4()}")

    assert answer == "[Ответ заблокирован фильтром провайдера]"


async def test_max_round_exhausted(build_agent, fake_llm_factory) -> None:
    """Пустые ответы до исчерпания раундов: возвращается сообщение о провале."""
    empty_response = {
        "message": {"content": "", "tool_calls": None},
        "finish_reason": "length",
    }
    llm = fake_llm_factory([empty_response] * 3)  # ровно на max_round=3 кругов
    agent = build_agent(llm, loop=AgentLoop(max_round=3))

    answer = await agent.run("вопрос", f"conv-{uuid.uuid4()}")

    assert "Не удалось выполнить задачу" in answer
