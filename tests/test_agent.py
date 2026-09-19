import uuid

from agent.core import AgentLoop, ChatResponse, ToolCall


async def test_tool_loop(build_agent, fake_llm_factory) -> None:
    """Полный цикл: LLM вызывает инструмент, получает результат, даёт финальный ответ."""
    llm = fake_llm_factory(
        [
            # раунд 1: модель просит вызвать инструмент
            ChatResponse(
                content="",
                tool_calls=[ToolCall(id="1", name="upper", arguments={"text": "привет"})],
                finish_reason="tool_calls",
            ),
            # раунд 2: финальный ответ на основе результата инструмента
            ChatResponse(content="Модель увидела: ПРИВЕТ", tool_calls=[], finish_reason="stop"),
        ]
    )

    answer, _ = await build_agent(llm).run("привет", f"conv-{uuid.uuid4()}")

    assert answer == "Модель увидела: ПРИВЕТ"
    assert not llm.script  # оба сценария израсходованы, цикл не сделал лишних раундов


async def test_plain_answer_no_tools(build_agent, fake_llm_factory) -> None:
    """Модель отвечает сразу текстом — цикл завершается без вызова инструментов."""
    llm = fake_llm_factory([ChatResponse(content="Просто ответ", tool_calls=[], finish_reason="stop")])

    answer, _ = await build_agent(llm).run("вопрос", f"conv-{uuid.uuid4()}")

    assert answer == "Просто ответ"
    assert not llm.script


async def test_invalid_tool_arguments(build_agent, fake_llm_factory) -> None:
    """Кривые аргументы: валидация отдаёт модели ошибку, та исправляется и отвечает."""
    llm = fake_llm_factory(
        [
            ChatResponse(
                content="",
                tool_calls=[ToolCall(id="1", name="upper", arguments={})],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="Исправился", tool_calls=[], finish_reason="stop"),
        ]
    )

    answer, _ = await build_agent(llm).run("привет", f"conv-{uuid.uuid4()}")

    assert answer == "Исправился"
    tool_messages = [m for m in llm.last_messages if m["role"] == "tool"]
    assert any("валидации" in m["content"] for m in tool_messages)


async def test_content_filter_returns_text(build_agent, fake_llm_factory) -> None:
    """При content_filter с текстом цикл возвращает этот текст, а не продолжает работу."""
    llm = fake_llm_factory([ChatResponse(content="Частичный ответ", tool_calls=[], finish_reason="content_filter")])

    answer, _ = await build_agent(llm).run("вопрос", f"conv-{uuid.uuid4()}")

    assert answer == "Частичный ответ"
    assert not llm.script


async def test_content_filter_empty_text(build_agent, fake_llm_factory) -> None:
    """При content_filter без текста возвращается заглушка."""
    llm = fake_llm_factory([ChatResponse(content="", tool_calls=[], finish_reason="content_filter")])

    answer, _ = await build_agent(llm).run("вопрос", f"conv-{uuid.uuid4()}")

    assert answer == "[Ответ заблокирован фильтром провайдера]"


async def test_max_round_exhausted(build_agent, fake_llm_factory) -> None:
    """Пустые ответы до исчерпания раундов: возвращается сообщение о провале."""
    empty_response = ChatResponse(content="", tool_calls=[], finish_reason="length")
    llm = fake_llm_factory([empty_response] * 3)  # ровно на max_round=3 кругов
    agent = build_agent(llm, loop=AgentLoop(max_rounds=3))

    answer, _ = await agent.run("вопрос", f"conv-{uuid.uuid4()}")

    assert "Не удалось выполнить задачу" in answer
