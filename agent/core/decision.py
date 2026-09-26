"""
Слой типизированных решений (System One): semantic-решений без генерации текста.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


Entry = str | dict | list | None
State = str | dict | list


@dataclass
class Choice:
    """
    Вопрос «выбери одно из набора»: маршрутизация, категоризация, гейтинг.

    criteria - rubric: ключ -> описание опции; описание читает провайдер.
    """

    criteria: dict[str, Entry]
    instructions: Entry = None


@dataclass
class Score:
    """
    Вопрос «оцени по упорядоченной шкале»: релевантность, качество, риск.

    levels - от низкого к высокему; value ответа может быть дробным (1.4).
    """

    levels: list[Entry]
    instructions: Entry = None


@dataclass
class Noul:
    """
    Вопрос «верно ли утверждение» с вероятностью 0..1: фильтрация, проверки.

    instructions - само утверждение.
    Ответ без probabilities/confidence - только число.
    """

    instructions: Entry


Question = Choice | Score | Noul


class DecisionEngineError(Exception):
    """Провайдер не выполнил контракт decide() - вызывающий может деградировать до прежнего поведения."""


@dataclass
class Decision:
    """
    Типизированный ответ: ключ choice / значение score / вероятность noul.

    probabilities — распределение по опциям или уровням (у noul нет);
    confidence — 0..1, вторая ось: «что делать» против «действовать ли вообще».
    """

    value: str | float
    probabilities: dict[str, float] | None = None
    confidence: float | None = None


class BaseDecisionEngine(ABC):
    """
    Провайдер типизированных решений. Опционален в каждой точке потребления:
    None -> точка деградирует до прежнего поведения. Включение - в композиции, не в core.
    """

    @abstractmethod
    async def decide(self, state: State, questions: dict[str, Question]) -> dict[str, Decision]:
        """
        Один batched-вызов: вопросы смешанных типов оцениваются параллельно и изолированно.

        Ключи questions сохраняются в ответе:
        decide(..., {"a": ..., "b": ...}) -> {"a": Decision, "b": Decision}.
        """
        ...
