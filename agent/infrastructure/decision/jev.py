"""
JevEngine - провайдер BaseDecisionEngine на Jev (TypeSafe, POST /v1/systemone).
"""

from httpx2 import AsyncClient

from agent.core.decision import BaseDecisionEngine, Choice, Decision, DecisionEngineError, Noul, Question, Score, State
from agent.infrastructure.llm._http import post_with_retries


JEV_URL = "https://api.typesafe.ai/v1/systemone"


class JevEngine(BaseDecisionEngine):
    """Реализация на Jev: state -> типизированные вопросы -> вероятности и увереность."""

    def __init__(
        self, http_client: AsyncClient, api_key: str, model: str = "jev-latest", base_url: str | None = None
    ) -> None:
        self.http_client = http_client
        self.api_key = api_key
        self.model = model
        self.base_url = base_url if base_url else JEV_URL

    async def decide(self, state: State, questions: dict[str, Question]) -> dict[str, Decision]:
        """Один batched-POST: все вопросы в одном запросе, ответы по тем же ключам."""
        data = await post_with_retries(
            self.http_client,
            self.base_url,
            json={
                "state": state,
                "model": self.model,
                "questions": {k: self._to_request(q) for k, q in questions.items()},
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        answers = data.get("answers", {})
        missing = questions.keys() - answers.keys()
        if missing:
            raise DecisionEngineError(f"Jev не ответил на: {sorted(missing)}")
        return {k: self._from_response(q, answers[k]) for k, q in questions.items()}

    def _to_request(self, question: Question) -> dict:
        if isinstance(question, Choice):
            return {"type": "choice", "instructions": question.instructions, "criteria": question.criteria}
        if isinstance(question, Score):
            return {"type": "score", "instructions": question.instructions, "criteria": question.levels}
        if isinstance(question, Noul):
            return {"type": "noul", "instructions": question.instructions}
        raise TypeError(f"неизвестный тип вопроса: {type(question).__name__}")

    @staticmethod
    def _from_response(question: Question, answer: dict) -> Decision:
        if isinstance(question, Choice):
            return Decision(
                value=answer["choice"], probabilities=answer.get("probabilities"), confidence=answer.get("confidence")
            )
        if isinstance(question, Score):
            return Decision(
                value=answer["score"], probabilities=answer.get("probabilities"), confidence=answer.get("confidence")
            )
        if isinstance(question, Noul):
            return Decision(value=answer["noul"])
        raise TypeError(f"неизвестный тип вопроса: {type(question).__name__}")
