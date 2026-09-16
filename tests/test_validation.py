from agent.core.tools.registry import ToolRegistry
from agent.infrastructure.tools.fetch_url import FetchURLTool
from agent.infrastructure.tools.web_search import SearchWebTool


def test_valid_args_pass() -> None:
    registry = ToolRegistry([SearchWebTool(), FetchURLTool()])
    errors = registry.registry["search_web"].validate_params({"query": "rust"})
    assert errors == []


def test_missing_required_reported() -> None:
    tool = SearchWebTool()
    errors = tool.validate_params({})
    assert any("query" in e for e in errors)


def test_wrong_type_reported() -> None:
    errors = FetchURLTool().validate_params({"url": 123})
    assert errors


if __name__ == "__main__":
    test_valid_args_pass()
    test_missing_required_reported()
    test_wrong_type_reported()
    print("ok")
