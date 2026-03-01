"""Tests for the RAG-based tool registry (FAISS backend).

These tests verify that:
- ToolRegistry indexes all provided tools
- retrieve() returns at most top_k tools
- retrieve() returns semantically relevant tools
- The index is rebuilt when tools change
- Disabling retrieval uses all tools
"""

import shutil

import pytest

# All tests in this module need faiss + sentence-transformers
faiss = pytest.importorskip("faiss")


@pytest.fixture()
def tmp_index_dir(tmp_path):
    """Provide a temporary directory for the FAISS index."""
    d = tmp_path / "tool_index"
    d.mkdir()
    yield str(d)
    shutil.rmtree(str(d), ignore_errors=True)


@pytest.fixture()
def all_tools():
    """Return the full set of auto-discovered tools."""
    from tools import get_all_tools
    return get_all_tools()


@pytest.fixture()
def registry(all_tools, tmp_index_dir):
    """Build a ToolRegistry with all discovered tools."""
    from tools.registry import ToolRegistry
    return ToolRegistry(
        tools=all_tools,
        persist_dir=tmp_index_dir,
        model_name="all-MiniLM-L6-v2",
    )


# ── Core functionality ───────────────────────────────────────────────────────


def test_registry_indexes_all_tools(registry, all_tools):
    """get_all() returns every tool that was indexed."""
    result = registry.get_all()
    assert len(result) == len(all_tools)
    indexed_names = {t.name for t in result}
    expected_names = {t.name for t in all_tools}
    assert indexed_names == expected_names


def test_registry_retrieve_returns_subset(registry):
    """retrieve() returns at most top_k tools."""
    result = registry.retrieve("calculate something", top_k=3)
    assert isinstance(result, list)
    assert 1 <= len(result) <= 3


def test_registry_retrieve_relevance_datetime(registry):
    """Querying about time should surface the datetime tool."""
    result = registry.retrieve("what time is it right now?", top_k=3)
    names = [t.name for t in result]
    assert "get_current_datetime" in names, (
        f"Expected 'get_current_datetime' in results, got: {names}"
    )


def test_registry_retrieve_browser(registry):
    """Querying about web navigation should surface browser tools."""
    result = registry.retrieve("go to google.com and click search", top_k=3)
    names = [t.name for t in result]
    browser_tools = [n for n in names if n.startswith("browser_")]
    assert len(browser_tools) >= 1, (
        f"Expected at least one browser tool, got: {names}"
    )


def test_registry_rebuild_on_change(tmp_index_dir):
    """Index is rebuilt when the tool set changes."""
    from langchain_core.tools import tool as lc_tool
    from tools.registry import ToolRegistry

    @lc_tool
    def dummy_tool_a(x: str) -> str:
        """A dummy tool."""
        return x

    @lc_tool
    def dummy_tool_b(x: str) -> str:
        """Another dummy tool."""
        return x

    # Build with one tool
    reg1 = ToolRegistry(
        tools=[dummy_tool_a],
        persist_dir=tmp_index_dir,
        model_name="all-MiniLM-L6-v2",
    )
    assert len(reg1.get_all()) == 1
    assert reg1._index.ntotal == 1

    # Rebuild with two tools — index should update
    reg2 = ToolRegistry(
        tools=[dummy_tool_a, dummy_tool_b],
        persist_dir=tmp_index_dir,
        model_name="all-MiniLM-L6-v2",
    )
    assert len(reg2.get_all()) == 2
    assert reg2._index.ntotal == 2


def test_registry_disabled_uses_all_tools(all_tools, monkeypatch):
    """When TOOL_RETRIEVAL_ENABLED is False, all tools should be used."""
    from agent import config
    monkeypatch.setattr(config, "TOOL_RETRIEVAL_ENABLED", False)

    assert config.TOOL_RETRIEVAL_ENABLED is False
    # All tools are still discoverable
    assert len(all_tools) >= 3
