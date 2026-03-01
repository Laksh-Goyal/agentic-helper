"""RAG-based tool registry backed by FAISS.

Embeds tool definitions (name, description, argument schema) using a local
sentence-transformer model and retrieves the top-k most relevant tools for
a given user query via FAISS similarity search.

Usage::

    from tools.registry import ToolRegistry
    registry = ToolRegistry(tools, persist_dir, model_name)
    relevant = registry.retrieve("what time is it?", top_k=3)
"""

import hashlib
import json
import os
from typing import Optional

import faiss
import numpy as np
from langchain_core.tools import BaseTool
from sentence_transformers import SentenceTransformer

_INDEX_FILE = "tools.faiss"
_META_FILE = "tools_meta.json"
_HASH_FILE = "tool_hash.txt"


class ToolRegistry:
    """Vector-indexed registry for LangChain tools.

    On first use (or when the tool set changes), it builds a FAISS index
    by embedding a rich text representation of each tool.  Subsequent
    calls reuse the persisted index.
    """

    def __init__(
        self,
        tools: list[BaseTool],
        persist_dir: str,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._tools = {t.name: t for t in tools}
        self._persist_dir = persist_dir
        self._model = SentenceTransformer(model_name)

        # Ordered list of tool names matching the FAISS index row order
        self._index_names: list[str] = []
        self._index: Optional[faiss.IndexFlatIP] = None

        self._ensure_index()

    # ── Public API ────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list[BaseTool]:
        """Return the *top_k* tools most relevant to *query*."""
        if not self._index_names:
            return list(self._tools.values())

        n_results = min(top_k, len(self._index_names))
        query_vec = self._model.encode([query], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype=np.float32)

        _, indices = self._index.search(query_vec, n_results)
        matched_names = [
            self._index_names[i]
            for i in indices[0]
            if 0 <= i < len(self._index_names)
        ]
        return [self._tools[n] for n in matched_names if n in self._tools]

    def get_all(self) -> list[BaseTool]:
        """Return every registered tool (fallback path)."""
        return list(self._tools.values())

    # ── Index management ──────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        """Build or reload the FAISS index, rebuilding if the tool set changed."""
        current_hash = self._compute_tools_hash()
        hash_path = os.path.join(self._persist_dir, _HASH_FILE)
        index_path = os.path.join(self._persist_dir, _INDEX_FILE)
        meta_path = os.path.join(self._persist_dir, _META_FILE)

        # Try to reuse existing index
        if (
            os.path.exists(hash_path)
            and os.path.exists(index_path)
            and os.path.exists(meta_path)
        ):
            stored_hash = open(hash_path, "r").read().strip()
            if stored_hash == current_hash:
                self._index = faiss.read_index(index_path)
                with open(meta_path, "r") as f:
                    self._index_names = json.load(f)
                return

        # ── Build fresh index ─────────────────────────────────────────────
        documents: list[str] = []
        names: list[str] = []
        for tool in self._tools.values():
            documents.append(self._tool_to_document(tool))
            names.append(tool.name)

        if not documents:
            return

        embeddings = self._model.encode(documents, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner-product on normalized vecs = cosine
        index.add(embeddings)

        # Persist
        faiss.write_index(index, index_path)
        with open(meta_path, "w") as f:
            json.dump(names, f)
        with open(hash_path, "w") as f:
            f.write(current_hash)

        self._index = index
        self._index_names = names

    @staticmethod
    def _tool_to_document(tool: BaseTool) -> str:
        """Create a rich text representation of a tool for embedding.

        Includes the tool name, full docstring/description, and a formatted
        list of its arguments with types and descriptions.
        """
        lines = [
            f"Tool: {tool.name}",
            f"Description: {tool.description}",
        ]

        # Extract argument schema if available
        schema = getattr(tool, "args_schema", None)
        if schema is not None:
            try:
                # Prefer Pydantic V2 API; fall back to V1
                if hasattr(schema, "model_json_schema"):
                    props = schema.model_json_schema().get("properties", {})
                else:
                    props = schema.schema().get("properties", {})
                if props:
                    arg_parts: list[str] = []
                    for arg_name, arg_info in props.items():
                        arg_type = arg_info.get("type", "any")
                        arg_desc = arg_info.get("description", "")
                        arg_parts.append(f"  {arg_name} ({arg_type}): {arg_desc}")
                    lines.append("Arguments:\n" + "\n".join(arg_parts))
            except Exception:
                pass  # schema extraction failed — use name + description only

        return "\n".join(lines)

    def _compute_tools_hash(self) -> str:
        """Deterministic hash of the current tool set for staleness checks."""
        descriptions = sorted(
            f"{t.name}:{t.description}" for t in self._tools.values()
        )
        blob = json.dumps(descriptions, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


# ── Module-level singleton ────────────────────────────────────────────────────

_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Return (and lazily create) the global ToolRegistry singleton."""
    global _registry
    if _registry is None:
        from agent import config
        from tools import get_all_tools

        _registry = ToolRegistry(
            tools=get_all_tools(),
            persist_dir=config.TOOL_INDEX_DIR,
            model_name=config.TOOL_EMBEDDING_MODEL,
        )
    return _registry
