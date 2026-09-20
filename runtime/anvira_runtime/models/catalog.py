"""Model discovery: curated catalog + live Hugging Face search."""
from __future__ import annotations

import json
from importlib import resources
from typing import Any
from urllib.parse import quote

import httpx

from .compat import estimate_params_b

_QUANT_PREFERENCE = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0", "Q8_0", "Q5_K_S", "Q6_K", "IQ4_XS")


def load_catalog() -> list[dict[str, Any]]:
    raw = resources.files("anvira_runtime.models").joinpath("catalog.json").read_text(encoding="utf-8")
    out = []
    for m in json.loads(raw)["models"]:
        out.append({**m, "kind": "local", "size_approximate": True})
    return out


def pick_preferred_file(files: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the GGUF file to offer by default (Q4_K_M first, like Anvira's model library).

    Split shards other than 00001 are ignored (the entry shard stands for the set).
    """
    import re
    candidates = [f for f in files if f["name"].lower().endswith(".gguf")
                  and not re.search(r"-0000[2-9]-of-|-000[1-9]\d-of-", f["name"], re.I)]
    for quant in _QUANT_PREFERENCE:
        for f in candidates:
            if quant.lower() in f["name"].lower():
                return f
    return candidates[0] if candidates else None


class HuggingFaceClient:
    def __init__(self, endpoint: str = "https://huggingface.co", timeout: float = 20.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.endpoint = endpoint.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport,
                                         headers={"User-Agent": "anvira-runtime"}, follow_redirects=True)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        params = {"search": query, "filter": "gguf", "sort": "downloads", "direction": "-1", "limit": str(limit)}
        resp = await self._client.get(f"{self.endpoint}/api/models", params=params)
        resp.raise_for_status()
        out = []
        for item in resp.json():
            rid = item.get("id") or item.get("modelId")
            if not rid:
                continue
            out.append({
                "id": rid, "kind": "local", "name": rid.split("/")[-1], "author": rid.split("/")[0],
                "downloads": item.get("downloads"), "likes": item.get("likes"), "gated": bool(item.get("gated")),
                "params_b": estimate_params_b(rid, item.get("tags")), "installed": False,
                "source": {"type": "huggingface", "repo": rid},
            })
        return out

    async def files(self, repo: str) -> list[dict[str, Any]]:
        resp = await self._client.get(f"{self.endpoint}/api/models/{quote(repo, safe='/')}",
                                      params={"blobs": "true"})
        resp.raise_for_status()
        return [{"name": s["rfilename"], "size_bytes": s.get("size")}
                for s in resp.json().get("siblings", []) if s.get("rfilename")]

    def download_url(self, repo: str, filename: str) -> str:
        return f"{self.endpoint}/{quote(repo, safe='/')}/resolve/main/{quote(filename)}"
