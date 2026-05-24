"""Prefix-aware router: hash on the system-prompt prefix → stable replica."""
import hashlib

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel


REPLICAS = [
    "http://vllm-0:8000",
    "http://vllm-1:8000",
    "http://vllm-2:8000",
    "http://vllm-3:8000",
]


def route(prefix: str) -> str:
    """Consistent hash on prefix → replica index."""
    h = int(hashlib.md5(prefix.encode()).hexdigest()[:8], 16)
    return REPLICAS[h % len(REPLICAS)]


app = FastAPI()


class Req(BaseModel):
    prompt: str
    system: str = ""    # the cacheable prefix


@app.post("/v1/completions")
async def proxy(body: Req):
    backend = route(body.system)
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{backend}/v1/completions",
                          json={"prompt": body.system + body.prompt, "max_tokens": 200})
    return r.json()
