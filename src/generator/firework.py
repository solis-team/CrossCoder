import os
import asyncio
import random
import traceback
from pathlib import Path
from typing import List
from openai import (
    AsyncOpenAI,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)


CODE_GEN_SYSTEM = "You are a helpful coding assistant. Keep your reasoning brief — at most 5 sentences — then immediately write the code. Do not over-explain or repeat yourself."

DEFAULT_TIMEOUT = 300

DEFAULT_MAX_RETRIES = 3
_RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)


def _load_env_once():
    if getattr(_load_env_once, "_done", False):
        return
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.is_file():
        _load_env_once._done = True
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass
    if not os.getenv("FIREWORK_API_KEY"):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if v and (v.startswith('"') and v.endswith('"') or v.startswith("'") and v.endswith("'")):
                        v = v[1:-1]
                    if k:
                        os.environ.setdefault(k, v)
    _load_env_once._done = True


def _make_client(timeout: int = DEFAULT_TIMEOUT) -> AsyncOpenAI:
    _load_env_once()
    return AsyncOpenAI(
        api_key=os.getenv("FIREWORK_API_KEY"),
        base_url=(os.getenv("FIREWORK_BASE_URL") or "https://api.fireworks.ai/inference/v1").rstrip("/"),
        timeout=timeout,
        max_retries=0,
    )


async def generate_one_async(
    prompt: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    client: AsyncOpenAI,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                print(f"[firework.py] WARNING: empty content. finish_reason={response.choices[0].finish_reason}")
            return content if content else ""
        except _RETRYABLE as e:
            if attempt < max_retries:
                base = 5 * (3 ** attempt)
                wait = base * (1.0 + random.uniform(-0.2, 0.2))
                print(
                    f"[firework.py] retry {attempt+1}/{max_retries} "
                    f"after {type(e).__name__}: {e}  sleeping {wait:.1f}s..."
                )
                await asyncio.sleep(wait)
                continue
            print(f"[firework.py] GIVING UP after {max_retries} retries: {type(e).__name__}: {e}")
            return ""
        except Exception as e:
            print(f"[firework.py] non-retryable error: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            return ""
    return ""


async def _run_batch(
    prompts: List[str],
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    batch_size: int,
    max_retries: int,
) -> List[str]:
    model = model or os.getenv("FIREWORK_MODEL")
    client = _make_client(timeout=timeout)
    all_responses: List[str] = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tasks = [
                generate_one_async(
                    prompt=p,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    client=client,
                    max_retries=max_retries,
                )
                for p in batch
            ]
            batch_responses = await asyncio.gather(*tasks)
            all_responses.extend(batch_responses)
    finally:
        try:
            await client.close()
        except Exception as e:
            print(f"[firework.py] warning: client.close() failed: {type(e).__name__}: {e}")
        await asyncio.sleep(0)
    return all_responses


def generate(
    prompts: List[str],
    model: str = None,
    system_prompt: str = CODE_GEN_SYSTEM,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> List[str]:
    model = model or os.getenv("FIREWORK_MODEL")
    return asyncio.run(
        _run_batch(
            prompts=prompts,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            batch_size=len(prompts),
            max_retries=max_retries,
        )
    )


def generate_batch(
    prompts: List[str],
    model: str = None,
    system_prompt: str = CODE_GEN_SYSTEM,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    timeout: int = DEFAULT_TIMEOUT,
    batch_size: int = 10,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> List[str]:
    model = model or os.getenv("FIREWORK_MODEL")
    return asyncio.run(
        _run_batch(
            prompts=prompts,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            batch_size=batch_size,
            max_retries=max_retries,
        )
    )
