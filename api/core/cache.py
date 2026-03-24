"""Redis cache — init/close and get/set helpers with JSON serialisation."""

import json
from typing import Any

import redis.asyncio as aioredis

from .config import settings

_redis: aioredis.Redis | None = None


async def init_redis():
    global _redis
    _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)


async def close_redis():
    if _redis:
        await _redis.aclose()


def get_redis() -> aioredis.Redis:
    if _redis is None:
        raise RuntimeError("Redis not initialised. Call init_redis() first.")
    return _redis


async def cache_get(key: str) -> Any | None:
    value = await get_redis().get(key)
    return json.loads(value) if value else None


async def cache_set(key: str, value: Any, ttl: int = 300):
    await get_redis().setex(key, ttl, json.dumps(value))


async def cache_delete(key: str):
    await get_redis().delete(key)


async def cache_delete_pattern(pattern: str):
    keys = await get_redis().keys(pattern)
    if keys:
        await get_redis().delete(*keys)
