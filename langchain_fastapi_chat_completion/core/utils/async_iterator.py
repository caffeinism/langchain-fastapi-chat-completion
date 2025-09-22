from typing import Any, AsyncIterator


async def apreactivate(aiterator: AsyncIterator) -> AsyncIterator:
    aiterator = aiter(aiterator)
    prefetched = await anext(aiterator)
    return _apreactivate(prefetched, aiterator)


async def _apreactivate(prefetched: Any, aiterator: AsyncIterator) -> AsyncIterator:
    yield prefetched
    async for it in aiterator:
        yield it
