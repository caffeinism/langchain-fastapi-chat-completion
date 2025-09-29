import asyncio
from typing import Any, AsyncIterator

NOPE = object()


async def _apreactivate(queue: asyncio.Queue, aiterator: AsyncIterator):
    try:
        async for it in aiterator:
            await queue.put(it)
        await queue.put(NOPE)
    except BaseException as e:
        queue.shutdown()
        raise e from None


async def yield_from_queue(prefetched: Any, queue: asyncio.Queue, task: asyncio.Task):
    yield prefetched
    try:
        while (it := await queue.get()) is not NOPE:
            yield it
    except asyncio.QueueShutDown:
        await task


async def apreactivate(aiterator: AsyncIterator) -> AsyncIterator:
    queue = asyncio.Queue()
    task = asyncio.create_task(_apreactivate(queue, aiterator))

    while True:
        try:
            async with asyncio.Timeout(1):
                prefetched = await queue.get()
                break
        except TimeoutError:
            if task.done():
                await task

    return yield_from_queue(prefetched, queue, task)
