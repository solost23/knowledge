"""Shared thread pools for background application tasks."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Literal, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")

_EXECUTORS = {
    "document_import": ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="document-import"
    ),
    "ask": ThreadPoolExecutor(max_workers=4, thread_name_prefix="ask-run"),
    "agent": ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-run"),
}


def submit_task(
    pool: Literal["document_import", "ask", "agent"],
    fn: Callable[_P, _T],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> Future[_T]:
    """Submit a task to its shared pool without blocking the caller."""
    return _EXECUTORS[pool].submit(fn, *args, **kwargs)
