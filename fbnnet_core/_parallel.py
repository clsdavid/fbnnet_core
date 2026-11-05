"""Process-pool helper shared by the parallel stages (cube, rule mining, time-series reconstruction).

Where the ``fork`` start method exists (Linux, macOS) the workers inherit the
parent's state copy-on-write and nothing but the task index and the result is pickled. Where it does
not (Windows) the pool uses ``spawn``: every worker runs an initializer with
picklable arguments and rebuilds its state from them. Scripts that use ``use_parallel=True`` must
then be importable without side effects, i.e. guard the entry point with
``if __name__ == "__main__":``.

The environment variable ``FBNNET_START_METHOD`` (``fork`` or ``spawn``) overrides the choice, which
is also how the spawn path is exercised on Linux.
"""
import logging
import multiprocessing
import os
from typing import Any, Callable, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

START_METHOD_ENV = "FBNNET_START_METHOD"


def start_method() -> Optional[str]:
    """'fork' when available, else 'spawn', else None; FBNNET_START_METHOD overrides when valid."""
    available = multiprocessing.get_all_start_methods()
    requested = os.environ.get(START_METHOD_ENV, "").strip().lower()
    if requested:
        if requested in ("fork", "spawn") and requested in available:
            return requested
        logger.warning("%s=%r is not usable here (available: %s); ignoring it.", START_METHOD_ENV, requested, available)
    for method in ("fork", "spawn"):
        if method in available:
            return method
    return None


def pool_map(
    worker: Callable[[Any], Any],
    items: Sequence[Any],
    n_workers: int,
    chunksize: Optional[int],
    init: Callable[..., None],
    fork_args: Sequence[Any],
    spawn_args: Sequence[Any],
    clear: Callable[[], None],
) -> List[Any]:
    """``[worker(item) for item in items]`` in a process pool.

    `worker` and `init` must be module-level functions. `init(*fork_args)` runs once in the parent
    before a fork pool is created (and `clear()` afterwards), so the arguments may be unpicklable
    objects. With ``spawn``, `init(*spawn_args)` runs in each worker and `spawn_args` must be
    picklable.
    """
    method = start_method()
    context = multiprocessing.get_context(method)
    if method == "fork":
        init(*fork_args)
        try:
            with context.Pool(processes=n_workers) as pool:
                return pool.map(worker, items, chunksize=chunksize)
        finally:
            clear()
    with context.Pool(processes=n_workers, initializer=init, initargs=tuple(spawn_args)) as pool:
        return pool.map(worker, items, chunksize=chunksize)
