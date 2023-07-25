import collections
import copy
import logging
import time
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from lithops.executors import FunctionExecutor
from lithops.wait import ALWAYS, ANY_COMPLETED
from networkx import MultiDiGraph

from cubed.core.array import Callback, Spec
from cubed.core.plan import visit_node_generations, visit_nodes
from cubed.runtime.backup import should_launch_backup
from cubed.runtime.executors.lithops_retries import (
    RetryingFuture,
    map_with_retries,
    wait_with_retries,
)
from cubed.runtime.types import DagExecutor
from cubed.runtime.utils import handle_callbacks

logger = logging.getLogger(__name__)


def run_func(input, func=None, config=None, name=None):
    result = func(input, config=config)
    return result


def map_unordered(
    lithops_function_executor: FunctionExecutor,
    group_map_functions: Sequence[Callable[..., Any]],
    group_map_iterdata: Sequence[
        Iterable[Union[List[Any], Tuple[Any, ...], Dict[str, Any]]]
    ],
    group_names: Sequence[str],
    include_modules: List[str] = [],
    timeout: Optional[int] = None,
    retries: int = 2,
    use_backups: bool = False,
    return_stats: bool = False,
    **kwargs,
) -> Iterator[Any]:
    """
    Apply a function to items of an input list, yielding results as they are completed
    (which may be different to the input order).

    A generalisation of Lithops `map`, with retries, and relaxed return ordering.

    :param lithops_function_executor: The Lithops function executor to use.
    :param group_map_functions: A sequence of functions to map over the data.
    :param group_map_iterdata: A sequence of iterables of input data.
    :param group_names: The names of the function/iterable groups.
    :param include_modules: Modules to include.
    :param retries: The number of times to retry a failed task before raising an exception.
    :param use_backups: Whether to launch backup tasks to mitigate against slow-running tasks.
    :param return_stats: Whether to return lithops stats.

    :return: Function values (and optionally stats) as they are completed, not necessarily in the input order.
    """
    return_when = ALWAYS if use_backups else ANY_COMPLETED

    group_name_to_function: Dict[str, Callable[..., Any]] = {}
    # backups are launched based on task start and end times for the group
    start_times: Dict[str, Dict[RetryingFuture, float]] = {}
    end_times: Dict[str, Dict[RetryingFuture, float]] = collections.defaultdict(dict)
    backups: Dict[RetryingFuture, RetryingFuture] = {}
    pending: List[RetryingFuture] = []
    group_name: str

    for map_function, map_iterdata, group_name in zip(
        group_map_functions, group_map_iterdata, group_names
    ):
        # can't use functools.partial here as we get an error in lithops
        # also, lithops extra_args doesn't work for this case
        partial_map_function = lambda x: map_function(x, **kwargs)
        group_name_to_function[group_name] = partial_map_function

        futures = map_with_retries(
            lithops_function_executor,
            partial_map_function,
            map_iterdata,
            timeout=timeout,
            include_modules=include_modules,
            retries=retries,
            group_name=group_name,
        )
        start_times[group_name] = {k: time.monotonic() for k in futures}
        pending.extend(futures)

    while pending:
        finished, pending = wait_with_retries(
            lithops_function_executor,
            pending,
            throw_except=False,
            return_when=return_when,
            show_progressbar=False,
        )
        for future in finished:
            if future.error:
                # if the task has a backup that is not done, or is done with no exception, then don't raise this exception
                backup = backups.get(future, None)
                if backup:
                    if not backup.done or not backup.error:
                        continue
                future.status(throw_except=True)
            group_name = future.group_name  # type: ignore[assignment]
            end_times[group_name][future] = time.monotonic()
            if return_stats:
                yield future.result(), standardise_lithops_stats(future)
            else:
                yield future.result()

            # remove any backup task
            if use_backups:
                backup = backups.get(future, None)
                if backup:
                    if backup in pending:
                        pending.remove(backup)
                    del backups[future]
                    del backups[backup]
                    backup.cancel()

        if use_backups:
            now = time.monotonic()
            for future in copy.copy(pending):
                group_name = future.group_name  # type: ignore[assignment]
                if future not in backups and should_launch_backup(
                    future, now, start_times[group_name], end_times[group_name]
                ):
                    input = future.input
                    logger.info("Running backup task for %s", input)
                    futures = map_with_retries(
                        lithops_function_executor,
                        group_name_to_function[group_name],
                        [input],
                        timeout=timeout,
                        include_modules=include_modules,
                        retries=0,  # don't retry backup tasks
                        group_name=group_name,
                    )
                    start_times[group_name].update(
                        {k: time.monotonic() for k in futures}
                    )
                    pending.extend(futures)
                    backup = futures[0]
                    backups[future] = backup
                    backups[backup] = future
            time.sleep(1)


def execute_dag(
    dag: MultiDiGraph,
    callbacks: Optional[Sequence[Callback]] = None,
    array_names: Optional[Sequence[str]] = None,
    resume: Optional[bool] = None,
    spec: Optional[Spec] = None,
    compute_arrays_in_parallel: Optional[bool] = None,
    **kwargs,
) -> None:
    use_backups = kwargs.pop("use_backups", False)
    allowed_mem = spec.allowed_mem if spec is not None else None
    with FunctionExecutor(**kwargs) as executor:
        runtime_memory_mb = executor.config[executor.backend].get(
            "runtime_memory", None
        )
        if runtime_memory_mb is not None and allowed_mem is not None:
            runtime_memory = runtime_memory_mb * 1_000_000
            if runtime_memory < allowed_mem:
                raise ValueError(
                    f"Runtime memory ({runtime_memory}) is less than allowed_mem ({allowed_mem})"
                )
        if not compute_arrays_in_parallel:
            for name, node in visit_nodes(dag, resume=resume):
                pipeline = node["pipeline"]
                for stage in pipeline.stages:
                    if stage.mappable is not None:
                        for _, stats in map_unordered(
                            executor,
                            [run_func],
                            [stage.mappable],
                            [name],
                            func=stage.function,
                            config=pipeline.config,
                            name=name,
                            use_backups=use_backups,
                            return_stats=True,
                        ):
                            handle_callbacks(callbacks, stats)
                    else:
                        raise NotImplementedError()
        else:
            for gen in visit_node_generations(dag, resume=resume):
                group_map_functions = []
                group_map_iterdata = []
                group_names = []
                for name, node in gen:
                    pipeline = node["pipeline"]
                    stage = pipeline.stages[0]  # assume one
                    f = partial(run_func, func=stage.function, config=pipeline.config)
                    group_map_functions.append(f)
                    group_map_iterdata.append(stage.mappable)
                    group_names.append(name)
                for _, stats in map_unordered(
                    executor,
                    group_map_functions,
                    group_map_iterdata,
                    group_names,
                    use_backups=use_backups,
                    return_stats=True,
                ):
                    handle_callbacks(callbacks, stats)


def standardise_lithops_stats(future: RetryingFuture) -> Dict[str, Any]:
    stats = future.stats
    return dict(
        array_name=future.group_name,
        task_create_tstamp=stats["host_job_create_tstamp"],
        function_start_tstamp=stats["worker_func_start_tstamp"],
        function_end_tstamp=stats["worker_func_end_tstamp"],
        task_result_tstamp=stats["host_status_done_tstamp"],
        peak_measured_mem_start=stats["worker_peak_memory_start"],
        peak_measured_mem_end=stats["worker_peak_memory_end"],
    )


class LithopsDagExecutor(DagExecutor):
    """An execution engine that uses Lithops."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        array_names: Optional[Sequence[str]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        **kwargs,
    ) -> None:
        merged_kwargs = {**self.kwargs, **kwargs}
        execute_dag(
            dag,
            callbacks=callbacks,
            array_names=array_names,
            resume=resume,
            spec=spec,
            **merged_kwargs,
        )
