from functools import partial
from typing import Callable, Iterable

import networkx as nx
from lithops.executors import FunctionExecutor
from lithops.wait import ANY_COMPLETED
from rechunker.types import ParallelPipelines, PipelineExecutor
from six import reraise

from cubed.runtime.pipeline import already_computed
from cubed.runtime.types import DagExecutor

# Lithops represents delayed execution tasks as functions that require
# a FunctionExecutor.
Task = Callable[[FunctionExecutor], None]


class LithopsPipelineExecutor(PipelineExecutor[Task]):
    """An execution engine based on Lithops."""

    def pipelines_to_plan(self, pipelines: ParallelPipelines) -> Task:
        tasks = []
        for pipeline in pipelines:
            stage_tasks = []
            for stage in pipeline.stages:
                if stage.mappable is not None:
                    stage_func = build_stage_mappable_func(stage, pipeline.config)
                    stage_tasks.append(stage_func)
                else:
                    stage_func = build_stage_func(stage, pipeline.config)
                    stage_tasks.append(stage_func)

            # Stages for a single pipeline must be executed in series
            tasks.append(partial(_execute_in_series, stage_tasks))

        # TODO: execute tasks for different specs in parallel
        return partial(_execute_in_series, tasks)

    def execute_plan(self, plan: Task, **kwargs):
        with FunctionExecutor(**kwargs) as executor:
            plan(executor)


def map_as_completed(
    lithops_function_executor,
    map_function,
    map_iterdata,
    include_modules=[],
    max_failures=3,
):
    """
    Apply a function to items of an input list, yielding results as they are completed
    (which may be different to the input order).

    A generalisation of Lithops `map`, with retries, and relaxed return ordering.

    :param lithops_function_executor: The Lithops function executor to use.
    :param map_function: The function to map over the data.
    :param map_iterdata: An iterable of input data.
    :param include_modules: Modules to include.
    :param max_failures: The number of task failures to allow before raising an exception.

    :return: Function values as they are completed, not necessarily in the input order.
    """
    failures = 0

    inputs = map_iterdata
    futures_to_inputs = {}
    pending = []

    futures = lithops_function_executor.map(
        map_function,
        inputs,
        include_modules=include_modules,
    )
    futures_to_inputs.update({k: v for (k, v) in zip(futures, inputs)})
    pending.extend(futures)

    while pending:
        finished, pending = lithops_function_executor.wait(
            pending, throw_except=False, return_when=ANY_COMPLETED
        )

        errored = []
        for future in finished:
            if future.error:
                failures += 1
                if failures > max_failures:
                    # re-raise exception
                    # TODO: why does calling status not raise the exception?
                    future.status(throw_except=True)
                    reraise(*future._exception)
                errored.append(future)
            else:
                yield future.result()
        if errored:
            # rerun and add to pending
            inputs = [v for (fut, v) in futures_to_inputs.items() if fut in errored]
            # TODO: de-duplicate code from above
            futures = lithops_function_executor.map(
                map_function,
                inputs,
                include_modules=include_modules,
            )
            futures_to_inputs.update({k: v for (k, v) in zip(futures, inputs)})
            pending.extend(futures)


def build_stage_mappable_func(stage, config, callbacks=None):
    def sf(mappable):
        return stage.function(mappable, config=config)

    def stage_func(lithops_function_executor):
        for _ in map_as_completed(
            lithops_function_executor,
            sf,
            list(stage.mappable),
            include_modules=["cubed"],
        ):
            if callbacks is not None:
                [callback.on_task_end() for callback in callbacks]

    return stage_func


def build_stage_func(stage, config):
    def sf():
        return stage.function(config=config)

    def stage_func(lithops_function_executor):
        futures = lithops_function_executor.call_async(sf, ())
        lithops_function_executor.get_result(futures)

    return stage_func


def _execute_in_series(
    tasks: Iterable[Task], lithops_function_executor: FunctionExecutor
) -> None:
    for task in tasks:
        task(lithops_function_executor)


class LithopsDagExecutor(DagExecutor):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    # TODO: execute tasks for independent pipelines in parallel
    def execute_dag(self, dag, callbacks=None, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}
        with FunctionExecutor(**merged_kwargs) as executor:
            nodes = {n: d for (n, d) in dag.nodes(data=True)}
            for node in reversed(list(nx.topological_sort(dag))):
                if already_computed(nodes[node]):
                    continue
                pipeline = nodes[node]["pipeline"]

                for stage in pipeline.stages:
                    if stage.mappable is not None:
                        stage_func = build_stage_mappable_func(
                            stage, pipeline.config, callbacks=callbacks
                        )
                    else:
                        stage_func = build_stage_func(stage, pipeline.config)

                    # execute each stage in series
                    stage_func(executor)
