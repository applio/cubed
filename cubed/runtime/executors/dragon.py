import dragon

import asyncio
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Sequence

from aiostream import stream
from networkx import MultiDiGraph

from cubed.runtime.pipeline import visit_node_generations, visit_nodes
from cubed.runtime.types import Callback, DagExecutor
from cubed.runtime.utils import (
    handle_callbacks,
    handle_operation_start_callbacks,
)
from cubed.spec import Spec

from .local import check_runtime_memory, pipeline_to_stream


class DragonExecutor(DagExecutor):
    """An execution engine that uses Dragon processes."""

    def __init__(self, **kwargs):
        self.kwargs = {**kwargs, **dict(retries=0)}

    @property
    def name(self) -> str:
        return "dragon"

    @staticmethod
    async def async_execute_dag(
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        compute_arrays_in_parallel: Optional[bool] = None,
        **kwargs,
    ) -> None:
        concurrent_executor: Executor
        max_workers = kwargs.pop("max_workers", os.cpu_count())
        if spec is not None:
            check_runtime_memory(spec, max_workers)
        max_tasks_per_child = kwargs.pop("max_tasks_per_child", None)
        context = multiprocessing.get_context("dragon")
        # max_tasks_per_child is only supported from Python 3.11
        if max_tasks_per_child is None:
            concurrent_executor = ProcessPoolExecutor(
                max_workers=max_workers, mp_context=context
            )
        else:
            concurrent_executor = ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=context,
                max_tasks_per_child=max_tasks_per_child,
            )
        try:
            if not compute_arrays_in_parallel:
                # run one pipeline at a time
                for name, node in visit_nodes(dag, resume=resume):
                    handle_operation_start_callbacks(callbacks, name)
                    st = pipeline_to_stream(
                        concurrent_executor, name, node["pipeline"], **kwargs
                    )
                    async with st.stream() as streamer:
                        async for _, stats in streamer:
                            handle_callbacks(callbacks, stats)
            else:
                for gen in visit_node_generations(dag, resume=resume):
                    # run pipelines in the same topological generation in parallel by merging their streams
                    streams = [
                        pipeline_to_stream(
                            concurrent_executor, name, node["pipeline"], **kwargs
                        )
                        for name, node in gen
                    ]
                    merged_stream = stream.merge(*streams)
                    async with merged_stream.stream() as streamer:
                        async for _, stats in streamer:
                            handle_callbacks(callbacks, stats)

        finally:
            # don't wait for any cancelled tasks
            concurrent_executor.shutdown(wait=False)

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        compute_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        merged_kwargs = {**self.kwargs, **kwargs}
        asyncio.run(
            self.async_execute_dag(
                dag,
                callbacks=callbacks,
                resume=resume,
                spec=spec,
                compute_id=compute_id,
                **merged_kwargs,
            )
        )
