from typing import Any, Dict

from llm_perf.core.scheduler import CoreScheduler
from llm_perf.backends.NPU.npu_engine import NpuEngine
from llm_perf.backends.NPU.npu_sampler import NpuSampler
from llm_perf.backends.NPU.npu_scheduler import NpuScheduler
from llm_perf.utils.logger import logger

def setup_scheduler(
    model_config: Dict[str, Any], 
    pad_token_id, max_batch_size, 
    **kwargs
) -> CoreScheduler:
    # create engine
    engine = NpuEngine(model_config, pad_token_id)

    # create sampler
    sampler = NpuSampler()

    # create scheduler
    scheduler = NpuScheduler(
        engine=engine, 
        sampler=sampler, 
        max_batch_size=max_batch_size
    )

    return scheduler