import importlib
from typing import Any, Dict

from llm_perf.core.scheduler import CoreScheduler
from llm_perf.backends.NPU.npu_inferencer import NpuInferencer
from llm_perf.backends.NPU.npu_sampler import NpuSampler
from llm_perf.backends.NPU.npu_scheduler import NpuScheduler
from llm_perf.utils.logger import logger

def setup_scheduler(xpu_cfg) -> CoreScheduler:

    # get model impl
    hardware_type = xpu_cfg["hardware_type"]
    model_config = xpu_cfg["model_config"]
    model_name = model_config["model_name"]

    vendor_model_path = f"llm_perf/backends/{hardware_type}/model_impl"
    vendor_model_impl = importlib.import_module(
        ".", package=vendor_model_path.replace("/", ".")
    )
    vendor_model = vendor_model_impl.__all__[model_name]

    # create inferencer
    inferencer = NpuInferencer(vendor_model, xpu_cfg)

    # create sampler
    sampler = NpuSampler()

    # create scheduler
    scheduler = NpuScheduler(
        inferencer=inferencer, 
        sampler=sampler, 
        xpu_cfg=xpu_cfg
    )

    return scheduler
