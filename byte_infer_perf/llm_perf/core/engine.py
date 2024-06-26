import os
import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List

import torch
from torch import distributed as dist
import deepspeed

from llm_perf.core.generation import GenerateConfig, GenerateRequest, GenerateResult
from llm_perf.core.generation import ResultQueue
from llm_perf.utils.logger import logger
from llm_perf.model_zoo.llama2 import LlamaModel, LlamaDecoderLayer


# get model impl from orig or vendor 
def get_model_impl(
    model_config: Dict[str, Any], 
    hardware_type: str
):
    # for example, "ChatGLMForConditionalGeneration"
    model_inferface = model_config["model_interface"]
    model_name = model_config["model_name"]

    # Get orig model
    spec = importlib.util.spec_from_file_location(
        model_name, f"llm_perf/model_zoo/{model_name.split('-')[0]}.py"
    )
    base_module_impl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_module_impl)

    module = importlib.import_module(
        f"llm_perf.model_zoo.llama2")
    orig_model = getattr(module, "LlamaForCausalLM", None)
    # orig_model = getattr(base_module_impl, model_inferface)

    # Get vendor model
    vendor_model_path = f"llm_perf/backends/{hardware_type}/model_impl"
    if not os.path.exists(vendor_model_path):
        logger.info(
            f"{vendor_model_path} not exists, {model_inferface} model select <ByteMLPerf base model>"
        )
        return orig_model

    vendor_model_impl = importlib.import_module(
        ".", package=vendor_model_path.replace("/", ".")
    )
    if not model_name in vendor_model_impl.__all__.keys():
        logger.info(
            f"can't find {model_name} in {vendor_model_path}/__init__, model select <ByteMLPerf base model>"
        )
        return orig_model

    vendor_model = vendor_model_impl.__all__[model_name]
    logger.info(
        f"find {model_inferface} in {vendor_model_path}, model select <Vendor model>"
    )
    return vendor_model


# multiprocess mssager
class MultiProcessMsgr(ABC):
    @abstractmethod
    def broadcast(self, obj):
        raise NotImplementedError

    @abstractmethod
    def receive(self):
        raise NotImplementedError





class PacketStatus(Enum):
    ERROR = -1
    PENDING = 0
    RUNNING = 1
    FINISH = 2



class CoreEngine(ABC):

    @dataclass
    class Packet:
        request: GenerateRequest
        state: PacketStatus
        generate_ids: List[int]

        def __init__(self, request: GenerateRequest):
            self.request = request
            self.result_queue = ResultQueue()
            self.state = PacketStatus.PENDING
            self.generate_ids = []
            self.exception = None

        def add_result(self, res: GenerateResult):
            self.generate_ids.append(res.token_id)
            self.result_queue.put(res)

        async def get_result(self) -> GenerateRequest:
            return await self.result_queue.get()
        
        def finish(self) -> None:
            self.state = PacketStatus.FINISH
            self.result_queue.put(None)

        def error(self) -> None:
            self.state == PacketStatus.ERROR

        def is_finished(self) -> bool:
            return self.state == PacketStatus.FINISH

        def return_q_empty(self) -> bool:
            return self.result_queue.empty()



    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def load_model(model_config, hardware_type):
        # get rank if needed
        if dist.is_initialized():
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "0"))
        # get model impl
        model_impl = get_model_impl(model_config, hardware_type)

        # load model
        torch.cuda.set_device(local_rank)
        model_name = model_config['model_name']
        if model_name == 'gpt2':
            pass
        elif model_name == 'chatglm':
            from llm_perf.model_zoo.chatglm import ChatGLMConfig
            model = model_impl.from_pretrained(
                model_config["model_path"], config=ChatGLMConfig(**model_config["network"])
            )
        elif model_name == 'chatglm2':
            from llm_perf.model_zoo.chatglm2 import ChatGLMConfig
            model = model_impl.from_pretrained(
                model_config["model_path"], config=ChatGLMConfig(**model_config["network"])
            )
        elif model_name == 'llama2':
            from llm_perf.model_zoo.llama2 import LlamaConfig
            model = model_impl.from_pretrained(
                model_config["model_path"], config=LlamaConfig(**model_config["network"])
            )
            # model = model_impl.from_pretrained(
            #     model_config["model_path"], low_cpu_mem_usage=True, torch_dtype=torch.float16
            # )
            # self.model = model_cls.from_pretrained(self.model_path, low_cpu_mem_usage=True, torch_dtype=self.dtype)
            # print(f"----------set device rank {torch.npu.current_device()}, {world_size}, {local_rank} ------------")
            deepspeed.init_distributed(dist_backend="hccl")
            model = deepspeed.init_inference(
                model=model,
                mp_size=world_size,
                dtype=torch.float16,
                replace_with_kernel_inject=False,
                injection_policy={LlamaDecoderLayer: ('self_attn.o_proj', 'mlp.down_proj')}
            )
            model = model.npu()
            # print(f"torch.npu.memory_stats() {torch.npu.memory_stats()}")
        else:
            raise ValueError(f'Unknown model name: {model_name}')
        
        model.eval()
        model.half().cuda()
        logger.info(f"cuda model {model_config['model_path']} loaded {model}")

        return model


    @abstractmethod
    def setup(self):
        """ init environ and load model
        """
        raise NotImplementedError


    @abstractmethod
    def broadcast_inputs(self, *args):
        """Broadcast args from root rank to other ranks
        """
        raise NotImplementedError


    @abstractmethod
    def prepare_inputs(self, packets: List[Packet]):
        """prepare inputs for current inference
        """
        raise NotImplementedError


    @abstractmethod
    def do_inference(self, packets: List[Packet]):
        """Real inference function, do inference and return logits

        Args:
            packets: batch packets of inference

        Return:
            inference results of batch packets
        """
        raise NotImplementedError
