import os
from typing import Dict, List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from llm_perf.core.generation import GenerateRequest
from llm_perf.core.engine import CoreEngine
from llm_perf.backends.NPU.npu_process_messager import NPUMultiProcessMsgr
from llm_perf.utils.logger import logger


class NpuEngine(CoreEngine):
    
    class Packet(CoreEngine.Packet):
        def __init__(self, request: GenerateRequest):
            CoreEngine.Packet.__init__(self, request)

            self.generation_start_time = None

        def _is_finished(self) -> bool:
            return self.is_finished()

        @staticmethod
        def prepare_inputs(
            batch: List[CoreEngine.Packet],
            **kwargs
        ) -> Dict:
            model_config = kwargs["model_config"]
            pad_token_id = kwargs["pad_token_id"] 

            all_input_ids = []
            all_position_ids = []

            max_seq_len = -1
            for packet in batch:
                cur_id_len = len(packet.request.input_ids) + len(packet.generate_ids)
                max_seq_len = cur_id_len if cur_id_len > max_seq_len else max_seq_len

            for packet in batch:
                cur_id_len = len(packet.request.input_ids) + len(packet.generate_ids)
                pad_len = max_seq_len - cur_id_len
                # input_ids = (
                #     packet.request.input_ids + 
                #     packet.generate_ids + 
                #     [pad_token_id] * pad_len
                # )
                generate_type = ""
                if len(packet.generate_ids) == 0:
                    input_ids = (
                        packet.request.input_ids + 
                        packet.generate_ids + 
                        [pad_token_id] * pad_len
                    )
                    all_position_ids.append([i for i in range(max_seq_len)])
                    generate_type = "prefill"
                else:
                    # input_ids = ([packet.generate_ids[len(packet.generate_ids) - 1]])

                    input_ids = ([packet.generate_ids[len(packet.generate_ids) - 1]] + [pad_token_id] * (max_seq_len - 1))

                    # input_ids = (
                    #     packet.request.input_ids + 
                    #     packet.generate_ids + 
                    #     [pad_token_id] * pad_len
                    # )
                    all_position_ids.append([max_seq_len - 1] + [0] * (max_seq_len - 1))
                    generate_type = "decode"
                # print(f"input_ids ------ {input_ids}")
                all_input_ids.append(input_ids)
                # all_position_ids.append([i for i in range(max_seq_len)])

            # breakpoint()
            model_inputs = {
                "past_key_values": None, 
                "attention_mask": None, 
                "use_cache": None
            }
            model_inputs["input_ids"] = all_input_ids
            model_inputs["position_ids"] = all_position_ids
            model_inputs["generate_type"] = generate_type

            model_name = model_config['model_name']
            if model_name == 'chatglm2':
                model_inputs["return_last_logit"] = False
            return model_inputs



    def __init__(
        self, model_config, pad_token_id, 
        **kwarg
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.pad_token_id = pad_token_id
        
        # set up environ
        self.setup()
        
        # init multiprocessr msgr
        if self.world_size > 1:
            self.mlp_manager = NPUMultiProcessMsgr(
                self.local_rank, self.world_size, "MultiProcessMsgr"
            )
    
    def setup(self):
        # init distributed env if needed
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world_size > 1:
            torch.distributed.init_process_group(
                backend="nccl", 
                world_size=self.world_size, 
                rank=self.local_rank
            )

        # load model using base method
        self.model = NpuEngine.load_model(self.model_config, "NPU")


    def broadcast_inputs(self, *args):
        if self.world_size <= 1:
            return args
        
        if self.local_rank == 0:
            self.mlp_manager.broadcast(args)
            return args
        else:
            inputs = self.mlp_manager.receive()
            return inputs


    def prepare_inputs(self, batch: List[CoreEngine.Packet]) -> Dict:
        model_inputs = NpuEngine.Packet.prepare_inputs(
            batch, 
            model_config=self.model_config, 
            pad_token_id=self.pad_token_id
        )
        return model_inputs


    def do_inference(self, packets: List[CoreEngine.Packet]):
        # set device
        torch.cuda.set_device(self.local_rank)

        # prepare inputs for each process
        model_inputs = self.prepare_inputs(packets) if self.local_rank == 0 else None
        model_inputs = self.broadcast_inputs(model_inputs)[0]

        # convert inputs to torch tensor
        model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
        model_inputs["position_ids"] = torch.tensor(model_inputs["position_ids"])

        # cpu to cuda
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                model_inputs[k] = v.cuda()

        # model forward
        outputs = self.model(**model_inputs)

        # rank 0: return logits
        # other rank: return
        if self.local_rank == 0:
            input_logits = outputs.logits[..., :-1, :].contiguous()
            next_tokens_logits = outputs.logits[:, -1, :].contiguous()

            logger.debug(
                f"tensor shape: {outputs.logits.shape}\n"
                f"next tokens logits: {next_tokens_logits.shape}\n"
                f"input logits: {input_logits.shape}\n"
            )

            return {
                "input_logits": input_logits,
                "last_logits": next_tokens_logits,
            }