# import time
# import torch
# import numpy as np
# import random
# from transformers import (
#     AutoConfig,
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     AutoModelForSequenceClassification,
#     AutoModelForQuestionAnswering
# )

# from util.model_card import ModelCard
# from system_pipeline.model_split.utils import model_sharding, max_split_size, InferenceBuffer


# def run_pp_inference(prompt, modelcard, split_size, max_new_tokens=100):
#     ans = ""
#     model = modelcard.model
#     tokenizer = modelcard.tokenizer
#     input_for_export = tokenizer(prompt, return_tensors="pt")["input_ids"]

#     max_splitsize = max_split_size(model, transformer_model_option=True)
#     print("max_split_size:", max_splitsize)
#     if max_splitsize < split_size:
#         split_size = max_splitsize

#     # split model based on max number model split available
#     modules, sequential_dependency_map, residual_dependency_map, _ = model_sharding(model, split_size,
#                                                                                  transformer_model_option=True,
#                                                                                  residual_connection=True,
#                                                                                  debug_mod=True)
#     begin_time = time.time()
#     # # auto-regressive - per tokens
#     for i in range(max_new_tokens):
#         # try sample inference with buffer mechanism
#         buffer = InferenceBuffer(sequential_dependency_map, residual_dependency_map)

#         for index, submodule in enumerate(modules):
#             if index == 0:  # handle the first submodule edge case
#                 current_inputs = input_for_export
#                 output = submodule.forward(current_inputs)
#                 buffer.update(index, forward_output=output)
#                 output_ = output[0]
#             elif index == len(modules) - 1:  # handle the last submodule edge case
#                 current_inputs = buffer.get_inputs_for_submodule(index)
#                 output = submodule.forward(*current_inputs)
#                 output_ = output
#             else:
#                 current_inputs = buffer.get_inputs_for_submodule(index)
#                 for tensors in current_inputs:
#                     print(tensors.dtype)
#                 output = submodule.forward(*current_inputs)
#                 buffer.update(index, output)
#             # ## early exit on submodules
#             # next_token = torch.argmax(output_[:,-1,:], dim=-1)
#             # print(f'Inference at submodel_{index} and the next token is {next_token}')
#         # 用最后一个token的到output的idx
#         next_token = torch.argmax(output[:,-1,:], dim=-1)    
#         input_append = torch.tensor([[next_token]])
#         input_for_export = torch.cat((input_for_export,input_append),dim=1)
#         ans = ans + str(tokenizer.decode(next_token))
#     end_time = time.time()
#     print(f"\PP inference time on cpu: {end_time - begin_time}")
#     return ans, modules, sequential_dependency_map, residual_dependency_map





# if __name__ == "__main__":


#     # n_devices = 2

#     # modelcard = ModelCard("bloom3b", quantization_option=False)
#     # # ans = modelcard.generate_cpu_answer("什么是分布式大语言模型？",num_beams = 5, max_new_tokens = 100)
#     # # print(ans)
#     # prompt = "What is a distributed large language model?"

#     # ans, modules, sequential_dependency_map, residual_dependency_map = run_pp_inference(prompt, modelcard, n_devices, 30)
#     # print(ans)


#     # onnx_file_path = "/workspace/LinguaLinked-Inference/onnx_model_bloom/residual_module"
#     # input_for_export = modelcard.tokenizer(prompt, return_tensors="pt")["input_ids"]
#     # torch_to_onnx_residual(modules, input_for_export, onnx_file_path, sequential_dependency_map, residual_dependency_map,
#     #                        quantization_option=False)
    


from system_pipeline.model_split.model_split import ModelSplit
from system_pipeline.model_split.utils import model_sharding, max_split_size, InferenceBuffer, \
    get_receiver_seq_dependency_map, get_receiver_res_dependency_map, process_module_arrangement, \
    generate_fake_module_arrangement, get_module_flops
import torch
import numpy as np
import random
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering
)
import time
from system_pipeline.onnx_backend.onnx import torch_to_onnx_residual, model_output_size, device_module_assignment, torch_to_onnx
from util.model_card import ModelCard
from system_pipeline.onnx_backend.optimization import Optimizer
import onnx
from SecureConnection.monitor import Monitor
import threading

def run_residual_inference(input_for_export, model, split_size):
    # split model based on max number model split available
    modules, sequential_dependency_map, residual_dependency_map = model_sharding(model, split_size,
                                                                                 transformer_model_option=True,
                                                                                 residual_connection=True,
                                                                                 debug_mod=True)

    # try sample inference with buffer mechanism
    buffer = InferenceBuffer(sequential_dependency_map, residual_dependency_map)
    begin_time = time.time()
    for index, submodule in enumerate(modules):
        # Special handling for submodule 0
        print(f"\nResidual inference on submodule {index}.")
        if index == 0:  # handle the first submodule edge case
            current_inputs = input_for_export
            output = submodule.forward(current_inputs)
            buffer.update(index, forward_output=output)
        elif index == len(modules) - 1:  # handle the last submodule edge case
            current_inputs = buffer.get_inputs_for_submodule(index)
            output = submodule.forward(*current_inputs)
        else:
            current_inputs = buffer.get_inputs_for_submodule(index)
            for tensors in current_inputs:
                print(tensors.dtype)
            output = submodule.forward(*current_inputs)
            buffer.update(index, output)
    end_time = time.time()
    print(f"Residual inference time: {end_time - begin_time}")


def run_sequential_inference(input_for_export, model):
    split_size = max_split_size(model, transformer_model_option=True)
    modules = model_sharding(model, split_size,
                             transformer_model_option=True,
                             residual_connection=False,
                             debug_mod=False)
    output = input_for_export
    begin_time = time.time()
    for idx in range(len(modules)):
        if idx == 0:
            output = modules[idx](output)
        else:
            output = modules[idx](*output)
    end_time = time.time()
    print(f"\nSequential inference time: {end_time - begin_time}")


def run_residual_onnx_export(input_for_export, model, export_path, split_size, quantization_option):
    modules, sequential_dependency_map, residual_dependency_map = model_sharding(model, split_size,
                                                                                 transformer_model_option=True,
                                                                                 residual_connection=True,
                                                                                 debug_mod=True)
    torch_to_onnx_residual(modules, input_for_export, export_path, sequential_dependency_map, residual_dependency_map,
                           quantization_option=quantization_option)


def run_sequential_onnx_export(input_for_export, model, export_path, split_size, quantization_option):
    modules = model_sharding(model, split_size,
                             transformer_model_option=True,
                             residual_connection=False)
    torch_to_onnx(modules, input_for_export, export_path,
                           quantization_option=quantization_option)


def run_model_output_size(model, tokenizer, split_size):
    modules, sequential_dependency_map, residual_dependency_map = model_sharding(model, split_size,
                                                                                 transformer_model_option=True,
                                                                                 residual_connection=True,
                                                                                 debug_mod=True)

    model_output_size(tokenizer,
                      "/Users/junchenzhao/LinguaLinked/onnx_model",
                      split_size,
                      quantization_option=True,
                      residual_connection=True,
                      sequential_dependency_map=sequential_dependency_map,
                      residual_dependency_map=residual_dependency_map)


def run_module_flop_test(model, tokenizer, split_size):
    modules, sequential_dependency_map, residual_dependency_map = model_sharding(model, split_size,
                                                                                 transformer_model_option=True,
                                                                                 residual_connection=True,
                                                                                 debug_mod=True)
    get_module_flops(modules, tokenizer, sequential_dependency_map, residual_dependency_map)


if __name__ == "__main__":
    # model_name = "bigscience/bloom-560m"
    # torch.fx.wrap('len')
    #
    # Setup seed
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    #
    # # # load model to cpu
    model_name = "facebook/opt-125m"
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    split_size = max_split_size(model, transformer_model_option=True)
    
    
    input_for_export = tokenizer("I love distributed machine learning!", return_tensors="pt")["input_ids"]

    modules, sequential_dependency_map, residual_dependency_map, _ = model_sharding(model, split_size,
                                                                                 transformer_model_option=True,
                                                                                 residual_connection=True,
                                                                                 debug_mod=True)
    # # auto-regressive - 30tokens
    # for i in range(30):
    #     output = run_residual_inference(input_for_export, modules, sequential_dependency_map, residual_dependency_map, split_size)
    #     # output = run_sequential_inference(input_for_export, model)
    #     input_append = torch.tensor([[output]])
    #     input_for_export = torch.cat((input_for_export,input_append),dim=1)
    #     print(tokenizer.decode(output))

    

    modelcard = ModelCard("opt125m", quantization_option=False)
    # modelcard = ModelCard("bloom1b7", quantization_option=False)
    # mem_util, out_size_map, bytearray_path, flop_module_path, num_flop, module_flop_map, num_modules \
    #     = modelcard.prepare_optimization_info()



    ##### INFO REQUIRED FROM DEVICES TO SERVER (Monitor Part) ######
    

    

    # print("-----------------Test Optimizer Function----------------------")
    
    # num_devices = 2
    # ping_latency = np.array([[float("inf"), 91.865 / 1000],
    #                          [89.33 / 1000, float("inf")]])
    # bandwidths = np.array([[float("inf"), 12.1227],
    #                        [13.48, float("inf")]])
    # TotalMem = np.array([10 * 1024, 8 * 1024])
    # AvailMem = np.array([6 * 1024, 6 * 1024])
    # flop_speed = [3.31e10, 5.35e10]
    # mem_threshold = 0.1
    # TotalMem = [m * mem_threshold for m in TotalMem]
    # AvailMem = [m * mem_threshold for m in AvailMem]
    
    # ###### INFO REQUIRED FROM DEVICES TO SERVER ######
    
    # load_balancer = Optimizer(num_devices=num_devices, num_modules=2)
    # load_balancer.process_initial_info(num_flop=module_flop_map,
    #                                    flop_speed=flop_speed,
    #                                    ping_latency=ping_latency,
    #                                    bandwidths=bandwidths,
    #                                    m2m=out_size_map,
    #                                    model_size=mem_util,
    #                                    total_mem=TotalMem,
    #                                    ava_mem=AvailMem)
    # initial_module_arrangement = load_balancer.initial_module_arrangement()
    # overlapping_module_arrangement = load_balancer.dynamic_module_arrangement()
    
    # print("initial_module_arrangement")
    # print(initial_module_arrangement)
    # print("overlapping_module_arrangement")
    # print(overlapping_module_arrangement)
    initial_module_arrangement = torch.tensor([[1,255], [255,834]])
    
    
    mod_out_dir = modelcard.prepare_model_to_send(module_arrangement=initial_module_arrangement)
    
    






