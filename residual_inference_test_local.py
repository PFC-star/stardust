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
import os
import json
import tensor_parallel as tp
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
 


def run_residual_inference(input_for_export, modules, sequential_dependency_map, residual_dependency_map, split_size):
    # split model based on max number model split available
    # modules, sequential_dependency_map, residual_dependency_map, _ = model_sharding(model, split_size,
    #                                                                              transformer_model_option=True,
    #                                                                              residual_connection=True,
    #                                                                              debug_mod=True)

    # try sample inference with buffer mechanism
    buffer = InferenceBuffer(sequential_dependency_map, residual_dependency_map)
    begin_time = time.time()
    for index, submodule in enumerate(modules):
        # Special handling for submodule 0
        # print(f"\nResidual inference on submodule {index}.")
        if index == 0:  # handle the first submodule edge case
            current_inputs = input_for_export
            output = submodule.forward(current_inputs)
            buffer.update(index, forward_output=output)
            output_ = output[0]
        elif index == len(modules) - 1:  # handle the last submodule edge case
            # print(submodule) 
            # current_inputs = buffer.get_inputs_for_submodule(index)
            # for name, i in submodule.model.layers.named_modules():
            #     if '.' not in name:
            #         print(name, i)
            #         output = i(*current_inputs)
            #         output_ = output    
            #         next_token = torch.argmax(output_[:,-1,:], dim=-1)
            #         print(next_token)
            #         input()
            current_inputs = buffer.get_inputs_for_submodule(index)
            output = submodule.forward(*current_inputs)
            output_ = output
        else:
            current_inputs = buffer.get_inputs_for_submodule(index)
            for tensors in current_inputs:
                print(tensors.dtype)
            output = submodule.forward(*current_inputs)
            buffer.update(index, output)
        # 用最后一个token的到output的idx
        next_token = torch.argmax(output_[:,-1,:], dim=-1)
        print(f'Inference at submodel_{index} and the next token is {next_token}')
    end_time = time.time()
    print(f"\Residual inference time: {end_time - begin_time}")
    return next_token


def run_sequential_inference(input_for_export, model):
    split_size = max_split_size(model, transformer_model_option=True)
    modules, _ = model_sharding(model, split_size,
                             transformer_model_option=True,
                             residual_connection=False,
                             debug_mod=False)
    output = input_for_export
    begin_time = time.time()
    for idx in range(len(modules)):
        if idx == 0:
            output = modules[idx](output)
            output_ = output[0]
        else:
            output = modules[idx](*output)
            output_ = output
        next_token = torch.argmax(output_[:,-1,:], dim=-1)
        print(f'Inference at submodel_{idx} and the next token is {next_token}')
    end_time = time.time()
    print(f"\nSequential inference time: {end_time - begin_time}")
    next_token = torch.argmax(output[:,-1,:], dim=-1)
    return next_token


def run_residual_onnx_export(input_for_export, model, export_path, split_size, quantization_option):
    modules, sequential_dependency_map, residual_dependency_map, _ = model_sharding(model, split_size,
                                                                                 transformer_model_option=True,
                                                                                 residual_connection=True,
                                                                                 debug_mod=True)
    torch_to_onnx_residual(modules, input_for_export, export_path, sequential_dependency_map, residual_dependency_map,
                           quantization_option=quantization_option)


def run_sequential_onnx_export(input_for_export, model, export_path, split_size, quantization_option):
    modules, _ = model_sharding(model, split_size,
                             transformer_model_option=True,
                             residual_connection=False)
    torch_to_onnx(modules, input_for_export, export_path,
                           quantization_option=quantization_option)


def run_model_output_size(model, tokenizer, split_size):
    modules, sequential_dependency_map, residual_dependency_map, _ = model_sharding(model, split_size,
                                                                                 transformer_model_option=True,
                                                                                 residual_connection=True,
                                                                                 debug_mod=True)

    model_output_size(tokenizer,
                      "/workspace/LinguaLinked-Inference/onnx_model__/residual_module_test",
                      split_size,
                      quantization_option=True,
                      residual_connection=True,
                      sequential_dependency_map=sequential_dependency_map,
                      residual_dependency_map=residual_dependency_map)


def run_module_flop_test(model, tokenizer, split_size):
    modules, sequential_dependency_map, residual_dependency_map, _ = model_sharding(model, split_size,
                                                                                 transformer_model_option=True,
                                                                                 residual_connection=True,
                                                                                 debug_mod=True)
    get_module_flops(modules, tokenizer, sequential_dependency_map, residual_dependency_map)


if __name__ == "__main__":
    # torch.fx.wrap('len')
    #
    # Setup seed
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    #
    # # # ======== load model to cpu and inference
    # model_name = "../LLM_models/bloom-1b1"
    # model_name = "../LLM_models/bloom-560m"
    model_name = "/workspace/LLM_models/bloom-3b"

    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # start = time.time()
    # print(tokenizer.decode(
    #     model.generate(**tokenizer("什么是分布式大语言模型？", return_tensors="pt").to("cpu"), num_beams=5, max_new_tokens=100)[
    #         0])) # model.generate=> idx
    # end = time.time()
    # print(f"cpu inference time: {end-start}")

    # model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True,  attn_implementation="eager")
    # model = tp.tensor_parallel(model, ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    # start = time.time()
    # print(tokenizer.decode(
    #     model.generate(**tokenizer("什么是分布式大语言模型？", return_tensors="pt"), num_beams=5, max_new_tokens=100)[
    #         0])) # model.generate=> idx
    # end = time.time()
    # print(f"tensor parallel inference time: {end-start}")

    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto') # 以流水线形式分配到gpu卡中
    # start = time.time()
    # print(tokenizer.decode(
    #     model.generate(**tokenizer("什么是分布式大语言模型？", return_tensors="pt"), num_beams=5, max_new_tokens=100)[
    #         0])) # model.generate=> idx
    # end = time.time()
    # print(f"pipeline parallel inference time: {end-start}")

    # input()


    split_size = max_split_size(model, transformer_model_option=True)
    print("split_size:", split_size)
    split_size=2
    # split_size = 3
    

        
    input_for_export = tokenizer("什么是分布式大语言模型？", return_tensors="pt")["input_ids"]

    # output = run_sequential_inference(input_for_export, model)
    # print(tokenizer.decode(output))
    # input()
    modules, sequential_dependency_map, residual_dependency_map, _ = model_sharding(model, split_size,
                                                                                 transformer_model_option=True,
                                                                                 residual_connection=True,
                                                                                 debug_mod=True)
    # # auto-regressive - 30tokens
    for i in range(30):
        output = run_residual_inference(input_for_export, modules, sequential_dependency_map, residual_dependency_map, split_size)
        # output = run_sequential_inference(input_for_export, model)
        input_append = torch.tensor([[output]])
        input_for_export = torch.cat((input_for_export,input_append),dim=1)
        print(tokenizer.decode(output))
    
    # input()
    
    torch_to_onnx_residual(modules, input_for_export, "/workspace/ams-LinguaLinked-Inference/onnx_model__/residual_module_test", sequential_dependency_map, residual_dependency_map,
                           quantization_option=True)
    #
    with open('sdm.json', 'w') as f:
        json.dump(sequential_dependency_map, f)
    with open('rdm.json', 'w') as f:
        json.dump(residual_dependency_map, f)
    print(sequential_dependency_map, residual_dependency_map)

    # ==== load onnx_model and inference
    model_output_size(tokenizer,
                      "/workspace/ams-LinguaLinked-Inference/onnx_model__/residual_module_test",
                      split_size,

                      quantization_option=True,
                      residual_connection=True,
                      sequential_dependency_map=sequential_dependency_map,
                      residual_dependency_map=residual_dependency_map)



    # matrix = []
    # current_row = [1] * 126 + [0] * (3125-126)
    # matrix.append(current_row)
    # current_row = [0] * 126 + [1] * (3125-126)
    # matrix.append(current_row)
    # initial_module_arrangement = torch.tensor(matrix)
    #
    # modelcard = ModelCard("qwen2-7b", quantization_option=False)
    # # # for i in range
    # initial_module_arrangement = torch.tensor([[1, 126], [126, 3124]])
    # model_dirs = modelcard.prepare_model_to_send(module_arrangement=initial_module_arrangement)

    # modelcard = ModelCard("qwen2-7b", quantization_option=False)
    # # # for i in range
    # # initial_module_arrangement = torch.tensor([[1, 126], [126, 3124]])
    # model_dirs = modelcard.prepare_model_to_send(module_arrangement=initial_module_arrangement)



    # print("-----------------Test Optimizer Function----------------------")
    # modelcard = ModelCard("qwen2-7b", quantization_option=False)
    # mem_util, out_size_map, bytearray_path, flop_module_path, num_flop, module_flop_map, num_modules \
    #     = modelcard.prepare_optimization_info()
    # print(mem_util, out_size_map, bytearray_path, flop_module_path, num_flop, module_flop_map, num_modules)
    
    # num_devices = 2
    # ping_latency = np.array([[float("inf"), 91.865 / 1000],
    #                          [89.33 / 1000, float("inf")]])
    # bandwidths = np.array([[float("inf"), 12.1227],
    #                        [13.48, float("inf")]])
    # TotalMem = np.array([10000 * 1024, 8000 * 1024])
    # AvailMem = np.array([4000 * 1024, 1000 * 1024])
    # flop_speed = [3.31e10, 5.35e10]
    
    # ###### INFO REQUIRED FROM DEVICES TO SERVER ######
    
    # load_balancer = Optimizer(num_devices=num_devices, num_modules=num_modules)
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
    
    # mod_out_dir = modelcard.prepare_model_to_send(module_arrangement=initial_module_arrangement)



    
    


    




    

    # run_sequential_onnx_export(input_for_export, model, "./onnx_model__/residual_test_model", split_size=split_size,
    #                            quantization_option=False)

    # run_residual_onnx_export(input_for_export, model, "/workspace/LinguaLinked-Inference/onnx_model__/residual_module_test", split_size=split_size,
    #                        quantization_option=True)

    
    






