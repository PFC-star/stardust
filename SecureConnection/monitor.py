import socket

import zmq
import time
import json
import numpy as np
import struct
import os
import sys
import threading
import copy
from collections import deque
from .root_server import send_model_file
import traceback

TIMEOUT = 1000

class Monitor:
    monitor_info_map = {}
    latency_list = []
    bandwidth_list = []
    memory_list = []
    flop_list = []
    receive_num = 0

    def __init__(self, monitor_receive_interval, monitor_port, devices, model_name, byte_array_path,\
                 flop_module_path, num_flop, runtime_option):
        self.monitor_receive_interval = monitor_receive_interval
        self.portNum = monitor_port
        self.root_device_len = len(devices)
        self.model_name = model_name
        self.byte_array_path = byte_array_path
        self.flop_module_path = flop_module_path
        self.num_flop = num_flop
        self.ip_graph_requested = []
        self.socket = None
        self.latency = None
        self.bandwidth = None
        self.total_memory = None
        self.avail_memory = None
        self.flop_speed = None
        self.ip_map_id = {}
        self.receive_monitor = True
        self.is_monitor_ready = threading.Event()
        self.all_data_ready = threading.Event()
        self.runtime_option = runtime_option
        self.record_time = 1
        self.devices = devices
        self.lock = threading.Lock() # Lock for synchronizing access to shared resources
    
    def get_monitor_info(self):

        # return an average value of every metric
        device_num = len(self.devices)

        print(f'\nself.latency_list: {self.latency_list}')

        latency_total_list = []
        for l in self.latency_list:
            latency_total_list.append(np.array(l))
        latency_3d = np.array(latency_total_list)

        print(f'latency_3d: {latency_3d}')

        latency_avg = np.zeros((device_num, device_num))
        for i in range(device_num):
            for j in range(device_num):
                if i != j:
                    tmp = 0
                    for list in latency_3d:
                        tmp += list[i][j]
                    latency_avg[i][j] = tmp / len(latency_3d)

        # latency_avg = np.mean(latency_3d)
        print(f'latency_avg: {latency_avg}')

        print(f'self.bandwidth_list: {self.bandwidth_list}')
        bandwidth_total_list = []
        for b in self.bandwidth_list:
            bandwidth_total_list.append(np.array(b))
        bandwidth_3d = np.array(bandwidth_total_list)
        bandwidth_avg = np.zeros((device_num, device_num))
        for i in range(device_num):
            for j in range(device_num):
                if i != j:
                    tmp = 0
                    for list in bandwidth_3d:
                        tmp += list[i][j]
                    bandwidth_avg[i][j] = tmp / len(bandwidth_3d)
        # bandwidth_avg = np.mean(bandwidth_3d)
        print(f'bandwidth_avg: {bandwidth_avg}')

        print(f'self.memory_list: {self.memory_list}')
        memory_total_list = []
        for m in self.memory_list:
            memory_total_list.append(np.array(m))
        memory_2d = np.array(memory_total_list)

        memory_avg = np.zeros(device_num)
        for i in range(device_num):
            tmp = 0
            for list in memory_2d:
                tmp += list[i]
            memory_avg[i] = tmp / len(memory_2d)
        print(f'memory_avg: {memory_avg}')

        print(f'self.flop_list: {self.flop_list}')
        flop_total_list = []
        for f in self.flop_list:
            flop_total_list.append(np.array(f))
        flop_2d = np.array(flop_total_list)
        flop_avg = np.zeros(device_num)
        for i in range(device_num):
            tmp = 0
            for list in flop_2d:
                tmp += list[i]
            flop_avg[i] = tmp / len(flop_2d)
        print(f'flop_avg: {flop_avg}')

        return (latency_avg, bandwidth_avg, self.total_memory, memory_avg, flop_avg)


        # return only one length metric
        # return (len(self.devices), self.latency, self.bandwidth, self.total_memory, self.avail_memory, self.flop_speed)
    
    def start(self):
        print("monitor start running")
        context = zmq.Context()
        self.socket = context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{self.portNum}")

        continue_listening = True
        last_received_time = time.time()
        device_set = set()
        print("monitor start listening")
        print(f"device set size: {self.root_device_len}")
        
        # 增加超时计数，用于限制重试次数
        timeout_counter = 0
        max_timeouts = 3  # 最多尝试3次
        
        while continue_listening:
            if self.socket.poll(30000):  # 30秒超时
                timeout_counter = 0  # 收到消息，重置超时计数
                print("monitor start listening")
                try:
                    message_parts = self.socket.recv_multipart()
                    
                    if len(message_parts) < 3:
                        print(f"警告: 收到不完整消息，只有 {len(message_parts)} 个部分")
                        continue
                    
                    identifier = message_parts[0]
                    action = message_parts[1]
                    msg_content = message_parts[2]
                    
                    # print(f"monitor identifier: {identifier.hex()}")
                    print(f"monitor action: {action.decode()}")
                    print(f"monitor msg_content: {msg_content.decode()}")
                    print("monitor message received")
                    print("\n-----------------------------------\n")

                    if action.decode() == "MonitorIP":
                        print("In MonitorIP  monitor waiting for all device to be connected...")
                        self.ip_graph_requested.append(identifier)
                        jsonObject = json.loads(msg_content.decode())
                        ip = jsonObject.get("ip")
                        device_set.add(ip)
                        role = jsonObject.get("role")
                        print(f"device set: {device_set}")
                    last_received_time = time.time()
                except Exception as e:
                    print(f"监控消息接收处理错误: {e}")
                    traceback.print_exc()
                    continue
            else:
                # 增加超时处理逻辑
                timeout_counter += 1
                print(f"警告: 监控超时 {timeout_counter}/{max_timeouts}，无法接收设备消息")
                
                if timeout_counter >= max_timeouts:
                    print("错误: 多次尝试后仍无法接收设备消息，继续处理...")
                    # 而不是直接抛出异常，设置监控就绪标志并退出循环
                    self.is_monitor_ready.set()
                    return  # 退出函数，不执行后续代码
                
                continue  # 继续循环，尝试再次接收消息

            # if time.time() - last_received_time > TIMEOUT:
            if len(device_set) == self.root_device_len:
                # print("monitor No new devices connected in the last", TIMEOUT, "seconds. Broadcasting message.")
                print("all devices are added to monitor system, check devices in monitor")
                print(self.devices)
                self.sendIPGraph()
                continue_listening = False
        
        device_num = len(self.devices)
        self.latency = np.zeros((device_num, device_num))
        self.bandwidth = np.zeros((device_num, device_num))
        self.total_memory = np.zeros(device_num)
        self.avail_memory = np.zeros(device_num)
        self.flop_speed = np.zeros(device_num)
        
        last_monitor_time = time.time()
        monitor_start_time = last_monitor_time
        self.send_monitor_signal(1) # continue

        try:
            monitor_ready_set = set()
            timeout_counter = 0  # 重置超时计数
            
            while self.receive_monitor:
                if self.socket.poll(30000):  # 30秒超时
                    timeout_counter = 0  # 收到消息，重置超时计数
                    print("monitor start receiving results from edges")
                    try:
                        message_parts = self.socket.recv_multipart()
                        
                        if len(message_parts) < 3:
                            print(f"警告: 收到不完整消息，只有 {len(message_parts)} 个部分")
                            continue
                        
                        identifier = message_parts[0]
                        action = message_parts[1]
                        msg_content = message_parts[2]
                        
                        print("monitor information received")
                        if action.decode() == "Monitor":
                            jsonObject = json.loads(msg_content.decode())
                            ip = jsonObject.get("ip")
                            latency_arr = jsonObject.get("latency")
                            latency_arr = json.loads(latency_arr)
                            bandwidth_arr = jsonObject.get("bandwidth")
                            bandwidth_arr = json.loads(bandwidth_arr)
                            memory = jsonObject.get("memory")
                            memory = json.loads(memory)
                            total_mem, avail_mem = memory[0], memory[1]
                            # total_mem = jsonObject.get("totalMemory")
                            # avail_mem = jsonObject.get("availableMemory")
                            flop = jsonObject.get("flop")

                            print('-----------monitor info coming--------------')
                            print(f'ip: {ip}')
                            print(f'latency:\n {latency_arr}')
                            print(f'bandwidth:\n {bandwidth_arr}')
                            print(f'total mem: {total_mem}')
                            print(f'available mem: {avail_mem}')
                            print('--------------------------------------------')

                            self.updateMonitorInfo(ip, latency_arr, bandwidth_arr, total_mem, avail_mem, flop)
                            monitor_ready_set.add(ip)
                    except Exception as e:
                        print(f"监控数据处理错误: {e}")
                        traceback.print_exc()
                        continue
                else:
                    # 增加超时处理逻辑
                    timeout_counter += 1
                    print(f"警告: 监控超时 {timeout_counter}/{max_timeouts}，无法接收设备响应")
                    
                    if timeout_counter >= max_timeouts:
                        print("错误: 多次尝试后仍无法接收设备响应，继续处理...")
                        # 设置监控就绪标志
                        self.is_monitor_ready.set()
                        break  # 退出循环

                if len(monitor_ready_set) == device_num and self.record_time == 0:
                    print("monitor info ready for model init")
                    self.is_monitor_ready.set()
                    if not self.runtime_option:
                        self.send_monitor_signal(0)
                        self.receive_monitor = False
                if len(monitor_ready_set) == device_num and time.time() - last_monitor_time > self.monitor_receive_interval:
                    self.latency_list.append(self.latency)
                    self.bandwidth_list.append(self.bandwidth)
                    temp_memory = []
                    for i in range(len(self.avail_memory)):
                        temp_memory.append(self.avail_memory[i])
                    self.memory_list.append(temp_memory)
                    temp_flop = []
                    for i in range(len(self.flop_speed)):
                        temp_flop.append(self.flop_speed[i])
                    self.flop_list.append(temp_flop)

                    print("monitor list update")
                    print(f'latency list length: {len(self.latency_list)}')
                    print(f'bandwidth list length: {len(self.bandwidth_list)}')
                    print(f'memory list length: {len(self.memory_list)}')
                    print(f'flop list length: {len(self.flop_list)}')
                    if self.record_time > 0:
                        self.record_time -= 1
                    monitor_ready_set.clear()
                    last_monitor_time = time.time()
                    # print(f'Monitor record time: {self.record_time}')
        except Exception as e:
            print(f"监控线程异常: {e}")
            traceback.print_exc()
            # 确保即使发生异常，也会设置监控就绪标志
            self.is_monitor_ready.set()

        # 修复socket关闭问题，使用self.socket
        try:
            if hasattr(self, 'socket') and self.socket:
                self.socket.close()
            if 'context' in locals() and context:
                context.term()
            print("监控资源已清理")
        except Exception as e:
            print(f"关闭监控资源时出错: {e}")
            traceback.print_exc()

    def updateMonitorInfo(self, ip, latency_arr, bandwidth_arr, total_mem, avail_mem, flop):
        index = self.ip_map_id[ip]
        self.latency[index] = latency_arr
        self.latency[index][index] = float("inf")
        self.bandwidth[:, index] = bandwidth_arr
        self.bandwidth[index][index] = float("inf")
        self.total_memory[index] = total_mem
        self.avail_memory[index] = avail_mem
        self.flop_speed[index] = flop
        print("-------------update monitor info---------------\n")
        print("latency:")
        print(self.latency)
        print("bandwidth:")
        print(self.bandwidth)
        print("total memory:")
        print(self.total_memory)
        print("available memory:")
        print(self.avail_memory)
        print("floop speed:")
        print(self.flop_speed)

        self.receive_num += 1
        # record all info
        if ip not in self.monitor_info_map:
            self.monitor_info_map[ip] = {"ip": ip, "latency": [copy.deepcopy(self.latency[index])],\
             "bandwidth": [], "avail_memory": [avail_mem],\
             "total_memory": [total_mem], "flop": [flop]}
        else:
            self.monitor_info_map[ip]["latency"].append(copy.deepcopy(self.latency[index]))
            # self.monitor_info_map[ip]["bandwidth"].append(copy.deepcopy(self.bandwidth[index]))
            self.monitor_info_map[ip]["avail_memory"].append(avail_mem)
            self.monitor_info_map[ip]["total_memory"].append(total_mem)
            self.monitor_info_map[ip]["flop"].append(flop)
        if self.receive_num == len(self.devices):
            self.receive_num = 0
            for ip in self.devices:
                i = self.ip_map_id[ip["ip"]]
                self.monitor_info_map[ip["ip"]]["bandwidth"].append(copy.deepcopy(self.bandwidth[i]))


    def send_monitor_signal(self, tag):
        signal = b"continue" if tag == 1 else b"stop"

        print("send monitor signal {} to devices".format(tag))
        for request_address in self.ip_graph_requested:
            self.socket.send_multipart([request_address, signal])

    def sendIPGraph(self):
        print("monitor send IP back")
        graph = []
        for index, d in enumerate(self.devices):
            graph.append(d["ip"])
            self.ip_map_id[d["ip"]] = index
        
        graph_to_send = ",".join(graph)
        print(f'graph_to_send = {graph_to_send}')
        # current_path = os.path.abspath(__file__)
        # dir_path = os.path.dirname(current_path)
        # base_path = os.path.dirname(dir_path)
        # path_for_bin = os.path.join(base_path, "onnx_model", "backup", "bloom560m_quantized_int8", "flop_byte_array.bin")
        # print(f'path for bin: {path_for_bin}')
        # path_for_onnx = os.path.join(base_path, "onnx_model", "backup", "bloom560m_quantized_int8", "flop_test_module.onnx")
        trans = False
        for request_address in self.ip_graph_requested:
            # print("Sending response to address:", request_address)
            # flop_num = 184610000
            data = struct.pack("<Q", self.num_flop)
            self.socket.send_multipart([request_address, graph_to_send.encode()])
            self.socket.send_multipart([request_address, data])
            # self.socket.send_multipart([request_address, str(trans.encode())])
            if trans:
                send_model_file(self.byte_array_path, self.socket, request_address, True)
                send_model_file(self.flop_module_path, self.socket, request_address, True)

            
            # print(f'current path is: {current_path}')
            # path_for_bin = "d:\UCI\phd\Distributed_DL\Dist-CPU-Learn\onnx_model\backup\bloom560m_quantized_int8"
            # send_model_file( ,self.socket, request_address)
            print("Sent IP graph to requesting device:", request_address)

    def print_monitor_info(self):
        for ip in self.devices:
            print(f'----------------{ip} monitor info------------------')
            print("latency")
            print(self.monitor_info_map[ip["ip"]]["latency"])
            print("bandwidth")
            print(self.monitor_info_map[ip["ip"]]["bandwidth"])
            print("avail_memory")
            print(self.monitor_info_map[ip["ip"]]["avail_memory"])
            print("total_memory")
            print(self.monitor_info_map[ip["ip"]]["total_memory"])
            print("flop")
            print(self.monitor_info_map[ip["ip"]]["flop"])