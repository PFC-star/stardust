import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# os.environ["DS_ACCELERATOR"]='cpu'
import time
import zmq
from SecureConnection import root_server
from SecureConnection import server
from SecureConnection import monitor
import threading
import torch
import numpy as np
import heapq
import json
import os
from collections import deque
from util.model_card import available_models, ModelCard, retrieve_sending_dir, retrieve_sending_info, retrieve_file_cfg
from system_pipeline.onnx_backend.optimization import Optimizer
import traceback
import copy
import logging
import functools
import sys

monitor_receive_interval = 10  # set intervals for receiving monitor info from clients
monitor_port = "34567"  # set server port to receive monitor info
TIMEOUT =10 # Time to wait for new devices to connect to servers
MODEL_EXIST_ON_DEVICE = False  # set True if the model exists on the mobile device, will skip model creation and transmission
runtime_option = False  # set True if the load balance is runtime
split_size = 2
device_number =2
task = "Generation"
root_dir = os.path.dirname(os.path.abspath(__file__))
residual_connection_option = True

# 添加全局设备池和相关锁
all_devices_pool = deque()  # 全局设备池，存储所有已注册的设备
active_tasks = {}  # 格式: {task_id: {"devices": devices_list, "status": status}}
devices_pool_lock = threading.Lock()  # 设备池的线程锁

# 添加设备池管理类
class DevicePoolManager:
    """设备池管理器"""
    def __init__(self):
        self.lock = threading.Lock() # 确保线程安全
        self.device_pool = deque()            # 全部已注册活跃设备池（非工作设备）
        self.working_devices = deque()        # 工作设备池（初始阶段注册的设备）
        self.active_devices = {}              # {task_id: device_list} 当前活跃任务使用的设备
        self.failed_working_devices = deque() # 工作设备故障池
        self.failed_active_devices = deque()  # 活跃设备故障池
        self.task_counter = 0
        self.device_heartbeats = {}           # 记录设备最后心跳时间
        self.heartbeat_timeout = 15           # 心跳超时时间(秒)，从30秒减少到15秒
        self.heartbeat_check_interval = 5     # 心跳检查间隔(秒)，从10秒减少到5秒
        self.initialization_complete = False  # 标记是否完成初始化阶段
    
    def set_initialization_complete(self):
        """标记初始化阶段已完成，将当前设备池中的设备设为工作设备"""
        with self.lock:
            # 将当前所有设备从活跃设备池转移到工作设备池
            self.working_devices = deque(self.device_pool)
            # 清空活跃设备池，因为现在所有设备都转为工作设备
            self.device_pool.clear()
            self.initialization_complete = True
            print(f"初始化阶段完成！共有 {len(self.working_devices)} 个工作设备")
            # 打印所有工作设备的详细信息
            for i, device in enumerate(self.working_devices):
                device_id = device.get("device_id", "N/A")
                ip = device.get("ip", "N/A")
                role = device.get("role", "N/A")
                print(f"  工作设备 {i+1}: ID={device_id}, IP={ip}, 角色={role}")
            
            # 确保更新所有设备的状态
            print(f"当前状态: 工作设备: {len(self.working_devices)}个, 活跃设备: {len(self.device_pool)}个")
            print(f"          工作设备故障: {len(self.failed_working_devices)}个, 活跃设备故障: {len(self.failed_active_devices)}个")
    
    def register_device(self, device_info):
        """
        将设备注册到设备池
        如果是初始化阶段，设备会进入活跃设备池，之后被转移到工作设备池
        如果初始化已完成，设备仅进入活跃设备池
        """
        try:
            # 提取设备基本信息
            ip = device_info.get("ip", "unknown")
            role = device_info.get("role", "unknown")
            device_id = device_info.get("device_id", "")
            
            # 标识符，用于设备唯一性判断
            device_identifier = device_id or ip
            
            # 打印设备注册信息
            if device_id:
                print(f"注册设备，ID: {device_id}, IP: {ip}, 角色: {role}")
            else:
                print(f"注册设备，IP: {ip}, 角色: {role}")
            
            with self.lock:
                # 更新设备心跳
                self.update_device_heartbeat(device_info)
                
                # 构建设备字典
                device = {
                    "ip": ip,
                    "role": role,
                    "registration_time": time.time()
                }
                
                if device_id:
                    device["device_id"] = device_id
                
                print(f"设备数据已准备: {device}")
                
                # 处理设备注册
                device_updated = False
                
                try:
                    # 查找并更新现有设备（工作设备池）
                    for i, existing_device in enumerate(self.working_devices):
                        existing_id = existing_device.get("device_id") or existing_device.get("ip")
                        if existing_id == device_identifier:
                            self.working_devices[i] = device
                            device_updated = True
                            print(f"更新工作设备池中的设备: {device_identifier}")
                            break
                    
                    # 查找并更新现有设备（活跃设备池）
                    if not device_updated:
                        for i, existing_device in enumerate(self.device_pool):
                            existing_id = existing_device.get("device_id") or existing_device.get("ip")
                            if existing_id == device_identifier:
                                self.device_pool[i] = device
                                device_updated = True
                                print(f"更新活跃设备池中的设备: {device_identifier}")
                                break
                    
                    # 如果设备不存在于任何池中，则根据初始化状态添加到相应池
                    if not device_updated:
                        if self.initialization_complete:
                            # 初始化已完成，新设备只添加到活跃池
                            print(f"将设备添加到活跃设备池: {device_identifier}")
                            self.device_pool.append(device)
                            print(f"添加完成，当前活跃设备池大小: {len(self.device_pool)}")
                        else:
                            # 初始化阶段，设备添加到活跃池，之后会被转移到工作池
                            # 注意: 设备最初添加到活跃池(self.device_pool)，在初始化完成时，会被set_initialization_complete转移到工作池
                            print(f"将设备添加到初始化中的活跃设备池: {device_identifier}（稍后会转移到工作设备池）")
                            self.device_pool.append(device)
                            print(f"添加完成，当前活跃设备池大小: {len(self.device_pool)}")
                    
                    # 兼容旧代码：将设备添加到全局设备列表
                    try:
                        global devices
                        with devices_pool_lock:
                            # 检查设备是否已在全局设备列表中
                            exists_in_global = False
                            for d in devices:
                                if (device_id and d.get("device_id") == device_id) or (not device_id and d.get("ip") == ip):
                                    exists_in_global = True
                                    # 更新现有设备信息
                                    d.update(device)
                                    break
                            
                            # 如果不存在于全局列表，则添加
                            if not exists_in_global:
                                devices.append(device.copy())
                                print(f"设备已添加到全局设备列表，当前共有 {len(devices)} 个设备")
                    except Exception as e:
                        print(f"更新全局设备列表时出错: {str(e)}")
                    
                except Exception as e:
                    print(f"设备池更新时出错: {str(e)}")
                    traceback.print_exc()
                
                # 打印当前设备池状态
                working_count = len(self.working_devices)
                active_count = len(self.device_pool)
                working_failure_count = len(self.failed_working_devices)
                active_failure_count = len(self.failed_active_devices)
                
                print(f"设备池状态更新:")
                print(f"  工作设备: {working_count}个, 工作设备故障: {working_failure_count}个")
                print(f"  活跃设备: {active_count}个, 活跃设备故障: {active_failure_count}个")
                print(f"  初始化阶段: {'否' if self.initialization_complete else '是'}")
                
            return True
        except Exception as e:
            print(f"注册设备出错: {str(e)}")
            traceback.print_exc()
            return False
    
    def update_device_heartbeat(self, device_info):
        """更新设备心跳时间"""
        device_id = device_info.get("device_id") or device_info.get("ip", "unknown")
        
        with self.lock:
            # 直接存储当前时间作为最后接收时间
            self.device_heartbeats[device_id] = time.time()
            
            # 如果是请求消息中的心跳更新，打印一个简单的日志
            is_heartbeat_msg = device_info.get("message_type") == "Heartbeat"
            if is_heartbeat_msg:
                print(f"收到设备 {device_id} 的心跳消息，更新时间戳")
            
            return True
    
    def check_and_recover_failed_device(self, device_id):
        """
        检查设备是否在故障池中，如果在则尝试恢复
        """
        # 从工作设备故障池中恢复
        for i, device in enumerate(list(self.failed_working_devices)):
            if device.get("device_id") == device_id or device.get("ip") == device_id:
                recovered_device = self.failed_working_devices.pop(i)
                print(f"设备 {device_id} 发送了心跳，从工作设备故障池中恢复")
                self.working_devices.append(recovered_device)
                print(f"设备池状态更新: 工作:{len(self.working_devices)}, 工作故障:{len(self.failed_working_devices)}")
                return True
        
        # 从活跃设备故障池中恢复
        for i, device in enumerate(list(self.failed_active_devices)):
            if device.get("device_id") == device_id or device.get("ip") == device_id:
                recovered_device = self.failed_active_devices.pop(i)
                print(f"设备 {device_id} 发送了心跳，从活跃设备故障池中恢复")
                self.device_pool.append(recovered_device)
                print(f"设备池状态更新: 活跃:{len(self.device_pool)}, 活跃故障:{len(self.failed_active_devices)}")
                return True
        
        return False

    def handle_working_device_failure(self, device):
        """
        处理工作设备故障
        从工作设备池中移除设备，添加到故障列表
        """
        device_id = device.get("device_id") or device.get("ip", "unknown")
        
        # 检查设备是否已经在故障池中，避免重复处理
        for failed_device in self.failed_working_devices:
            if (failed_device.get("device_id") == device.get("device_id") or 
                failed_device.get("ip") == device.get("ip")):
                print(f"工作设备 {device_id} 已在故障池中，跳过处理")
                return
        
        # 从工作设备池移除
        with self.lock:
            for i, d in enumerate(list(self.working_devices)):
                if (d.get("device_id") == device.get("device_id") or 
                    d.get("ip") == device.get("ip")):
                    self.working_devices.pop(i)
                    break
        
        # 添加到故障池中 (使用设备的深拷贝避免引用问题)
        device_copy = copy.deepcopy(device)
        with self.lock:
            self.failed_working_devices.append(device_copy)
        
        # 打印设备状态
        print(f"工作设备 {device_id} 故障处理完成，已从工作池移除并添加到故障池")
        print(f"设备池状态: 工作:{len(self.working_devices)}, 工作故障:{len(self.failed_working_devices)}")

    def handle_active_device_failure(self, device):
        """
        处理活跃设备故障
        从活跃设备池中移除设备，添加到故障列表
        """
        device_id = device.get("device_id") or device.get("ip", "unknown")
        
        # 检查设备是否已经在故障池中，避免重复处理
        for failed_device in self.failed_active_devices:
            if (failed_device.get("device_id") == device.get("device_id") or 
                failed_device.get("ip") == device.get("ip")):
                print(f"活跃设备 {device_id} 已在故障池中，跳过处理")
                return
        
        # 从活跃设备池移除
        with self.lock:
            for i, d in enumerate(list(self.device_pool)):
                if (d.get("device_id") == device.get("device_id") or 
                    d.get("ip") == device.get("ip")):
                    self.device_pool.pop(i)
                    break
        
        # 添加到故障池中 (使用设备的深拷贝避免引用问题)
        device_copy = copy.deepcopy(device)
        with self.lock:
            self.failed_active_devices.append(device_copy)
        
        # 打印设备状态
        print(f"活跃设备 {device_id} 故障处理完成，已从活跃池移除并添加到故障池")
        print(f"设备池状态: 活跃:{len(self.device_pool)}, 活跃故障:{len(self.failed_active_devices)}")
    
    def handle_device_failure(self, device):
        """处理设备故障（兼容旧代码）"""
        with self.lock:
            # 检查设备是工作设备还是活跃设备
            if device in self.working_devices:
                self.handle_working_device_failure(device)
            elif device in self.device_pool:
                self.handle_active_device_failure(device)
            else:
                device_id = device.get("device_id") or device["ip"]
                print(f"警告: 设备 {device_id} 不在任何设备池中，无法处理故障")
    
    def get_failed_devices(self):
        """获取所有故障设备列表（包括工作设备和活跃设备）"""
        with self.lock:
            return list(self.failed_working_devices) + list(self.failed_active_devices)
    
    def get_failed_working_devices(self):
        """获取工作设备故障列表"""
        with self.lock:
            return list(self.failed_working_devices)
    
    def get_failed_active_devices(self):
        """获取活跃设备故障列表"""
        with self.lock:
            return list(self.failed_active_devices)
    
    def get_device_status(self, device_id):
        """获取设备状态"""
        with self.lock:
            # 检查是否在工作设备故障池中
            for device in self.failed_working_devices:
                if (device.get("device_id") == device_id) or (device["ip"] == device_id):
                    return {
                        "status": "failed_working",
                        "failure_time": device.get("failure_time"),
                        "failure_reason": device.get("failure_reason")
                    }
            
            # 检查是否在活跃设备故障池中
            for device in self.failed_active_devices:
                if (device.get("device_id") == device_id) or (device["ip"] == device_id):
                    return {
                        "status": "failed_active",
                        "failure_time": device.get("failure_time"),
                        "failure_reason": device.get("failure_reason")
                    }
            
            # 检查是否在工作设备池中
            for device in self.working_devices:
                if (device.get("device_id") == device_id) or (device["ip"] == device_id):
                    last_heartbeat = self.device_heartbeats.get(device_id, 0)
                    return {
                        "status": "working",
                        "last_heartbeat": last_heartbeat,
                        "role": device.get("role")
                    }
            
            # 检查是否在活跃设备池中
            for device in self.device_pool:
                if (device.get("device_id") == device_id) or (device["ip"] == device_id):
                    last_heartbeat = self.device_heartbeats.get(device_id, 0)
                    return {
                        "status": "active",
                        "last_heartbeat": last_heartbeat,
                        "role": device.get("role")
                    }
            
            return {"status": "unknown"}
    
    def check_device_pool_integrity(self):
        """检查设备池中是否有重复项"""
        with self.lock:
            device_ids = set()
            ip_without_id = set()
            duplicates = []
            
            for device in self.device_pool:
                if "device_id" in device:
                    if device["device_id"] in device_ids:
                        duplicates.append(f"Duplicate device_id: {device['device_id']}")
                    device_ids.add(device["device_id"])
                else:
                    if device["ip"] in ip_without_id:
                        duplicates.append(f"Duplicate IP without device_id: {device['ip']}")
                    ip_without_id.add(device["ip"])
            
            if duplicates:
                print("WARNING: Found duplicates in device pool:")
                for dup in duplicates:
                    print(f"  - {dup}")
                return False
            return True
    
    def get_all_devices(self):
        """获取所有已注册设备"""
        with self.lock:
            return list(self.device_pool)
    
    def get_working_devices(self):
        """获取所有工作设备"""
        with self.lock:
            return list(self.working_devices)
    
    def get_device_count(self):
        """获取设备总数"""
        with self.lock:
            return len(self.device_pool)
    
    def get_working_device_count(self):
        """获取工作设备总数"""
        with self.lock:
            return len(self.working_devices)
    
    def get_active_task_count(self):
        """获取活跃任务数量"""
        with self.lock:
            return len(self.active_devices)

    def get_main_thread_device_count(self):
        """获取主线程中的设备数量（非活跃设备）"""
        with self.lock:
            active_device_count = sum(len(devices) for devices in self.active_devices.values())
            return len(self.working_devices) - active_device_count
    
    def get_available_devices(self, required_count=None, working_only=False):
        """获取可用的设备列表
           working_only=True 时只返回工作设备池中的设备
        """
        with self.lock:
            # 找出未被分配给活跃任务的设备
            busy_devices = set()
            for devices in self.active_devices.values():
                for device in devices:
                    # 使用设备ID或IP作为标识
                    if "device_id" in device:
                        busy_devices.add(device["device_id"])
                    else:
                        busy_devices.add(device["ip"])
            
            # 选择设备池
            device_pool = self.working_devices if working_only else self.device_pool
            
            available = []
            for d in device_pool:
                if "device_id" in d and d["device_id"] in busy_devices:
                    continue
                if "device_id" not in d and d["ip"] in busy_devices:
                    continue
                available.append(d)
            
            if required_count is not None and len(available) < required_count:
                print(f"警告: 只有 {len(available)} 个可用设备, 但需要 {required_count} 个")
                return None
            
            return available
    
    def allocate_devices_for_task(self, device_count, task_id=None, working_only=True):
        """为任务分配设备, working_only=True 时只从工作设备池中分配"""
        with self.lock:
            available = self.get_available_devices(working_only=working_only)
            
            if len(available) < device_count:
                print(f"没有足够的可用设备。需要: {device_count}, 可用: {len(available)}")
                return None
            
            # 分配设备
            allocated = []
            header_allocated = False
            
            # 优先分配header设备
            for device in available:
                if not header_allocated and device["role"] == "header":
                    allocated.append(device)
                    header_allocated = True
                    if len(allocated) == device_count:
                        break
            
            # 分配其他设备
            for device in available:
                if device not in allocated:
                    allocated.append(device)
                    if len(allocated) == device_count:
                        break
            
            # 如果没有指定task_id，创建新的
            if task_id is None:
                self.task_counter += 1
                task_id = f"task_{self.task_counter}"
            
            # 记录任务使用的设备
            self.active_devices[task_id] = allocated
            
            return task_id, allocated
    
    def release_task_devices(self, task_id):
        """释放任务使用的设备"""
        with self.lock:
            if task_id in self.active_devices:
                del self.active_devices[task_id]
                print(f"Released devices for task {task_id}")
                return True
            return False
    
    def get_task_devices(self, task_id):
        """获取指定任务的设备列表"""
        with self.lock:
            return self.active_devices.get(task_id, [])

    def is_working_device(self, device):
        """检查设备是否为工作设备"""
        with self.lock:
            device_id = device.get("device_id")
            ip = device.get("ip")
            
            for d in self.working_devices:
                d_id = d.get("device_id")
                d_ip = d.get("ip")
                
                # 如果有设备ID，使用设备ID比较；否则使用IP比较
                if (device_id and d_id and device_id == d_id) or (ip and d_ip and ip == d_ip):
                    return True
            
            return False

    def check_device_heartbeats(self):
        """检查所有设备的心跳状态，将超时的设备移到故障池"""
        current_time = time.time()
        found_failure = False
        working_failures = []
        active_failures = []

        # 检查工作设备心跳
        with self.lock:
            for device in list(self.working_devices):
                device_id = device.get("device_id") or device.get("ip", "unknown")
                
                # 初始化心跳记录 (如果是新设备)
                if device_id not in self.device_heartbeats:
                    self.device_heartbeats[device_id] = current_time
                    print(f"初始化工作设备 {device_id} 的心跳记录")
                    continue
                    
                # 检查心跳超时
                last_received_time = self.device_heartbeats.get(device_id, 0)
                time_since_last_heartbeat = current_time - last_received_time
                
                if time_since_last_heartbeat > self.heartbeat_timeout:
                    if not found_failure:
                        print(f"\n====== 心跳检查 {time.strftime('%H:%M:%S')} - 发现故障 ======")
                        found_failure = True
                    
                    # 收集故障设备 (稍后处理)
                    working_failures.append(device)
                    print(f"工作设备 {device_id} 心跳超时: {time_since_last_heartbeat:.1f}秒无响应")

        # 检查活跃设备心跳
        with self.lock:
            for device in list(self.device_pool):
                device_id = device.get("device_id") or device.get("ip", "unknown")
                
                # 初始化心跳记录 (如果是新设备)
                if device_id not in self.device_heartbeats:
                    self.device_heartbeats[device_id] = current_time
                    print(f"初始化活跃设备 {device_id} 的心跳记录")
                    continue
                    
                # 检查心跳超时
                last_received_time = self.device_heartbeats.get(device_id, 0)
                time_since_last_heartbeat = current_time - last_received_time
                
                if time_since_last_heartbeat > self.heartbeat_timeout:
                    if not found_failure:
                        print(f"\n====== 心跳检查 {time.strftime('%H:%M:%S')} - 发现故障 ======")
                        found_failure = True
                    
                    # 收集故障设备 (稍后处理)
                    active_failures.append(device)
                    print(f"活跃设备 {device_id} 心跳超时: {time_since_last_heartbeat:.1f}秒无响应")
        
        # 处理收集到的工作设备故障
        for device in working_failures:
            try:
                self.handle_working_device_failure(device)
            except Exception as e:
                device_id = device.get("device_id") or device.get("ip", "unknown")
                print(f"处理工作设备 {device_id} 故障时出错: {str(e)}")
                traceback.print_exc()

        # 处理收集到的活跃设备故障
        for device in active_failures:
            try:
                self.handle_active_device_failure(device)
            except Exception as e:
                device_id = device.get("device_id") or device.get("ip", "unknown")
                print(f"处理活跃设备 {device_id} 故障时出错: {str(e)}")
                traceback.print_exc()
        
        # 尝试从故障池恢复设备
        self.check_for_device_recovery()
        
        # 打印当前设备状态总结
        if found_failure or working_failures or active_failures:
            print(f"\n当前设备池状态: 工作:{len(self.working_devices)}, 工作故障:{len(self.failed_working_devices)}")
            print(f"当前设备池状态: 活跃:{len(self.device_pool)}, 活跃故障:{len(self.failed_active_devices)}\n")
    
    def check_for_device_recovery(self):
        """检查故障设备是否恢复"""
        current_time = time.time()
        recovered_working = []
        recovered_active = []

        # 检查工作设备故障池
        with self.lock:
            for device in list(self.failed_working_devices):
                device_id = device.get("device_id") or device.get("ip", "unknown")
                
                # 如果设备在超时期后恢复了心跳
                last_received_time = self.device_heartbeats.get(device_id, 0)
                time_since_last_heartbeat = current_time - last_received_time
                
                if time_since_last_heartbeat <= self.heartbeat_timeout:
                    recovered_working.append(device)
                    print(f"工作设备 {device_id} 恢复心跳，从故障池移除")
        
        # 恢复工作设备
        with self.lock:
            for device in recovered_working:
                # 从故障池移除
                self.failed_working_devices.remove(device)
                # 添加回工作设备池
                self.working_devices.append(device)
                device_id = device.get("device_id") or device.get("ip", "unknown")
                print(f"工作设备 {device_id} 已恢复到工作池")
        
        # 检查活跃设备故障池
        with self.lock:
            for device in list(self.failed_active_devices):
                device_id = device.get("device_id") or device.get("ip", "unknown")
                
                # 如果设备在超时期后恢复了心跳
                last_received_time = self.device_heartbeats.get(device_id, 0)
                time_since_last_heartbeat = current_time - last_received_time
                
                if time_since_last_heartbeat <= self.heartbeat_timeout:
                    recovered_active.append(device)
                    print(f"活跃设备 {device_id} 恢复心跳，从故障池移除")
        
        # 恢复活跃设备
        with self.lock:
            for device in recovered_active:
                # 从故障池移除
                self.failed_active_devices.remove(device)
                # 添加回活跃设备池
                self.device_pool.append(device)
                device_id = device.get("device_id") or device.get("ip", "unknown")
                print(f"活跃设备 {device_id} 已恢复到活跃池")

# 创建设备池管理器实例
device_pool_manager = DevicePoolManager()

def heartbeat_check_thread(device_pool_manager):
    """心跳检查线程，定期检查所有设备的心跳状态"""
    print(f"启动心跳检查线程，每 {device_pool_manager.heartbeat_check_interval} 秒检查一次...")
    
    while True:
        try:
            # 进行设备心跳检查
            device_pool_manager.check_device_heartbeats()
            
            # 定期休眠
            time.sleep(device_pool_manager.heartbeat_check_interval)
        except KeyboardInterrupt:
            print("心跳检查线程接收到退出信号")
            break
        except Exception as e:
            print(f"心跳检查线程出错: {str(e)}")
            traceback.print_exc()
        # 出错后短暂休眠，避免故障速度过快
        time.sleep(1)

def handle_device_registration_and_heartbeat(context, sock, device_pool_manager):
    """处理设备注册和心跳请求"""
    try:
        # 接收消息
        message_parts = sock.recv_multipart()
        
        # 打印原始消息用于调试
        print(f"接收到消息，部分数量: {len(message_parts)}")
        for i, part in enumerate(message_parts):
            print(f"  部分 {i}: {part[:50].decode('utf-8', errors='replace') if isinstance(part, bytes) else part[:50]}...")
        
        # 检查消息部分
        if len(message_parts) < 2:
            print(f"警告: 收到不完整消息，只有 {len(message_parts)} 个部分")
            return
            
        # 在DEALER模式下，第一部分是标识符，第二部分是操作，第三部分是数据
        identifier = message_parts[0]
        
        # 检查标识符是否为二进制数据
        if not isinstance(identifier, bytes):
            print(f"警告: 标识符不是二进制数据，将转换为二进制类型")
            identifier = str(identifier).encode('utf-8')
        
        # 处理不同的消息格式，兼容DEALER套接字
        if len(message_parts) == 2:
            # 客户端只发送了[action, content]，ZMQ自动添加了identifier
            action_bytes = message_parts[1]
            msg_content = None
        else:
            # 客户端发送了完整的[identifier, action, content]
            action_bytes = message_parts[1]
            msg_content = message_parts[2] if len(message_parts) > 2 else None
        
        # 解码action
        try:
            action = action_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # 如果解码失败，尝试查看数据内容
            print(f"警告: 无法解码操作: {action_bytes}")
            action = "unknown"
        
        print(f"处理消息: 标识符={identifier.decode('utf-8', errors='replace')}, 操作={action}")
        
        # 解析消息内容
        data = {}
        if msg_content:
            try:
                data = json.loads(msg_content.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"警告: 消息内容解析失败: {str(e)}")
                print(f"原始内容: {msg_content[:100] if msg_content else 'None'}")
                # 尝试解析第二部分作为JSON (用于某些客户端格式)
                if action_bytes and not action in ["RegisterIP", "HEARTBEAT", "GET_STATUS"]:
                    try:
                        data = json.loads(action_bytes.decode('utf-8'))
                        # 如果成功解析，说明第二部分是数据而不是操作
                        # 需要尝试从数据中确定操作类型
                        if 'role' in data and 'ip' in data:
                            action = "RegisterIP"
                            print(f"自动检测到注册操作: {action}")
                        elif 'timestamp' in data and ('status' in data or 'device_id' in data):
                            action = "HEARTBEAT"
                            print(f"自动检测到心跳操作: {action}")
                    except:
                        pass
                
        # 获取设备信息
        ip = data.get("ip", "unknown")
        device_id = data.get("device_id", "")
        device_identifier = device_id or ip
        
        # 将完整的设备数据和消息格式信息整合为设备信息
        device_info = data.copy()
        device_info["identifier"] = identifier.decode('utf-8', errors='replace')
        
        # 准备响应格式
        response_data = {
            "status": "success", 
            "message": "",
            "data": {}
        }
        
        # 处理设备注册
        if action == "RegisterIP":
            # 添加消息类型标记
            device_info["message_type"] = "Registration"
            
            role = data.get("role", "unknown")
            print(f"接收到设备注册请求: {ip}, 角色: {role}, 设备ID: {device_id}")
            
            # 注册设备到设备池
            result = device_pool_manager.register_device(device_info)
            
            if result:
                response_data["message"] = "设备注册成功"
                response_data["data"] = {
                    "registered": True,
                    "pool_status": {
                        "working_devices": len(device_pool_manager.working_devices),
                        "active_devices": len(device_pool_manager.device_pool),
                        "working_failures": len(device_pool_manager.failed_working_devices),
                        "active_failures": len(device_pool_manager.failed_active_devices)
                    }
                }
                
                # 发送ZMQ格式的响应
                try:
                    print(f"发送注册成功响应到 {device_identifier}, 标识符类型: {type(identifier)}")
                    
                    # 确保响应格式正确
                    response_json = json.dumps(response_data).encode('utf-8')
                    print(f"已编码响应JSON，长度: {len(response_json)}")
                    
                    # 调试信息
                    sending_parts = [identifier, b"REGISTRATION_SUCCESSFUL", response_json]
                    for i, part in enumerate(sending_parts):
                        print(f"发送部分 {i}: 类型={type(part)}, 长度={len(part)}")
                    
                    # 发送响应
                    sock.send_multipart(sending_parts)
                    print(f"注册成功响应发送完成")
                    
                except Exception as e:
                    print(f"发送注册成功响应时出错: {str(e)}")
                    print(f"标识符类型: {type(identifier)}, 长度: {len(identifier)}")
                    print(f"标识符内容: {identifier}")
                    traceback.print_exc()
                    
                    # 尝试使用更简单的响应格式
                    try:
                        print("尝试使用简化响应格式重新发送")
                        sock.send_multipart([
                            identifier,
                            b"OK",
                            b"Registration successful"
                        ])
                        print("简化响应发送成功")
                    except Exception as e2:
                        print(f"发送简化响应也失败: {str(e2)}")
                        traceback.print_exc()
            else:
                response_data["status"] = "error"
                response_data["message"] = "设备注册失败"
                response_data["data"] = {"registered": False}
                
                # 发送ZMQ格式的响应
                try:
                    print(f"发送注册失败响应到 {device_identifier}")
                    response_json = json.dumps(response_data).encode('utf-8')
                    sock.send_multipart([
                        identifier, 
                        b"REGISTRATION_FAILED", 
                        response_json
                    ])
                    print(f"注册失败响应发送完成")
                except Exception as e:
                    print(f"发送注册失败响应时出错: {str(e)}")
                    traceback.print_exc()
        
        # 处理心跳消息
        elif action == "HEARTBEAT":
            # 添加消息类型标记
            device_info["message_type"] = "Heartbeat"
            
            # 更新设备心跳
            device_pool_manager.update_device_heartbeat(device_info)
            
            response_data["message"] = "心跳更新成功"
            response_data["data"] = {
                "acknowledged": True,
                "server_time": time.time()
            }
            
            # 发送ZMQ格式的响应
            try:
                print(f"发送心跳确认响应到 {device_identifier}")
                sock.send_multipart([
                    identifier, 
                    b"HEARTBEAT_ACK", 
                    json.dumps(response_data).encode('utf-8')
                ])
                print(f"心跳确认响应发送完成")
            except Exception as e:
                print(f"发送心跳确认响应时出错: {str(e)}")
                traceback.print_exc()
        
        # 处理状态查询
        elif action == "GET_STATUS":
            # 获取设备池状态
            working_count = len(device_pool_manager.working_devices)
            active_count = len(device_pool_manager.device_pool)
            working_failure_count = len(device_pool_manager.failed_working_devices)
            active_failure_count = len(device_pool_manager.failed_active_devices)
            
            # 构建响应
            response_data["message"] = "设备池状态查询成功"
            response_data["data"] = {
                "pool_status": {
                    "initialization_complete": device_pool_manager.initialization_complete,
                    "working_devices": working_count,
                    "active_devices": active_count,
                    "working_failures": working_failure_count,
                    "active_failures": active_failure_count,
                    "total_devices": working_count + active_count + working_failure_count + active_failure_count
                }
            }
            
            # 添加详细设备信息
            if data.get("include_details", False):
                response_data["data"]["devices"] = {
                    "working": [device for device in device_pool_manager.working_devices],
                    "active": [device for device in device_pool_manager.device_pool],
                    "working_failures": [device for device in device_pool_manager.failed_working_devices],
                    "active_failures": [device for device in device_pool_manager.failed_active_devices]
                }
                
            # 发送ZMQ格式的响应
            try:
                print(f"发送状态信息响应")
                sock.send_multipart([
                    identifier, 
                    b"STATUS_INFO", 
                    json.dumps(response_data).encode('utf-8')
                ])
                print(f"状态信息响应发送完成")
            except Exception as e:
                print(f"发送状态信息响应时出错: {str(e)}")
                traceback.print_exc()
        
        # 处理IP地址获取请求（兼容旧代码）
        elif action == "GET_IP_ADDRESSES":
            # 将标识符添加到请求列表
            global ip_graph_requested
            if identifier not in ip_graph_requested:
                ip_graph_requested.append(identifier)
                
            print(f"收到IP地址请求，已添加 {identifier.decode('utf-8', errors='replace')} 到请求列表")
            
            # 发送简单确认
            try:
                sock.send_multipart([
                    identifier, 
                    b"IP_REQUEST_RECEIVED", 
                    b"Request received"
                ])
                print(f"IP请求确认响应发送完成")
            except Exception as e:
                print(f"发送IP请求确认响应时出错: {str(e)}")
                traceback.print_exc()
        
        # 未知操作
        else:
            print(f"收到未知操作请求: {action}")
            response_data["status"] = "error"
            response_data["message"] = f"未知操作: {action}"
            
            # 发送ZMQ格式的响应
            try:
                sock.send_multipart([
                    identifier, 
                    b"ERROR", 
                    json.dumps(response_data).encode('utf-8')
                ])
                print(f"未知操作错误响应发送完成")
            except Exception as e:
                print(f"发送未知操作错误响应时出错: {str(e)}")
                traceback.print_exc()
        
    except Exception as e:
        print(f"处理设备注册或心跳消息时出错: {str(e)}")
        traceback.print_exc()
        
        # 尝试发送错误响应（如果可能）
        error_response = {
            "status": "error",
            "message": f"处理请求时出错: {str(e)}",
            "data": {}
        }
        try:
            # 只有当我们有标识符时才能发送响应
            if 'identifier' in locals():
                sock.send_multipart([
                    identifier, 
                    b"ERROR", 
                    json.dumps(error_response).encode('utf-8')
                ])
                print(f"错误响应发送完成")
        except Exception as e:
            print(f"尝试发送错误响应时出错: {str(e)}")
            traceback.print_exc()

def handle_model_loading_and_distribution(requested_model, working_devices, ip_graph_requested):
    """处理模型加载和分配"""
    print(f"\n开始处理模型加载和分配...")
    print(f"模型名称: {requested_model}")
    print(f"工作设备数量: {len(working_devices)}")
    
    # 1. 确定模型和量化选项
    if requested_model == "bloom560m":
        Quntization_Option = False
    elif requested_model == "bloom560m-int8":
            Quntization_Option = True
            requested_model = "bloom560m"  # 内部使用非量化名称
    else:
        print(f"使用默认模型: bloom560m-int8")
        Quntization_Option = True
        requested_model = "bloom560m"

    # 2. 检索模型发送目录
    to_send_path = retrieve_sending_dir(root_dir, requested_model, 
                                      quantization_option=Quntization_Option,
                                            residual_connection=residual_connection_option)

    # 3. 检查模型目录是否存在
    if os.path.isdir(to_send_path):
        print('模型目录已存在，使用现有模型')
        # 加载现有的IP模块映射和会话信息
        with open(os.path.join(to_send_path, 'ip_module.json'), 'r') as file:
            ip_module_json = file.read()

        with open(os.path.join(to_send_path, 'session.json'), 'r') as file:
            session_index_json = file.read()

        ip_module = json.loads(ip_module_json)
        session = json.loads(session_index_json)
        file_cfg = retrieve_file_cfg(ip_module)

        # 向设备发送监控初始化信号(False表示使用现有模型)
        for ip in ip_graph_requested:
            registration_socket.send_multipart([ip, b"False"])
    else:
        print('模型目录不存在，开始准备模型...')
        # 向设备发送监控初始化信号(True表示需要准备新模型)
        for ip in ip_graph_requested:
            registration_socket.send_multipart([ip, b"True"])
    
        # 创建模型卡片对象
        model_card = ModelCard(requested_model, 
                              quantization_option=Quntization_Option, 
                              task_type=task,
                              residual_connection=residual_connection_option, 
                              load_balancing_option=False,
                              split_size=split_size)
        
        # 准备优化信息
        mem_util, out_size_map, bytearray_path, flop_module_path, num_flop, module_flop_map, num_modules = model_card.prepare_optimization_info()
        
        # 获取分词器路径
        tokenizer_dir = model_card.retreive_tokenizer_path()
        directory_path = os.path.dirname(bytearray_path)

        print(f'bytearray_path: {bytearray_path}')
        print(f'flop_module_path: {flop_module_path}')
        print(f'num_flop: {num_flop}')
        print(f'out_size_map: {out_size_map}')

         # 向设备发送准备监控的信号
        for ip in ip_graph_requested:
            registration_socket.send_multipart([ip, b"ready for monitor"])
    
        # 启动监控
        device_list = list(working_devices) # 转换为列表
        monitoring = monitor.Monitor(monitor_receive_interval, monitor_port, 
                                   device_list, requested_model,
                                      bytearray_path, flop_module_path, num_flop, runtime_option)
        monitor_thread = threading.Thread(target=monitoring.start)
        monitor_thread.start()

        # 等待监控就绪
        num_devices = len(device_list)
        monitoring.is_monitor_ready.wait()
        print("监控就绪，准备获取监控信息")

        try:
            # 获取监控信息
            ping_latency, bandwidths, TotalMem, AvailMem, flop_speed = monitoring.get_monitor_info()
            
            # 检查获取到的数据是否有效
            if (len(monitoring.latency_list) == 0 or 
                len(monitoring.bandwidth_list) == 0 or 
                len(monitoring.memory_list) == 0):
                print("警告: 监控数据不完整，使用默认值")
                
                # 创建默认值
                ping_latency = np.ones((num_devices, num_devices)) * 10  # 默认10ms延迟
                np.fill_diagonal(ping_latency, 0)  # 对角线为0
                
                bandwidths = np.ones((num_devices, num_devices)) * 10  # 默认10MB/s带宽
                np.fill_diagonal(bandwidths, 0)  # 对角线为0
                
                TotalMem = np.ones(num_devices) * 4000  # 默认4GB内存
                AvailMem = np.ones(num_devices) * 2000  # 默认2GB可用
                flop_speed = np.ones(num_devices) * 1000  # 默认1000 MFLOPS
        except Exception as e:
            print(f"获取监控信息时出错: {e}")
            traceback.print_exc()
            
            # 出错时使用默认值
            print("使用默认监控值")
            ping_latency = np.ones((num_devices, num_devices)) * 10
            np.fill_diagonal(ping_latency, 0)
            
            bandwidths = np.ones((num_devices, num_devices)) * 10
            np.fill_diagonal(bandwidths, 0)
            
            TotalMem = np.ones(num_devices) * 4000
            AvailMem = np.ones(num_devices) * 2000
            flop_speed = np.ones(num_devices) * 1000
        
        # 设置内存阈值
        mem_threshold = 0.7  # 设置内存阈值
        TotalMem = [m * mem_threshold for m in TotalMem]
        AvailMem = [m * mem_threshold for m in AvailMem]
    
        # 打印监控信息
        print("-----------------监控信息----------------------")
        print(f"设备数量: {num_devices}")
        print(f"延迟: {ping_latency}")
        print(f"带宽: {bandwidths}")
        print(f"总内存: {TotalMem}")
        print(f"可用内存: {AvailMem}")
        print(f"计算能力: {flop_speed}")
        
        # 使用Round-Robin方式进行模块分配
        if model_card.split_size:
            print(f"模型分割大小: {model_card.split_size}")
        print("使用Round-Robin分配方法")
        
        def round_robin_module_arrangement(num_devices, num_modules):
            arrangement = [[0 for _ in range(num_modules)] for _ in range(num_devices)]
            modules_per_device = num_modules // num_devices
            extra_modules = num_modules % num_devices
            start = 0
            for i in range(num_devices):
                end = start + modules_per_device + (1 if i < extra_modules else 0)
                for j in range(start, end):
                    arrangement[i][j] = 1
                start = end
            return np.array(arrangement)

            initial_module_arrangement = round_robin_module_arrangement(split_size, split_size)
            overlapping_module_arrangement = initial_module_arrangement
            print(f"模块分配方案:\n{initial_module_arrangement}")

            # 准备发送模型
            model_dirs = model_card.prepare_model_to_send(module_arrangement=initial_module_arrangement)
            device_module_order = model_card.device_module_arrangement
            device_dir_map = {tuple(device_module_order[i]): model_dirs[i] for i in range(len(model_dirs))}
                
            # 映射IP到设备模块
            ip_device_module_map = {}
            for i in range(len(device_list)):
                device_ip = device_list[i].get("ip").encode("utf-8")
                ip_device_module_map[device_ip] = device_module_order[i]
                
            # 创建推理会话
            session = [str(j) for i in device_module_order for j in i]
                
            # 排序IP图顺序
            ip_graph = []
            ip_module = []
            sorted_device_module_order = sorted(device_module_order)
            final_sorted_device_module = [[0]] * len(sorted_device_module_order)
                
            for ip, val in ip_device_module_map.items():
                if sorted_device_module_order.index(val) == 0:  # 头节点
                    final_sorted_device_module[0] = [ip, device_dir_map[tuple(val)]]
                elif sorted_device_module_order.index(val) != 0 and sorted_device_module_order.index(val) != len(sorted_device_module_order) - 1:
                    insert_index = sorted_device_module_order.index(val)
                    final_sorted_device_module[insert_index] = [ip, device_dir_map[tuple(val)]]
                else:  # 尾节点
                    final_sorted_device_module[-1] = [ip, device_dir_map[tuple(val)]]

            print(f"会话索引: {session}")

            # 构建文件配置
            file_cfg = {}
            for d in range(len(final_sorted_device_module)):
                ip_encode = final_sorted_device_module[d][0]
                if final_sorted_device_module[d][1]:
                    print(f"{ip_encode}:{final_sorted_device_module[d][1][0]}")
                    file_cfg[ip_encode] = final_sorted_device_module[d][1][0]
                    ip_graph.append(ip_encode.decode("utf-8"))
                    ip_module.append([ip_encode.decode("utf-8"), file_cfg[ip_encode]])

            # 保存配置到文件
            to_send_model_path = retrieve_sending_dir(root_dir, requested_model, 
                                                    quantization_option=Quntization_Option,
                                                      residual_connection=residual_connection_option)
                
            ip_module_json = json.dumps(ip_module)
            session_index_json = json.dumps(session)

            with open(os.path.join(to_send_model_path, "ip_module.json"), 'w') as file:
                file.write(ip_module_json)

            with open(os.path.join(to_send_model_path, "session.json"), 'w') as file:
                file.write(session_index_json)
   

    # 修改file_cfg JSON文件中的IP地址
    pathLists = []
    for index, device in enumerate(working_devices):
        ip = device.get("ip")
        role = device.get("role")

        if not Quntization_Option:
            pathList = [str(ip), f"/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom560m_unquantized_res/device{index}/module{index}/module.zip"]
        else:
            pathList = [str(ip), f"/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom560m_quantized_int8_res/device{index}/module{index}/module.zip"]
        
        pathLists.append(pathList)

    # 保存路径列表
    with open(os.path.join(to_send_path, 'ip_module.json'), 'w') as file:
        json.dump(pathLists, file)

    # 读取保存的JSON
    with open(os.path.join(to_send_path, 'ip_module.json'), 'r') as file:
        ip_module_json = file.read()
    
    # 处理IP模块数据
    ip_module = json.loads(ip_module_json)
    file_cfg = retrieve_file_cfg(ip_module)
    ip_graph, dependencyMap = retrieve_sending_info(
        root_dir, requested_model, 
        ip_module_list=ip_module,
                                                    quantization_option=Quntization_Option,
        residual_connection=residual_connection_option
    )

    print(f'\n图: {ip_graph}')
    print(f"会话索引: {session}")

    # 创建配置
    config = {
        "file_path": file_cfg,
              "num_sample": b'1000',
        "num_device": len(working_devices),
              "max_length": b'40',
              "task_type": "generation".encode('utf-8'),
              "core_pool_size": b'1',
              "head_node": ip_graph[0],
              "tail_node": ip_graph[-1],
              "dependency": dependencyMap,
              "session_index": ";".join(session).encode('utf-8'),
              "graph": ",".join(ip_graph).encode('utf-8'),
              "skip_model_transmission": MODEL_EXIST_ON_DEVICE,
              "model_name": requested_model,
              "reload_sampleId": None,
              "onnx": True,
        "ids": {}
    }

    # 读取依赖关系JSON文件
    for idx, fPath in dependencyMap.items():
        file = open(fPath, "r")
        data = json.load(file)
        config["dependency"][idx] = data

    print("配置完成，准备发送模型...")

    # 启动通信线程
    status = {}
    threads = []
    lock = threading.Lock()
    locks = [threading.Lock(), threading.Lock()]
    conditions = [threading.Condition() for i in range(len(working_devices) + 1)]

    for i in range(config["num_device"]):
        t = threading.Thread(
            target=root_server.communication_open_close, 
            args=(registration_socket, config, status, conditions, locks)
        )
        threads.append(t)

    # 启动所有线程
    for i in threads:
        i.start()

    # 等待所有线程完成
    for t in threads:
        t.join()
        if hasattr(t, 'exception') and t.exception:
            print(f"线程 {t.name} 出现异常: {t.exception}")

    print("模型加载和分配完成!")

def main_thread():
    """主服务线程，处理设备注册和通信"""
    global requested_model  # 启用对全局变量的修改
    global devices          # 用于兼容旧代码
    global ip_graph_requested
    
    print(f"设备注册/通信线程已启动，监听端口: 23456")
    print(f"阶段1: 等待初始设备注册 (超时: {TIMEOUT} 秒)...")
    
    ip_graph_requested = []  # 存储所有请求设备的IP地址
    last_received_time = time.time()
    in_initial_phase = True  # 是否处于初始化阶段
    last_device_count = 0    # 上次检查时的设备数量
    
    while True:  # 持续运行
        current_device_count = len(device_pool_manager.device_pool)
        
        # 每当设备数量变化时打印一次
        if current_device_count != last_device_count:
            print(f"\n设备池更新: {last_device_count} -> {current_device_count} 个设备")
            last_device_count = current_device_count
            
            # 如果有设备加入，更新最后接收时间，避免过早进入下一阶段
            if current_device_count > 0:
                last_received_time = time.time()
                print(f"检测到新设备加入，重置初始化超时计时器...")
        
        # 检查是否有消息等待处理
        if registration_socket.poll(1000):  # 1秒超时
            # 使用设备注册和心跳处理函数处理请求
            handle_device_registration_and_heartbeat(context, registration_socket, device_pool_manager)
            # 更新最后接收时间 (只有在初始化阶段才会更新)
            if in_initial_phase:
                last_received_time = time.time()
        
        # 检查初始化阶段是否结束
        if in_initial_phase and (time.time() - last_received_time > TIMEOUT):
            print(f"\n初始化阶段超时: 已有 {TIMEOUT} 秒未收到新的设备注册请求")
            in_initial_phase = False
            
            # 检查是否有设备注册
            if len(device_pool_manager.device_pool) > 0:
                print(f"初始化结束时有 {len(device_pool_manager.device_pool)} 个设备在活跃池中")
                for i, device in enumerate(device_pool_manager.device_pool):
                    device_id = device.get("device_id") or device.get("ip", "unknown")
                    role = device.get("role", "unknown")
                    print(f"  {i+1}. {device_id} (角色: {role})")
            else:
                print("警告: 初始化结束时没有设备在活跃池中")
            
            # 将当前设备池中的设备设为工作设备
            device_pool_manager.set_initialization_complete()
            
            # 检查转移后的结果
            if len(device_pool_manager.working_devices) > 0:
                print(f"成功将 {len(device_pool_manager.working_devices)} 个设备转移到工作设备池")
                for i, device in enumerate(device_pool_manager.working_devices):
                    device_id = device.get("device_id") or device.get("ip", "unknown")
                    role = device.get("role", "unknown")
                    print(f"  {i+1}. {device_id} (角色: {role})")
            else:
                print("警告: 转移后工作设备池为空")
                
            print(f"进入正常运行阶段: 后续注册的设备将只作为活跃设备，不会成为工作设备")
            
            # 初始化阶段结束后，启动模型加载和分配
            if len(device_pool_manager.working_devices) > 0:
                model_thread = threading.Thread(
                    target=handle_model_loading_and_distribution,
                    args=(requested_model, device_pool_manager.working_devices, ip_graph_requested),
                    daemon=True
                )
                model_thread.start()
                print("已启动模型加载和分配线程")
            else:
                print("警告: 没有工作设备可用，无法启动模型加载和分配")
        
        # 每10秒打印一次当前设备池状态
        if time.time() % 10 < 1:  # 接近每10秒的整数时间点
            with device_pool_manager.lock:
                working_count = len(device_pool_manager.working_devices)
                active_count = len(device_pool_manager.device_pool)
                working_failures = len(device_pool_manager.failed_working_devices)
                active_failures = len(device_pool_manager.failed_active_devices)
                
                if working_count > 0 or active_count > 0 or working_failures > 0 or active_failures > 0:
                    print("\n当前设备池状态:")
                    print(f"  工作设备: {working_count}个, 工作设备故障: {working_failures}个")
                    print(f"  活跃设备: {active_count}个, 活跃设备故障: {active_failures}个")
                    print(f"  初始化阶段: {'是' if in_initial_phase else '否'}")
                    
            time.sleep(1)  # 避免在同一秒内多次打印

if __name__ == "__main__":
    start = time.time()
    
    # 设置更详细的日志输出
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("DistributedServer")
    logger.setLevel(logging.DEBUG)
    
    # 确保所有print立即输出（不缓冲）
    print = functools.partial(print, flush=True)
    
    # 初始化ZMQ上下文
    context = zmq.Context()
    context.setsockopt(zmq.LINGER, 1000)  # 关闭时等待消息发送1秒
    
    print("==== 分布式推理系统启动 ====")
    print(f"等待设备注册，初始化阶段超时: {TIMEOUT}秒")
    
    # 初始化设备管理器
    device_pool_manager = DevicePoolManager()
    
    # 设置ZMQ套接字进行设备注册和通信
    try:
        # 清理之前可能存在的连接
        import socket
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            test_sock.bind(('0.0.0.0', 23456))
            test_sock.close()
            print("端口23456可用，准备启动服务")
        except:
            print("警告: 端口23456已被占用，可能存在其他实例")
            import os
            os.system("pkill -f 560m_root_bloom.py")
            print("尝试终止其他实例并等待2秒...")
            time.sleep(2)
    except:
        print("无法测试端口状态，继续尝试启动")
    
    # 创建ZMQ套接字并绑定
    try:
        print("创建ROUTER套接字并绑定到端口23456...")
        registration_socket = context.socket(zmq.ROUTER)
        registration_socket.setsockopt(zmq.LINGER, 1000)  # 关闭时等待消息发送1秒
        registration_socket.setsockopt(zmq.SNDTIMEO, 10000)  # 发送超时10秒
        registration_socket.setsockopt(zmq.RCVTIMEO, 10000)  # 接收超时10秒
        registration_socket.setsockopt(zmq.SNDHWM, 1000)  # 设置发送高水位标记
        registration_socket.setsockopt(zmq.RCVHWM, 1000)  # 设置接收高水位标记
        registration_socket.bind("tcp://*:23456")
        print("ROUTER套接字绑定成功")
    except Exception as e:
        print(f"绑定套接字时出错: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # 启动心跳检查线程
    heartbeat_thread = threading.Thread(
        target=heartbeat_check_thread,
        args=(device_pool_manager,),
        daemon=True
    )
    heartbeat_thread.start()
    
    # 启动设备注册和通信线程
    registration_thread = threading.Thread(
        target=main_thread,
        daemon=True
    )
    registration_thread.start()
    
    print("\n初始设备池状态:")
    print(f"工作设备: {len(device_pool_manager.working_devices)}个")
    print(f"活跃设备: {len(device_pool_manager.device_pool)}个")
    print(f"工作设备故障: {len(device_pool_manager.failed_working_devices)}个") 
    print(f"活跃设备故障: {len(device_pool_manager.failed_active_devices)}个")
    
    # 主线程等待退出信号，同时定期打印设备池状态
    try:
        while True:
            # 每隔30秒打印一次完整状态
            time.sleep(30)
            
            print("\n系统运行状态 ====")
            print(f"已运行时间: {(time.time() - start) / 60:.1f}分钟")
            with device_pool_manager.lock:
                working_count = len(device_pool_manager.working_devices)
                active_count = len(device_pool_manager.device_pool)
                working_failures = len(device_pool_manager.failed_working_devices)
                active_failures = len(device_pool_manager.failed_active_devices)
                
                print(f"设备池状态: {working_count}工作, {active_count}活跃, {working_failures}工作故障, {active_failures}活跃故障")
                print(f"初始化完成: {'是' if device_pool_manager.initialization_complete else '否'}")
                
                # 打印部分设备详情
                if working_count > 0:
                    print("\n工作设备示例:")
                    for i, d in enumerate(device_pool_manager.working_devices[:3]):
                        device_id = d.get("device_id") or d.get("ip", "unknown")
                        print(f"  {i+1}. ID: {device_id}, 角色: {d.get('role', 'unknown')}")
                    if working_count > 3:
                        print(f"  ...及其他{working_count-3}个设备")
                
                if active_count > 0:
                    print("\n活跃设备示例:")
                    for i, d in enumerate(device_pool_manager.device_pool[:3]):
                        device_id = d.get("device_id") or d.get("ip", "unknown")
                        print(f"  {i+1}. ID: {device_id}, 角色: {d.get('role', 'unknown')}")
                    if active_count > 3:
                        print(f"  ...及其他{active_count-3}个设备")
                
    except KeyboardInterrupt:
        print("\n收到退出信号，正在关闭服务器...")
    except Exception as e:
        print(f"主线程异常: {e}")
        traceback.print_exc()
    finally:
        # 清理资源
        try:
            registration_socket.close()
            context.term()
            print("ZMQ资源已释放")
        except:
            pass
        print("服务器已关闭")
