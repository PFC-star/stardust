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
import socket
import traceback

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
    def __init__(self):
        self.lock = threading.Lock()
        self.device_pool = deque()            # 全部已注册活跃设备池（非工作设备）
        self.working_devices = deque()        # 工作设备池（初始阶段注册的设备）
        self.active_devices = {}              # {task_id: device_list} 当前活跃任务使用的设备
        self.failed_working_devices = deque() # 工作设备故障池
        self.failed_active_devices = deque()  # 活跃设备故障池
        self.task_counter = 0
        self.device_heartbeats = {}           # 记录设备最后心跳时间
        self.heartbeat_timeout = 30           # 心跳超时时间(秒)
        self.heartbeat_check_interval = 10    # 心跳检查间隔(秒)
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
            device_pool_manager.printInfo()
    def register_device(self, device_info):
        """注册新设备到设备池"""
        try:
            with self.lock:
                # 先获取设备的标识信息
                device_id = device_info.get("device_id")
                ip = device_info.get("ip")
                
                if not device_id and not ip:
                    print("错误: 设备注册没有提供ID或IP地址")
                    return False
                    
                # 初始化心跳时间
                identifier = device_id or ip
                self.device_heartbeats[identifier] = time.time()
                print(f"为设备 {identifier} 初始化心跳时间")
                
                # 检查设备应该注册到哪个池
                if self.initialization_complete:
                    # 初始化完成后，先检查该设备是否已经在工作设备池中
                    working_device_exists = False
                    for device in self.working_devices:
                        if (device_id and device.get("device_id") == device_id) or (not device_id and device.get("ip") == ip):
                            # 设备已在工作设备池中，更新其信息
                            device.update(device_info)
                            print(f"工作设备已更新: ID={device_id or 'None'}, IP={ip}")
                            working_device_exists = True
                            break
                    
                    # 如果不在工作设备池中，再检查活跃设备池
                    if not working_device_exists:
                        active_device_exists = False
                        for device in self.device_pool:
                            if (device_id and device.get("device_id") == device_id) or (not device_id and device.get("ip") == ip):
                                # 更新设备信息
                                device.update(device_info)
                                print(f"活跃设备已更新: ID={device_id or 'None'}, IP={ip}")
                                active_device_exists = True
                                break
                        
                        # 如果两个池中都不存在，添加为新活跃设备
                        if not active_device_exists:
                            self.device_pool.append(device_info)
                            print(f"运行阶段 - 新设备已注册为活跃设备: ID={device_id or 'None'}, IP={ip}, 角色={device_info.get('role')}")
                else:
                    # 初始化阶段，设备应添加到初始设备池（稍后会通过set_initialization_complete转移到工作设备池）
                    device_exists = False
                    for device in self.device_pool:
                        if (device_id and device.get("device_id") == device_id) or (not device_id and device.get("ip") == ip):
                            # 更新设备信息
                            device.update(device_info)
                            print(f"初始设备已更新: ID={device_id or 'None'}, IP={ip}")
                            device_exists = True
                            break
                    
                    if not device_exists:
                        # 添加为新初始设备
                        self.device_pool.append(device_info)
                        print(f"初始化阶段 - 新设备已注册: ID={device_id or 'None'}, IP={ip}, 角色={device_info.get('role')}")
                
                # 最后打印当前设备池状态
                device_pool_manager.printInfo()
                return True
        except Exception as e:
            print(f"设备注册时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    def update_device_heartbeat(self, device_id):
        """更新设备心跳时间"""
        try:
            if not device_id:
                print("警告: 尝试更新无效设备ID的心跳")
                return
            
            with self.lock:
                current_time = time.time()
                old_time = self.device_heartbeats.get(device_id, 0)
                self.device_heartbeats[device_id] = current_time
                
                # 记录时间差，用于监控
                if old_time > 0:
                    time_diff = current_time - old_time
                    if time_diff > self.heartbeat_timeout / 2:
                        print(f"警告: 设备 {device_id} 心跳间隔较长: {time_diff:.1f}秒")
                    else:
                        print(f"设备 {device_id} 心跳更新: {time_diff:.1f}秒前")
                else:
                    print(f"设备 {device_id} 首次心跳记录")
        except Exception as e:
            print(f"更新设备心跳时出错: {e}")
    
    def check_device_heartbeats(self):
        """检查所有设备的心跳状态"""
        try:
            current_time = time.time()
            failed_working_devices = []
            failed_active_devices = []
            
            # 收集故障设备，避免长时间持有锁
            with self.lock:
                # 检查工作设备的心跳
                for device in list(self.working_devices):
                    device_id = device.get("device_id") or device.get("ip")
                    if not device_id:
                        print(f"警告: 工作设备没有ID或IP，无法检查心跳: {device}")
                        continue
                        
                    last_heartbeat = self.device_heartbeats.get(device_id, 0)
                    
                    # 忽略心跳时间为0的设备，这可能是新注册的设备
                    if last_heartbeat == 0:
                        print(f"设备 {device_id} 尚未发送心跳，更新为当前时间")
                        self.device_heartbeats[device_id] = current_time
                        continue
                    
                    heartbeat_age = current_time - last_heartbeat
                    if heartbeat_age > self.heartbeat_timeout:
                        print(f"工作设备 {device_id} 心跳超时 ({heartbeat_age:.1f}秒)，可能已故障")
                        # 创建设备副本以避免修改原始引用
                        failed_working_devices.append(device.copy())
                    else:
                        print(f"工作设备 {device_id} 心跳正常，最后心跳: {heartbeat_age:.1f}秒前")
                
                # 检查活跃设备的心跳
                for device in list(self.device_pool):
                    device_id = device.get("device_id") or device.get("ip")
                    if not device_id:
                        print(f"警告: 活跃设备没有ID或IP，无法检查心跳: {device}")
                        continue
                        
                    last_heartbeat = self.device_heartbeats.get(device_id, 0)
                    
                    # 忽略心跳时间为0的设备，这可能是新注册的设备
                    if last_heartbeat == 0:
                        print(f"设备 {device_id} 尚未发送心跳，更新为当前时间")
                        self.device_heartbeats[device_id] = current_time
                        continue
                    
                    heartbeat_age = current_time - last_heartbeat
                    if heartbeat_age > self.heartbeat_timeout:
                        print(f"活跃设备 {device_id} 心跳超时 ({heartbeat_age:.1f}秒)，可能已故障")
                        # 创建设备副本以避免修改原始引用
                        failed_active_devices.append(device.copy())
                    else:
                        print(f"活跃设备 {device_id} 心跳正常，最后心跳: {heartbeat_age:.1f}秒前")
            
            # 释放锁后处理故障设备
            failures_count = 0
            # 处理工作设备故障
            for device in failed_working_devices:
                print(f"准备处理工作设备故障: {device.get('device_id') or device.get('ip')}")
                self.handle_working_device_failure(device)
                failures_count += 1
            
            # 处理活跃设备故障
            for device in failed_active_devices:
                print(f"准备处理活跃设备故障: {device.get('device_id') or device.get('ip')}")
                self.handle_active_device_failure(device)
                failures_count += 1
            
            return failures_count
        except Exception as e:
            print(f"检查设备心跳时出错: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def handle_working_device_failure(self, device):
        """处理工作设备故障"""
        device_id = device.get("device_id") or device.get("ip")
        if not device_id:
            print("警告: 设备没有ID或IP，无法处理故障")
            return
            
        try:
            print(f"开始处理工作设备故障: {device_id}")
            
            # 操作1: 从工作设备池中查找并移除设备
            device_to_remove = None
            with self.lock:
                device_found = False
                for d in list(self.working_devices):
                    d_id = d.get("device_id")
                    d_ip = d.get("ip")
                    if (device_id == d_id) or (device_id == d_ip):
                        device_found = True
                        device_to_remove = d  # 保存找到的设备引用
                        self.working_devices.remove(d)
                        print(f"工作设备 {device_id} 已从工作设备池中移除")
                        break
                
                if not device_found:
                    print(f"警告: 工作设备 {device_id} 不在设备池中，可能已被移除")
                    return
            
            if not device_to_remove:
                return
                
            # 操作2: 从活跃任务中移除设备
            with self.lock:
                tasks_to_update = []
                for task_id, devices in self.active_devices.items():
                    device_index = None
                    for i, d in enumerate(devices):
                        d_id = d.get("device_id")
                        d_ip = d.get("ip")
                        if (device_id == d_id) or (device_id == d_ip):
                            device_index = i
                            break
                    
                    if device_index is not None:
                        tasks_to_update.append((task_id, device_index))
                        
                # 执行实际的移除操作
                for task_id, device_index in tasks_to_update:
                    devices = self.active_devices[task_id]
                    devices.pop(device_index)
                    print(f"工作设备 {device_id} 已从任务 {task_id} 中移除")
                    
                    if not devices:  # 如果任务没有设备了，删除该任务
                        del self.active_devices[task_id]
                        print(f"任务 {task_id} 已删除，因为没有可用设备")
            
            # 操作3: 添加到故障池和清理心跳记录
            with self.lock:
                # 使用完整的原始设备信息
                device_to_remove["failure_time"] = time.time()
                device_to_remove["failure_reason"] = "heartbeat_timeout"
                device_to_remove["device_type"] = "working"
                self.failed_working_devices.append(device_to_remove)
                print(f"工作设备 {device_id} 已添加到工作设备故障池")
                
                # 从心跳记录中移除
                if device_id in self.device_heartbeats:
                    del self.device_heartbeats[device_id]
                    print(f"已清除设备 {device_id} 的心跳记录")
                    
        except Exception as e:
            print(f"处理工作设备故障时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_active_device_failure(self, device):
        """处理活跃设备故障"""
        device_id = device.get("device_id") or device.get("ip")
        if not device_id:
            print("警告: 设备没有ID或IP，无法处理故障")
            return
            
        try:
            print(f"开始处理活跃设备故障: {device_id}")
            
            # 操作1: 从活跃设备池中查找并移除设备
            device_to_remove = None
            with self.lock:
                device_found = False
                for d in list(self.device_pool):
                    d_id = d.get("device_id")
                    d_ip = d.get("ip")
                    if (device_id == d_id) or (device_id == d_ip):
                        device_found = True
                        device_to_remove = d  # 保存找到的设备引用
                        self.device_pool.remove(d)
                        print(f"活跃设备 {device_id} 已从活跃设备池中移除")
                        break
                
                if not device_found:
                    print(f"警告: 活跃设备 {device_id} 不在设备池中，可能已被移除")
                    return
            
            if not device_to_remove:
                return
                
            # 操作2: 添加到故障池和清理心跳记录
            with self.lock:
                # 使用完整的原始设备信息
                device_to_remove["failure_time"] = time.time()
                device_to_remove["failure_reason"] = "heartbeat_timeout"
                device_to_remove["device_type"] = "active"
                self.failed_active_devices.append(device_to_remove)
                print(f"活跃设备 {device_id} 已添加到活跃设备故障池")
                
                # 从心跳记录中移除
                if device_id in self.device_heartbeats:
                    del self.device_heartbeats[device_id]
                    print(f"已清除设备 {device_id} 的心跳记录")
                    
        except Exception as e:
            print(f"处理活跃设备故障时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_device_failure(self, device):
        """处理设备故障（兼容旧代码）"""
        device_id = device.get("device_id") or device.get("ip")
        if not device_id:
            print("警告: 设备没有ID或IP，无法处理故障")
            return
            
        # 检查设备是工作设备还是活跃设备
        is_working = False
        is_active = False
        
        with self.lock:
            # 使用ID或IP进行比较，而不是直接比较对象
            for d in self.working_devices:
                d_id = d.get("device_id")
                d_ip = d.get("ip")
                if (device_id == d_id) or (device_id == d_ip):
                    is_working = True
                    break
                    
            if not is_working:
                for d in self.device_pool:
                    d_id = d.get("device_id")
                    d_ip = d.get("ip")
                    if (device_id == d_id) or (device_id == d_ip):
                        is_active = True
                        break
        
        if is_working:
            self.handle_working_device_failure(device)
        elif is_active:
            self.handle_active_device_failure(device)
        else:
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
    def printInfo(self):
        print("\n初始设备池状态:")
        print(f"工作设备: {len(self.working_devices)}个")
        print(f"活跃设备: {len(self.device_pool)}个")
        print(f"工作设备故障: {len(self.failed_working_devices)}个") 
        print(f"活跃设备故障: {len(self.failed_active_devices)}个")
# 创建设备池管理器实例
device_pool_manager = DevicePoolManager()

def heartbeat_check_thread():
    """心跳检查线程"""
    print("心跳检查线程已启动，每 {} 秒检查一次设备心跳状态，超时时间 {} 秒".format(
        device_pool_manager.heartbeat_check_interval, 
        device_pool_manager.heartbeat_timeout
    ))
    
    consecutive_empty_checks = 0
    
    while True:
        try:
            print(f"\n正在检查所有设备的心跳状态... 当前时间: {time.time():.2f}")
            
            # 获取故障前的设备状态
            before_count = {
                'working': len(device_pool_manager.working_devices),
                'active': len(device_pool_manager.device_pool),
                'failed_working': len(device_pool_manager.failed_working_devices),
                'failed_active': len(device_pool_manager.failed_active_devices)
            }
            
            # 执行心跳检查
            failures_detected = device_pool_manager.check_device_heartbeats()
            
            # 获取故障后的设备状态
            after_count = {
                'working': len(device_pool_manager.working_devices),
                'active': len(device_pool_manager.device_pool),
                'failed_working': len(device_pool_manager.failed_working_devices),
                'failed_active': len(device_pool_manager.failed_active_devices)
            }
            
            # 检查是否有变化
            status_changed = (
                before_count['working'] != after_count['working'] or
                before_count['active'] != after_count['active'] or
                before_count['failed_working'] != after_count['failed_working'] or
                before_count['failed_active'] != after_count['failed_active']
            )
            
            # 确定是否需要打印详细信息
            if failures_detected > 0 or status_changed:
                print("\n⚠️ 设备池状态发生变化:")
                print(f"  工作设备: {before_count['working']} -> {after_count['working']} 个")
                print(f"  活跃设备: {before_count['active']} -> {after_count['active']} 个")
                print(f"  工作设备故障: {before_count['failed_working']} -> {after_count['failed_working']} 个") 
                print(f"  活跃设备故障: {before_count['failed_active']} -> {after_count['failed_active']} 个")
                
                if failures_detected > 0:
                    print(f"\n本次检测到 {failures_detected} 个新故障设备")
                
                # 打印详细的故障设备信息
                failed_devices = device_pool_manager.get_failed_devices()
                if failed_devices:
                    print("\n故障设备列表:")
                    for i, device in enumerate(failed_devices):
                        device_id = device.get("device_id", "N/A")
                        ip = device.get("ip", "N/A")
                        role = device.get("role", "N/A")
                        failure_time = device.get("failure_time", "N/A")
                        if isinstance(failure_time, (int, float)):
                            failure_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(failure_time))
                        else:
                            failure_time_str = str(failure_time)
                        
                        failure_reason = device.get("failure_reason", "N/A")
                        device_type = device.get("device_type", "unknown")
                        
                        print(f"  {i+1}. ID: {device_id}, IP: {ip}, 角色: {role}")
                        print(f"     类型: {device_type}, 故障时间: {failure_time_str}")
                        print(f"     故障原因: {failure_reason}")
                
                consecutive_empty_checks = 0
            else:
                consecutive_empty_checks += 1
                if consecutive_empty_checks <= 2:  # 只在连续空检查次数较少时打印常规状态
                    print("\n设备池状态正常 (无变化):")
                    device_pool_manager.printInfo()
                else:
                    print(f"设备池状态正常 (已连续 {consecutive_empty_checks} 次无变化)")
                    
            # 每5次（约50秒）无变化检查后，重新完整打印一次状态以保持信息更新
            if consecutive_empty_checks > 0 and consecutive_empty_checks % 5 == 0:
                print("\n定期状态更新:")
                device_pool_manager.printInfo()
                
        except Exception as e:
            print(f"心跳检查线程出错: {e}")
            import traceback
            traceback.print_exc()
            
        # 等待下一次检查
        time.sleep(device_pool_manager.heartbeat_check_interval)

def handle_device_registration_and_heartbeat(socket, port):
    """在单独的线程中处理设备注册、心跳和状态查询"""
    try:
        print(f"设备注册和心跳服务已启动，监听端口 {port}")
        
        while True:
            try:
                # 接收消息
                message = socket.recv_multipart()
                if not message or len(message) < 2:
                    print("警告: 收到空消息或不完整的消息")
                    continue
                
                # 解析消息
                identifier = message[0]  # 设备标识符
                action = message[1].decode()  # 动作类型
                
                # 根据消息类型获取数据
                if len(message) > 2:
                    data_raw = message[2]
                    try:
                        data = json.loads(data_raw.decode())
                    except:
                        data = {}
                else:
                    data = {}
                
                print(f"收到消息: 标识符={identifier.decode() if isinstance(identifier, bytes) else identifier}, 动作={action}")
                
                # 处理不同类型的消息
                if action == "RegisterIP":
                    # 处理设备注册
                    device_id = data.get("device_id")
                    ip = data.get("ip")
                    role = data.get("role")
                    
                    if not all([device_id, ip, role]):
                        print(f"警告: 设备注册信息不完整: {data}")
                        socket.send_multipart([
                            identifier,
                            b"REGISTRATION_FAILED",
                            b"Missing required fields"
                        ])
                        continue
                    
                    # 添加设备到设备池
                    device = {
                        "device_id": device_id,
                        "ip": ip,
                        "role": role,
                        "device_type": data.get("device_type", "unknown"),
                        "os": data.get("os", "unknown"),
                        "model": data.get("model", None)
                    }
                    
                    print(f"处理设备注册: ID={device_id}, IP={ip}, 角色={role}")
                    
                    with device_pool_manager.lock:
                        # 检查设备是否已存在
                        for existing_device in device_pool_manager.device_pool:
                            if (existing_device.get("device_id") == device_id or 
                                existing_device.get("ip") == ip):
                                print(f"设备已存在: {device_id} ({ip})")
                                socket.send_multipart([
                                    identifier,
                                    b"REGISTRATION_SUCCESSFUL",
                                    b"Device already registered"
                                ])
                                break
                        else:
                            # 设备不存在，添加到设备池
                            device_pool_manager.device_pool.append(device)
                            print(f"新设备注册成功: {device_id} ({ip})")
                            socket.send_multipart([
                                identifier,
                                b"REGISTRATION_SUCCESSFUL",
                                b"Device registered successfully"
                            ])
                
                elif action == "HEARTBEAT":
                    # 处理心跳
                    device_id = data.get("device_id")
                    if not device_id:
                        print("警告: 心跳消息缺少设备ID")
                        continue
                    
                    # 更新设备心跳时间
                    device_pool_manager.update_device_heartbeat(device_id)
                    socket.send_multipart([
                        identifier,
                        b"HEARTBEAT_RECEIVED"
                    ])
                
                else:
                    print(f"未知的消息类型: {action}")
                    socket.send_multipart([
                        identifier,
                        b"UNKNOWN_ACTION",
                        b"Unknown action type"
                    ])
                    
            except zmq.error.Again:
                # 超时，继续循环
                continue
            except Exception as e:
                print(f"处理消息时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    except Exception as e:
        print(f"设备注册和心跳服务出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    try:
        start = time.time()
        context = zmq.Context()
        
        # 创建一个单一的注册/通信/心跳套接字
        PORT = 23456  # 设置统一的服务器端口
        registration_socket = context.socket(zmq.ROUTER)
        registration_socket.bind(f"tcp://*:{PORT}")
        
        # 设置默认模型，防止未定义错误
        requested_model = "bloom560m-int8"  # 默认模型
        
        # 定义常量
        HOST = '0.0.0.0'  # 监听所有网络接口
        running = True  # 控制主线程运行的标志
        
        # 创建设备集合（兼容旧代码）
        devices = deque()
        ip_graph_requested = []  # 存储所有请求设备的IP地址
        
        print("==== 分布式推理系统启动 ====")
        print(f"等待设备注册，初始化阶段超时: {TIMEOUT}秒")
        print(f"正在监听端口: {PORT}")
        
        # 启动设备注册和心跳服务线程
        registration_thread = threading.Thread(
            target=handle_device_registration_and_heartbeat,
            args=(registration_socket, PORT),  # 传递套接字和端口
            daemon=True
        )
        registration_thread.start()
        
        # 启动心跳检查线程
        heartbeat_thread = threading.Thread(
            target=heartbeat_check_thread,
            daemon=True
        )
        heartbeat_thread.start()
        
        # 主线程等待退出信号，同时定期打印设备池状态
        while running:
            time.sleep(10)  # 每10秒打印一次设备池状态
            print("\n当前设备池状态:")
            device_pool_manager.printInfo()
            print(f"初始化完成: {'是' if device_pool_manager.initialization_complete else '否'}")
            
            # 检查设备池完整性
            device_pool_manager.check_device_pool_integrity()
    
    except KeyboardInterrupt:
        print("\n用户中断，程序退出...")
        running = False
    except Exception as e:
        print(f"主线程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print("清理资源...")
        try:
            registration_socket.close()
            context.term()
        except Exception as e:
            print(f"关闭资源时出错: {e}")
        print("程序已退出")

if __name__ == "__main__":
    main()
