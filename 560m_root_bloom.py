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
        with self.lock:
            # 先获取设备的标识信息
            device_id = device_info.get("device_id")
            ip = device_info.get("ip")
            
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
    def update_device_heartbeat(self, device_id):
        """更新设备心跳时间"""
        with self.lock:
            self.device_heartbeats[device_id] = time.time()
    
    def check_device_heartbeats(self):
        """检查所有设备的心跳状态"""
        with self.lock:
            current_time = time.time()
            failed_working_devices = []
            failed_active_devices = []
            
            # 检查工作设备的心跳
            for device in self.working_devices:
                device_id = device.get("device_id") or device["ip"]
                last_heartbeat = self.device_heartbeats.get(device_id, 0)
                
                if current_time - last_heartbeat > self.heartbeat_timeout:
                    failed_working_devices.append(device)
                    print(f"工作设备 {device_id} 心跳超时，可能已故障")
            
            # 检查活跃设备的心跳
            for device in self.device_pool:
                device_id = device.get("device_id") or device["ip"]
                last_heartbeat = self.device_heartbeats.get(device_id, 0)
                
                if current_time - last_heartbeat > self.heartbeat_timeout:
                    failed_active_devices.append(device)
                    print(f"活跃设备 {device_id} 心跳超时，可能已故障")
            
            # 处理工作设备故障
            for device in failed_working_devices:
                self.handle_working_device_failure(device)
            
            # 处理活跃设备故障
            for device in failed_active_devices:
                self.handle_active_device_failure(device)
    
    def handle_working_device_failure(self, device):
        """处理工作设备故障"""
        with self.lock:
            device_id = device.get("device_id") or device["ip"]
            
            # 从工作设备池中移除
            if device in self.working_devices:
                self.working_devices.remove(device)
                print(f"工作设备 {device_id} 已从工作设备池中移除")
            
            # 从活跃任务中移除
            for task_id, devices in list(self.active_devices.items()):
                if device in devices:
                    devices.remove(device)
                    print(f"工作设备 {device_id} 已从任务 {task_id} 中移除")
                    if not devices:  # 如果任务没有设备了，删除该任务
                        del self.active_devices[task_id]
                        print(f"任务 {task_id} 已删除，因为没有可用设备")
            
            # 添加到工作设备故障池
            device["failure_time"] = time.time()
            device["failure_reason"] = "heartbeat_timeout"
            device["device_type"] = "working"
            self.failed_working_devices.append(device)
            print(f"工作设备 {device_id} 已添加到工作设备故障池")
            
            # 从心跳记录中移除
            if device_id in self.device_heartbeats:
                del self.device_heartbeats[device_id]
    
    def handle_active_device_failure(self, device):
        """处理活跃设备故障"""
        with self.lock:
            device_id = device.get("device_id") or device["ip"]
            
            # 从活跃设备池中移除
            if device in self.device_pool:
                self.device_pool.remove(device)
                print(f"活跃设备 {device_id} 已从活跃设备池中移除")
            
            # 添加到活跃设备故障池
            device["failure_time"] = time.time()
            device["failure_reason"] = "heartbeat_timeout"
            device["device_type"] = "active"
            self.failed_active_devices.append(device)
            print(f"活跃设备 {device_id} 已添加到活跃设备故障池")
            
            # 从心跳记录中移除
            if device_id in self.device_heartbeats:
                del self.device_heartbeats[device_id]
    
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
    
    while True:
        print(f"\n正在检查所有设备的心跳状态... 当前时间: {time.time():.2f}")
        before_count = device_pool_manager.get_device_count()
        failed_before = len(device_pool_manager.get_failed_devices())
        
        # 检查心跳
        device_pool_manager.check_device_heartbeats()
        
        # 报告结果
        after_count = device_pool_manager.get_device_count()
        failed_after = len(device_pool_manager.get_failed_devices())
        
        if before_count != after_count or failed_before != failed_after:
           device_pool_manager.printInfo()
        else:
            device_pool_manager.printInfo()
            
        time.sleep(device_pool_manager.heartbeat_check_interval)

def main_thread():
    """主线程，处理任务分配和模型推理请求"""
    global running
    
    # 创建服务器套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    server_socket.settimeout(1)
    
    print(f"Server started on {HOST}:{PORT}")
    
    try:
        while running:
            try:
                conn, addr = server_socket.accept()
                print(f"Connected by {addr}")
                
                data = receive_data(conn)
                if not data:
                    continue
                
                request_type = data.get("type")
                
                if request_type == "inference":
                    # 处理推理请求
                    process_inference_request(data, conn, addr)
                elif request_type == "GET_STATUS":
                    # 处理状态查询请求
                    process_status_query(data, conn, addr)
                else:
                    print(f"Unknown request type: {request_type}")
                    response = json.dumps({"status": "error", "message": "Unknown request type"}).encode()
                    conn.sendall(response)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error in main thread: {str(e)}")
                traceback.print_exc()
    finally:
        server_socket.close()
        print("Main thread stopped")

def process_status_query(data, conn, addr):
    """处理HTTP方式的状态查询请求"""
    try:
        # 使用全局设备池管理器
        global device_pool_manager
        
        # 检查设备池完整性
        device_pool_manager.check_device_pool_integrity()
        
        # 获取设备池状态
        all_devices = device_pool_manager.get_all_devices()
        working_devices = device_pool_manager.get_working_devices()
        failed_working_devices = device_pool_manager.get_failed_working_devices()
        failed_active_devices = device_pool_manager.get_failed_active_devices()
        
        # 构建状态信息
        status_info = {
            "total_devices": len(all_devices) + len(working_devices),
            "working_devices": len(working_devices),
            "active_devices": len(all_devices),
            "failed_working_devices": len(failed_working_devices),
            "failed_active_devices": len(failed_active_devices),
            "active_tasks": device_pool_manager.get_active_task_count(),
            "initialization_complete": device_pool_manager.initialization_complete,
            "working_devices_list": [
                {
                    "id": d.get("device_id", "N/A"),
                    "ip": d.get("ip", "N/A"),
                    "role": d.get("role", "N/A"),
                    "status": "working",
                    "last_heartbeat": device_pool_manager.device_heartbeats.get(d.get("device_id") or d.get("ip"), 0)
                } for d in working_devices
            ],
            "active_devices_list": [
                {
                    "id": d.get("device_id", "N/A"),
                    "ip": d.get("ip", "N/A"),
                    "role": d.get("role", "N/A"),
                    "status": "active",
                    "last_heartbeat": device_pool_manager.device_heartbeats.get(d.get("device_id") or d.get("ip"), 0)
                } for d in all_devices
            ],
            "failed_working_devices": [
                {
                    "id": d.get("device_id", "N/A"),
                    "ip": d.get("ip", "N/A"),
                    "role": d.get("role", "N/A"),
                    "failure_time": d.get("failure_time", "Unknown"),
                    "failure_reason": d.get("failure_reason", "Unknown"),
                    "device_type": "working"
                } for d in failed_working_devices
            ],
            "failed_active_devices": [
                {
                    "id": d.get("device_id", "N/A"),
                    "ip": d.get("ip", "N/A"),
                    "role": d.get("role", "N/A"),
                    "failure_time": d.get("failure_time", "Unknown"),
                    "failure_reason": d.get("failure_reason", "Unknown"),
                    "device_type": "active"
                } for d in failed_active_devices
            ]
        }
        
        # 构建 HTTP 响应
        response_json = json.dumps(status_info)
        http_response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(response_json)}\r\n\r\n{response_json}"
        conn.sendall(http_response.encode())
        
        # 打印状态信息到服务器控制台
        print("\n设备池状态:")
        print(f"  总设备数: {len(all_devices) + len(working_devices)}")
        print(f"  工作设备数: {len(working_devices)}")
        print(f"  活跃设备数: {len(all_devices)}")
        print(f"  工作设备故障数: {len(failed_working_devices)}")
        print(f"  活跃设备故障数: {len(failed_active_devices)}")
        print(f"  活跃任务数: {device_pool_manager.get_active_task_count()}")
        
        # 打印工作设备详情
        if working_devices:
            print("\n工作设备列表:")
            for i, device in enumerate(working_devices):
                device_id = device.get("device_id", "N/A")
                ip = device.get("ip", "N/A")
                role = device.get("role", "N/A")
                print(f"  {i+1}. ID: {device_id}, IP: {ip}, 角色: {role}")
        
        # 打印活跃设备详情
        if all_devices:
            print("\n活跃设备列表:")
            for i, device in enumerate(all_devices):
                device_id = device.get("device_id", "N/A")
                ip = device.get("ip", "N/A")
                role = device.get("role", "N/A")
                print(f"  {i+1}. ID: {device_id}, IP: {ip}, 角色: {role}")
        
        # 打印工作设备故障详情
        if failed_working_devices:
            print("\n工作设备故障列表:")
            for i, device in enumerate(failed_working_devices):
                device_id = device.get("device_id", "N/A")
                ip = device.get("ip", "N/A")
                failure_time = device.get("failure_time", "N/A")
                failure_reason = device.get("failure_reason", "N/A")
                print(f"  {i+1}. ID: {device_id}, IP: {ip}, 故障时间: {failure_time}, 原因: {failure_reason}")
        
        # 打印活跃设备故障详情
        if failed_active_devices:
            print("\n活跃设备故障列表:")
            for i, device in enumerate(failed_active_devices):
                device_id = device.get("device_id", "N/A")
                ip = device.get("ip", "N/A")
                failure_time = device.get("failure_time", "N/A")
                failure_reason = device.get("failure_reason", "N/A")
                print(f"  {i+1}. ID: {device_id}, IP: {ip}, 故障时间: {failure_time}, 原因: {failure_reason}")
    
    except Exception as e:
        print(f"处理状态查询时出错: {e}")
        response = json.dumps({"status": "error", "message": str(e)}).encode()
        http_response = f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {len(response)}\r\n\r\n"
        conn.sendall(http_response.encode() + response)







    
# 创建独立的设备注册和通信处理函数
def handle_device_registration_and_heartbeat():
    """在单独的线程中处理设备注册、心跳和状态查询"""
    global requested_model  # 启用对全局变量的修改
    global devices          # 用于兼容旧代码
    global ip_graph_requested
    
    print(f"设备注册/通信线程已启动，监听端口: 23456")
    print(f"阶段1: 等待初始设备注册 (超时: {TIMEOUT} 秒)...")
    
    ip_graph_requested = []  # 存储所有请求设备的IP地址
    last_received_time = time.time()
    in_initial_phase = True  # 是否处于初始化阶段
    
    while True:  # 持续运行
        if registration_socket.poll(1000):  # 1秒超时
            try:
                # 接收消息
                message_parts = registration_socket.recv_multipart()
                
                if len(message_parts) < 2:
                    print(f"警告: 收到不完整消息，只有 {len(message_parts)} 个部分")
                    continue
                
                identifier = message_parts[0]
                action = message_parts[1].decode('utf-8')
                
                # 获取消息内容（如果有）
                msg_content = message_parts[2] if len(message_parts) > 2 else b"{}"
                current_time = time.time()
                
                if action == "RegisterIP":
                    # 处理设备注册
                    jsonObject = json.loads(msg_content.decode())
                    ip = jsonObject.get("ip")
                    role = jsonObject.get("role")
                    model_request = jsonObject.get("model", None)
                    device_id = jsonObject.get("device_id", None)
                    
                    print(f"收到设备注册请求: IP={ip}, ID={device_id}, 角色={role}")
                    
                    # 创建设备信息
                    device_info = {
                        "ip": ip, 
                        "role": role, 
                        "identifier": identifier,
                        "last_seen": time.time(),
                        "model_request": model_request
                    }
                    
                    # 如果提供了设备ID，则添加到设备信息中
                    if device_id:
                        device_info["device_id"] = device_id
                    
                    # 注册到新的设备池管理器
                    device_pool_manager.register_device(device_info)
                    
                    # 更新设备心跳
                    device_pool_manager.update_device_heartbeat(device_id or ip)
                    
                    # 如果处于初始阶段，还需要将设备添加到旧的设备集合以保持兼容性
                    if in_initial_phase:
                        with devices_pool_lock:
                            device_entry = {"ip": ip, "role": role}
                            if device_id:
                                device_entry["device_id"] = device_id
                            
                            # 避免重复添加
                            exists = False
                            for dev in devices:
                                if (device_id and dev.get("device_id") == device_id) or (not device_id and dev["ip"] == ip):
                                    exists = True
                                    break
                                    
                            if not exists:
                                if role == "header":
                                    devices.appendleft(device_entry)
                                    # 更新请求的模型
                                    if model_request:
                                        requested_model = model_request
                                else:
                                    devices.append(device_entry)
                    
                    # 把设备ID加入请求列表（用于之后向设备发送消息）
                    if identifier not in ip_graph_requested:
                        ip_graph_requested.append(identifier)
                    
                    # 回复设备已注册成功
                    registration_socket.send_multipart([identifier, b"REGISTRATION_SUCCESSFUL", b"Device registered successfully"])
                    print(f"已向设备 {ip} 发送注册成功响应")
                    
                    # 更新最后接收时间（用于初始化阶段计时）
                    if in_initial_phase:
                        last_received_time = current_time
                
                elif action == "HEARTBEAT":
                    # 处理心跳
                    jsonObject = json.loads(msg_content.decode())
                    device_id = jsonObject.get("device_id")
                    # 无论是否有设备ID，都尝试更新心跳
                    identifier_device_id = device_id or identifier.decode()
                    device_pool_manager.update_device_heartbeat(identifier_device_id)
                    # 回复心跳确认
                    registration_socket.send_multipart([identifier, b"HEARTBEAT_ACK", b"Heartbeat received"])
                
                elif action == "GET_STATUS":
                    # 处理状态查询
                    all_devices = device_pool_manager.get_all_devices()
                    working_devices = device_pool_manager.get_working_devices()
                    failed_devices = device_pool_manager.get_failed_devices()
                    
                    # 打印状态信息用于诊断
                    print("\n当前设备池状态:")
                    print(f"  总设备数: {len(all_devices)}")
                    print(f"  工作设备数: {len(working_devices)}")
                    print(f"  故障设备数: {len(failed_devices)}")
                    print(f"  活跃任务数: {device_pool_manager.get_active_task_count()}")
                    
                    # 创建状态信息
                    status_info = {
                        "total_devices": len(all_devices),
                        "working_devices": len(working_devices),
                        "active_devices": len(all_devices) - len(working_devices),
                        "failed_devices": len(failed_devices),
                        "active_tasks": device_pool_manager.get_active_task_count(),
                        "initialization_complete": device_pool_manager.initialization_complete,
                        "devices": [
                            {
                                "id": d.get("device_id", "N/A"),
                                "ip": d.get("ip", "N/A"),
                                "role": d.get("role", "N/A"),
                                "status": "working" if device_pool_manager.is_working_device(d) else "active",
                                "last_heartbeat": device_pool_manager.device_heartbeats.get(d.get("device_id") or d.get("ip"), 0)
                            } for d in all_devices
                        ],
                        "failed_devices": [
                            {
                                "id": d.get("device_id", "N/A"),
                                "ip": d.get("ip", "N/A"),
                                "failure_time": d.get("failure_time", "N/A"),
                                "failure_reason": d.get("failure_reason", "N/A")
                            } for d in failed_devices
                        ]
                    }
                    
                    # 发送状态信息
                    registration_socket.send_multipart([
                        identifier, 
                        b"STATUS_INFO", 
                        json.dumps(status_info).encode('utf-8')
                    ])
                    print("已向客户端发送状态信息")
                
                elif action == "GET_IP_ADDRESSES":
                    # 兼容旧代码中的IP地址获取请求
                    ip_graph_requested.append(identifier)
                    print(f"收到IP地址请求，已添加 {identifier.decode()} 到请求列表")
                
                elif action == "INFERENCE":
                    # 处理推理请求
                    print(f"收到推理请求，来自设备: {identifier.decode()}")
                    # 这里可以添加推理处理代码，现在暂时返回未实现
                    registration_socket.send_multipart([
                        identifier, 
                        b"INFERENCE_RESPONSE", 
                        json.dumps({"status": "error", "message": "推理功能尚未实现"}).encode('utf-8')
                    ])
                
                else:
                    print(f"收到未知操作请求: {action}")
                    registration_socket.send_multipart([identifier, b"ERROR", b"Unknown action"])
            
            except Exception as e:
                print(f"注册/通信线程处理错误: {e}")
                traceback.print_exc()
                continue
        
        # 检查初始化阶段是否结束
        if in_initial_phase and (time.time() - last_received_time > TIMEOUT):
            print(f"\n初始化阶段超时: 已有 {TIMEOUT} 秒未收到新的设备注册请求")
            in_initial_phase = False
            # 将当前设备池中的设备设为工作设备
            device_pool_manager.set_initialization_complete()
            print(f"进入正常运行阶段: 后续注册的设备将只作为活跃设备，不会成为工作设备")
            
            # 初始化阶段结束后，启动模型加载和分配
            if len(device_pool_manager.working_devices) > 0:
                model_thread = threading.Thread(
                    target=handle_model_loading_and_distribution,
                    args=(requested_model, device_pool_manager.working_devices, ip_graph_requested),
                    daemon=True
                )
                model_thread.start()
            else:
                print("警告: 没有工作设备可用，无法启动模型加载和分配")

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

        # 获取监控信息
        ping_latency, bandwidths, TotalMem, AvailMem, flop_speed = monitoring.get_monitor_info()

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
        else:
            raise RuntimeError("分割大小不能为空! 请检查model_card.prepare_to_split()")
    
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

    # 启动独立的设备注册和通信线程
    registration_thread = threading.Thread(
        target=handle_device_registration_and_heartbeat,
        daemon=True
    )
    registration_thread.start()
    
    # 创建设备集合（兼容旧代码）
    devices = deque()
    
    # 主线程等待退出信号
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
        running = False
    
    # 清理资源
    registration_socket.close()
    context.term()
    print("程序已退出")

if __name__ == "__main__":
    start = time.time()
    context = zmq.Context()
    # 创建一个单一的注册/通信/心跳套接字
    registration_socket = server.establish_connection(context, zmq.ROUTER, 23456)
    
    # 设置默认模型，防止未定义错误
    requested_model = "bloom560m-int8"  # 默认模型
    
    # 定义常量
    HOST = '0.0.0.0'  # 监听所有网络接口
    PORT = 5000  # HTTP服务器端口
    running = True  # 控制主线程运行的标志
    
    # 创建设备集合（兼容旧代码）
    devices = deque()
    ip_graph_requested = []  # 存储所有请求设备的IP地址
    
    print("==== 分布式推理系统启动 ====")
    print(f"等待设备注册，初始化阶段超时: {TIMEOUT}秒")
    
    # 启动心跳检查线程
    heartbeat_thread = threading.Thread(
        target=heartbeat_check_thread,
        daemon=True
    )
    heartbeat_thread.start()
    
    # 启动设备注册和通信线程
    registration_thread = threading.Thread(
        target=handle_device_registration_and_heartbeat,
        daemon=True
    )
    registration_thread.start()
    
    # 打印初始设备池状态
    print("\n初始设备池状态:")
    print(f"工作设备: {len(device_pool_manager.working_devices)}个")
    print(f"活跃设备: {len(device_pool_manager.device_pool)}个")
    print(f"工作设备故障: {len(device_pool_manager.failed_working_devices)}个") 
    print(f"活跃设备故障: {len(device_pool_manager.failed_active_devices)}个")
    
    # 启动模型分割和分发线程
    # 注意：初始化阶段结束后，handle_device_registration_and_heartbeat会自动触发set_initialization_complete
    # 并调用handle_model_loading_and_distribution处理模型加载和分配，无需在主线程中显式调用
    
    # 启动HTTP服务器线程处理状态查询和推理请求
    http_server_thread = threading.Thread(
        target=main_thread,
        daemon=True
    )
    http_server_thread.start()
    
    # 主线程等待退出信号，同时定期打印设备池状态
    try:
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
    
    # 清理资源
    print("清理资源...")
    registration_socket.close()
    context.term()
    print("程序已退出")
