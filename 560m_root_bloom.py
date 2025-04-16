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
        # 使用线程安全的数据结构
        self.device_pool = deque()            # 全部已注册活跃设备池（非工作设备）
        self.working_devices = deque()        # 工作设备池（初始阶段注册的设备）
        self.active_devices = {}              # {task_id: device_list} 当前活跃任务使用的设备
        self.failed_working_devices = deque() # 工作设备故障池
        self.failed_active_devices = deque()  # 活跃设备故障池
        self.task_counter = 0
        
        # 使用原子操作来管理设备状态
        self.device_status = {}  # {device_id: {status, last_heartbeat, info}}
        self.device_heartbeats = {}           # 记录设备最后心跳时间
        self.heartbeat_timeout = 30           # 心跳超时时间(秒)
        self.heartbeat_check_interval = 10    # 心跳检查间隔(秒)
        self.initialization_complete = False  # 标记是否完成初始化阶段
    
    def set_initialization_complete(self):
        """标记初始化阶段已完成，将当前设备池中的设备设为工作设备"""
        # 使用原子操作来更新设备池
        self.working_devices = deque(self.device_pool)
        self.device_pool.clear()
        self.initialization_complete = True
        
        print(f"初始化阶段完成！共有 {len(self.working_devices)} 个工作设备")
        # 打印所有工作设备的详细信息
        for i, device in enumerate(self.working_devices):
            device_id = device.get("device_id", "N/A")
            ip = device.get("ip", "N/A")
            role = device.get("role", "N/A")
            print(f"  工作设备 {i+1}: ID={device_id}, IP={ip}, 角色={role}")
        
        # 更新每个工作设备的状态
        for device in self.working_devices:
            device_id = device.get("device_id")
            if device_id:
                self.device_status[device_id] = {
                    "status": "working",
                    "last_heartbeat": time.time(),
                    "info": device.copy()
                }
        
        # 确保更新所有设备的状态
        self.printInfo()
    
    def register_device(self, device_info):
        """注册新设备到设备池"""
        try:
            device_id = device_info.get("device_id")
            ip = device_info.get("ip")
            
            if not device_id or not ip:
                print("错误: 设备注册没有提供ID或IP地址")
                return False
            
            current_time = time.time()
            
            # 更新设备心跳时间和状态（原子操作）
            self.device_heartbeats[device_id] = current_time
            
            # 检查设备是否已存在
            device_exists = False
            device_in_working_pool = False
            
            # 检查工作设备池
            for device in self.working_devices:
                if device.get("device_id") == device_id:
                    device.update(device_info)
                    device_exists = True
                    device_in_working_pool = True
                    print(f"设备已在工作设备池中，已更新: ID={device_id}, IP={ip}")
                    break
            
            # 如果不在工作设备池中，检查活跃设备池
            if not device_exists:
                for device in self.device_pool:
                    if device.get("device_id") == device_id:
                        device.update(device_info)
                        device_exists = True
                        print(f"设备已在活跃设备池中，已更新: ID={device_id}, IP={ip}")
                        break
            
            # 如果已存在，更新设备状态
            if device_exists:
                status = "working" if device_in_working_pool else "active"
                self.device_status[device_id] = {
                    "status": status,
                    "last_heartbeat": current_time,
                    "info": device_info.copy()
                }
                print(f"更新设备状态: ID={device_id}, 状态={status}")
                return True
            
            # 设备不存在，需要添加
            if self.initialization_complete:
                # 初始化完成后，新设备直接添加到活跃设备池
                self.active_devices.append(device_info)
                status = "active"
                print(f"运行阶段 - 新设备已注册为活跃设备: ID={device_id}, IP={ip}")
            else:
                # 初始化阶段，添加到设备池
                self.device_pool.append(device_info)
                status = "working"
                print(f"初始化阶段 - 新设备已注册为工作设备: ID={device_id}, IP={ip}, 角色={device_info.get('role')}")
            
            # 更新设备状态（原子操作）
            self.device_status[device_id] = {
                "status": status,
                "last_heartbeat": current_time,
                "info": device_info.copy()
            }
            
            # 打印设备池状态
            self.printInfo()
            return True
            
        except Exception as e:
            print(f"设备注册时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_device_heartbeat(self, device_id):
        """更新设备心跳时间，使用原子操作"""
        if not device_id:
            print("警告: 尝试更新无效设备ID的心跳")
            return False
        
        current_time = time.time()
        old_time = self.device_heartbeats.get(device_id, 0)
        
        # 使用原子操作更新心跳时间
        self.device_heartbeats[device_id] = current_time
        
        # 如果设备状态存在，更新状态中的心跳时间
        if device_id in self.device_status:
            self.device_status[device_id]["last_heartbeat"] = current_time
        
        # 记录心跳时间差，用于监控
        if old_time > 0:
            time_diff = current_time - old_time
            if time_diff > self.heartbeat_timeout / 2:
                print(f"警告: 设备 {device_id} 心跳间隔较长: {time_diff:.1f}秒")
            else:
                print(f"设备 {device_id} 心跳更新: {time_diff:.1f}秒前")
        else:
            print(f"设备 {device_id} 首次心跳记录")
        
        return True
    
    def printInfo(self):
        print("\n设备池状态:")
        print(f"工作设备: {len(self.working_devices)}个")
        print(f"活跃设备: {len(self.device_pool)}个")
        print(f"工作设备故障: {len(self.failed_working_devices)}个") 
        print(f"活跃设备故障: {len(self.failed_active_devices)}个")
        print(f"初始化状态: {'已完成' if self.initialization_complete else '未完成'}")

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
            current_time = time.time()
            
            # 获取故障前的设备状态
            before_count = {
                'working': len(device_pool_manager.working_devices),
                'active': len(device_pool_manager.device_pool),
                'failed_working': len(device_pool_manager.failed_working_devices),
                'failed_active': len(device_pool_manager.failed_active_devices)
            }
            
            # 不持有锁的情况下收集超时设备
            failed_devices = []
            
            # 检查所有设备的心跳状态
            for device_id, last_heartbeat in list(device_pool_manager.device_heartbeats.items()):
                heartbeat_age = current_time - last_heartbeat
                
                # 如果设备心跳超时，标记为失败
                if heartbeat_age > device_pool_manager.heartbeat_timeout:
                    # 确定设备在哪个池
                    device_status = device_pool_manager.device_status.get(device_id, {}).get("status", "unknown")
                    device_info = device_pool_manager.device_status.get(device_id, {}).get("info", {})
                    
                    print(f"设备 {device_id} 心跳超时 ({heartbeat_age:.1f}秒)，当前状态: {device_status}")
                    
                    if device_info:
                        # 添加失败信息
                        device_info["failure_time"] = current_time
                        device_info["failure_reason"] = f"心跳超时 ({heartbeat_age:.1f}秒)"
                        failed_devices.append((device_id, device_status, device_info.copy()))
                else:
                    device_status = device_pool_manager.device_status.get(device_id, {}).get("status", "unknown")
                    print(f"设备 {device_id} 心跳正常 ({heartbeat_age:.1f}秒)，当前状态: {device_status}")
            
            # 处理超时设备，使用原子操作
            failures_count = 0
            
            for device_id, status, device_info in failed_devices:
                # 根据设备状态处理故障
                if status == "working":
                    # 从工作设备池中移除
                    for i, device in enumerate(device_pool_manager.working_devices):
                        if device.get("device_id") == device_id:
                            device_pool_manager.working_devices.remove(device)
                            device_pool_manager.failed_working_devices.append(device_info)
                            print(f"工作设备 {device_id} 已移至故障池")
                            failures_count += 1
                            break
                elif status == "active":
                    # 从活跃设备池中移除
                    for i, device in enumerate(device_pool_manager.device_pool):
                        if device.get("device_id") == device_id:
                            device_pool_manager.device_pool.remove(device)
                            device_pool_manager.failed_active_devices.append(device_info)
                            print(f"活跃设备 {device_id} 已移至故障池")
                            failures_count += 1
                            break
                
                # 更新设备状态
                if device_id in device_pool_manager.device_status:
                    device_pool_manager.device_status[device_id]["status"] = f"failed_{status}"
            
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
            
            # 如果有状态变化，打印详细信息
            if failures_count > 0 or status_changed:
                print("\n⚠️ 设备池状态发生变化:")
                print(f"  工作设备: {before_count['working']} -> {after_count['working']} 个")
                print(f"  活跃设备: {before_count['active']} -> {after_count['active']} 个")
                print(f"  工作设备故障: {before_count['failed_working']} -> {after_count['failed_working']} 个") 
                print(f"  活跃设备故障: {before_count['failed_active']} -> {after_count['failed_active']} 个")
                
                if failures_count > 0:
                    print(f"\n本次检测到 {failures_count} 个新故障设备")
                
                consecutive_empty_checks = 0
            else:
                consecutive_empty_checks += 1
                if consecutive_empty_checks <= 2:
                    print("\n设备池状态正常 (无变化):")
                    device_pool_manager.printInfo()
                else:
                    print(f"设备池状态正常 (已连续 {consecutive_empty_checks} 次无变化)")
            
            # 每5次无变化检查后，重新打印状态
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
    global ip_graph_requested  # 添加全局声明
    
    try:
        print(f"设备注册和心跳服务已启动，监听端口 {port}")
        # 配置套接字超时，防止阻塞操作
        socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1秒接收超时
        socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1秒发送超时
        
        while True:
            try:
                # 检查socket是否已关闭
                if socket.closed:
                    print("Socket已关闭，退出注册和心跳服务")
                    break
                    
                # 接收消息
                try:
                    message = socket.recv_multipart()
                except zmq.error.Again:
                    # 接收超时，继续循环
                    continue
                
                if not message or len(message) < 2:
                    print("警告: 收到空消息或不完整的消息")
                    continue
                
                # 解析消息
                identifier = message[0]  # 设备标识符
                action = message[1].decode()  # 动作类型
                
                # 安全地显示标识符，避免解码错误
                if isinstance(identifier, bytes):
                    try:
                        id_str = identifier.decode('utf-8')
                    except UnicodeDecodeError:
                        # 如果无法解码为UTF-8，则使用十六进制表示
                        id_str = identifier.hex()
                else:
                    id_str = str(identifier)
                
                print(f"收到消息: 标识符={id_str}, 动作={action}")
                
                # 根据消息类型获取数据
                if len(message) > 2:
                    data_raw = message[2]
                    try:
                        data = json.loads(data_raw.decode())
                    except:
                        data = {}
                else:
                    data = {}
                
                # 根据动作类型处理消息
                if action == "RegisterIP":
                    # 处理设备注册
                    ip = data.get("ip")
                    role = data.get("role")
                    model_request = data.get("model", None)  # 只有header设备会发送model
                    
                    if not all([ip, role]):
                        print(f"警告: 设备注册信息不完整: {data}")
                        socket.send_multipart([
                            identifier,
                            b"REGISTRATION_FAILED",
                            b"Missing required fields"
                        ])
                        continue
                    
                    # 创建设备信息 - 使用唯一标识符的十六进制表示作为设备ID
                    device_id = identifier.hex() if isinstance(identifier, bytes) else str(identifier)
                    
                    device = {
                        "device_id": device_id,
                        "ip": ip,
                        "role": role,
                        "device_type": "mobile",  # 默认设备类型
                        "os": "android",  # 默认操作系统
                        "model": model_request  # 保存请求的模型
                    }
                    
                    print(f"处理设备注册: ID={device_id}, IP={ip}, 角色={role}")
                    
                    # 添加到ip_graph_requested以便后续发送模型
                    if identifier not in ip_graph_requested:
                        ip_graph_requested.append(identifier)
                        print(f"将设备标识符添加到ip_graph_requested")
                    
                    # 注册设备
                    success = device_pool_manager.register_device(device)
                    
                    # 发送响应消息
                    try:
                        if success:
                            # 发送是否需要监控的信号
                            need_monitor = b"True" if not MODEL_EXIST_ON_DEVICE else b"False"
                            socket.send(need_monitor)
                        else:
                            socket.send(b"False")
                    except zmq.error.ZMQError as e:
                        print(f"发送注册响应时出错: {e}")
                
                elif action == "HEARTBEAT" or action == "HeartDetect":
                    # 处理心跳消息 - 使用唯一标识符的十六进制表示作为设备ID
                    device_id = identifier.hex() if isinstance(identifier, bytes) else str(identifier)
                    
                    if not device_id:
                        print("警告: 心跳消息缺少设备ID")
                        socket.send(b"HEARTBEAT_FAILED")
                        continue
                    
                    # 更新心跳时间
                    success = device_pool_manager.update_device_heartbeat(device_id)
                    
                    # 发送响应
                    try:
                        if success:
                            socket.send(b"HEARTBEAT_RECEIVED")
                        else:
                            socket.send(b"HEARTBEAT_FAILED")
                    except zmq.error.ZMQError as e:
                        print(f"发送心跳响应时出错: {e}")
                
                else:
                    print(f"未知的消息类型: {action}")
                    try:
                        pass
                        # socket.send(b"UNKNOWN_ACTION")
                    except zmq.error.ZMQError as e:
                        print(f"发送未知动作响应时出错: {e}")
                    
            except zmq.error.ZMQError as e:
                print(f"ZMQ错误: {e}")
                if socket.closed:
                    print("Socket已关闭，退出注册和心跳服务")
                    break
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
    finally:
        print("设备注册和心跳服务已停止")

def main():
    """主函数，包含设备注册、模型分割和发送功能"""
    global devices        # 引用全局变量
    global ip_graph_requested
    global ip_graph  # 添加全局声明
    
    try:
        start = time.time()
        context = zmq.Context()
        
        # 创建一个单一的注册/通信/心跳套接字
        PORT = 23456  # 设置统一的服务器端口
        registration_socket = context.socket(zmq.ROUTER)
        registration_socket.bind(f"tcp://*:{PORT}")
        
        # 设置注册套接字的超时，只用于注册和心跳
        registration_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1秒接收超时
        registration_socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1秒发送超时
        
        # 设置默认模型，防止未定义错误
        requested_model = "bloom560m-int8"  # 默认模型
        
        # 定义常量
        running = True  # 控制主线程运行的标志
        
        # 初始化全局设备集合
        devices = deque()
        ip_graph_requested = []  # 存储所有请求设备的IP地址
        ip_graph = []  # 初始化ip_graph列表
        
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
        
        # 等待初始化阶段完成
        print("等待初始化阶段完成...")
        initialization_complete = False
        last_registration_time = time.time()  # 记录最后设备注册时间
        device_count = 0  # 记录当前设备数量
        
        # 在主循环中使用超时检查，避免永久阻塞
        while not initialization_complete:
            current_device_count = 0
            
            # 获取当前设备数量并检查变化 - 最短持有锁的时间
            with devices_pool_lock:
                current_device_count = len(device_pool_manager.device_pool)
                initialization_complete = device_pool_manager.initialization_complete
            
            # 如果设备数量发生变化，更新最后注册时间
            if current_device_count > device_count:
                last_registration_time = time.time()
                device_count = current_device_count
                print(f"新设备注册，当前设备数: {device_count}")
            
            # 检查是否超过10秒没有新设备注册
            if time.time() - last_registration_time >= TIMEOUT and not initialization_complete:
                if device_count > 0:  # 确保至少有一个设备
                    # 初始化阶段结束，将当前设备设为工作设备
                    with devices_pool_lock:
                        if not device_pool_manager.initialization_complete:  # 再次检查以避免竞态条件
                            device_pool_manager.set_initialization_complete()
                            # 将设备添加到兼容旧代码的设备集合
                            devices.clear()  # 清空现有设备
                            for device in device_pool_manager.working_devices:
                                device_entry = {
                                    "ip": device.get("ip"),
                                    "role": device.get("role"),
                                    "device_id": device.get("device_id")
                                }
                                if device_entry["role"] == "header":
                                    devices.appendleft(device_entry)
                                else:
                                    devices.append(device_entry)
                            initialization_complete = True
                    print(f"初始化完成，收集到 {device_count} 个工作设备")
                else:
                    print("警告: 初始化超时，但没有设备注册")
                    return  # 如果没有设备，直接退出
            
            # 周期性打印状态
            if time.time() - last_registration_time > 0 and int(time.time() - last_registration_time) % 2 == 0:
                print(f"等待初始化... 距离上次设备注册已过去 {int(time.time() - last_registration_time)} 秒")
                print(f"当前已收集到 {device_count} 个设备")
            
            time.sleep(0.5)  # 减少等待时间，更频繁地检查
        
        if device_count == 0:
            print("初始化失败：没有设备注册")
            return
        
        print(f"初始化阶段结束，工作设备数: {len(device_pool_manager.working_devices)}")
        print(f"准备分割模型和发送模型...")
        
        # ============== 模型分割和发送部分 ==============
        if requested_model:
        # 确定模型和量化选项
            if requested_model == "bloom560m":
                Quntization_Option = False
            elif requested_model == "bloom560m-int8":
                Quntization_Option = True
                requested_model = "bloom560m"  # 内部使用非量化名称
            else:
                print(f"使用默认模型: bloom560m-int8")
                Quntization_Option = True
                requested_model = "bloom560m"
            
            # 检索模型发送目录
            to_send_path = retrieve_sending_dir(root_dir, requested_model, 
                                            quantization_option=Quntization_Option,
                                            residual_connection=residual_connection_option)
            
            # 检查模型目录是否存在
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
                tokenizer_dir = model_card.retreive_tokenizer_path()
                directory_path = os.path.dirname(bytearray_path)

                print(f'bytearray_path: {bytearray_path}')
                print(f'flop_module_path: {flop_module_path}')
                print(f'num_flop: {num_flop}')
                print(f'out_size_map: {out_size_map}')
            
                print(f"模型分割大小: {model_card.split_size}")
                print("使用Round-Robin分配方法")
                for ip in ip_graph_requested:
                    send.send_multipart([ip, b"ready for monitor"])
                # # start monitor
                monitor_instance = monitor.Monitor(monitor_receive_interval, monitor_port, devices, requested_model, \
                                        bytearray_path, flop_module_path, num_flop, runtime_option)
                thread = threading.Thread(target=monitor_instance.start)
                thread.start()

                num_devices = len(devices)
                monitor_instance.is_monitor_ready.wait()  # 等待监控数据就绪

                # 参数
                ping_latency, bandwidths, TotalMem, AvailMem, flop_speed = monitor_instance.get_monitor_info()


                mem_threshold = .7  # set threshold for memory
                TotalMem = [m * mem_threshold for m in TotalMem]
                AvailMem = [m * mem_threshold for m in AvailMem]
                print("-----------------Test Optimizer Function----------------------")
                print("num_devices")
                print(num_devices)
                print("latency")
                print(ping_latency)
                print("bandwidth")
                print(bandwidths)
                print("totalMem")
                print(TotalMem)
                print("AvailMem")
                print(AvailMem)
                print("flop")
                print(flop_speed)

                if model_card.split_size:
                    print("model_card.split_size: ", model_card.split_size)
                    # load_balancer = Optimizer(num_devices=num_devices, num_modules=model_card.split_size)
                    print("we use a round-robin approach")
                else:
                    raise RuntimeError("The number of modules cannot be None! Check model_card.prepare_to_split().")
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
                
                # 分配模块
                initial_module_arrangement = round_robin_module_arrangement(split_size, split_size)
                overlapping_module_arrangement = initial_module_arrangement
                print(f"模块分配方案:\n{initial_module_arrangement}")
                
                # 准备发送模型
                model_dirs = model_card.prepare_model_to_send(module_arrangement=initial_module_arrangement)
                device_module_order = model_card.device_module_arrangement
                device_dir_map = {tuple(device_module_order[i]): model_dirs[i] for i in range(len(model_dirs))}
                ip_device_module_map = {}
                for i in range(len(devices)):
                    ip_device_module_map[devices[i]["ip"].encode("utf-8")] = device_module_order[
                        i]  # .26: [0], .19: [2], ..

                # retreive session for inference
                session = [str(j) for i in device_module_order for j in i]  # [0, 2, 1]

                # sort the order of ip graph for transmission
                ip_module_map = {}
                sorted_device_module_order = sorted(device_module_order)
                final_sorted_device_module = [[0]] * len(sorted_device_module_order)  # [[ip, [0]], [ip, [1]], [ip, [2]]]
                for ip, val in ip_device_module_map.items():
                    if sorted_device_module_order.index(val) == 0:  # for header
                        final_sorted_device_module[0] = [ip, device_dir_map[tuple(val)]]
                    elif sorted_device_module_order.index(val) != 0 and \
                            sorted_device_module_order.index(val) != len(sorted_device_module_order) - 1:
                        insert_index = sorted_device_module_order.index(val)
                        final_sorted_device_module[insert_index] = [ip, device_dir_map[tuple(val)]]
                    else:  # for tailer
                        final_sorted_device_module[-1] = [ip, device_dir_map[tuple(val)]]

                print(f"session index: {session}")

                for d in range(len(final_sorted_device_module)):
                    ip_encode = final_sorted_device_module[d][0]
                    # current only retrieve single module path
                    if final_sorted_device_module[d][1]:
                        print(f"{ip_encode}:{final_sorted_device_module[d][1][0]}")
                        file_cfg[ip_encode] = final_sorted_device_module[d][1][0]
                        ip_graph.append(ip_encode.decode("utf-8"))
                        ip_module.append([ip_encode.decode("utf-8"), file_cfg[ip_encode]])

                to_send_model_path = retrieve_sending_dir(root_dir, requested_model, quantization_option=Quntization_Option,
                                                        residual_connection=residual_connection_option)
                ip_module_json = json.dumps(ip_module)
                session_index_json = json.dumps(session)

                # Save the JSON string to a file
                with open(os.path.join(to_send_model_path, "ip_module.json"), 'w') as file:
                    file.write(ip_module_json)

                with open(os.path.join(to_send_model_path, "session.json"), 'w') as file:
                    file.write(session_index_json)
        else:       
            raise RuntimeError("requested model cannot be None!")    
        # 修改file_cfg JSON文件中的IP地址
        ##################################################################################
        ####################### 3. Sending models and tokenizer to devices ###############
        ##################################################################################
        print("------file_cfg--------")
        print(file_cfg)
        pathLists = []
        for index, device in enumerate(device_pool_manager.working_devices):
            ip = device.get("ip")
            role = device.get("role")
            
            if not Quntization_Option:
                pathList = [str(ip), f"/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom560m_unquantized_res/device{index}/module{index}/module.zip"]
            else:
                pathList = [str(ip), f"/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom560m_quantized_int8_res/device{index}/module{index}/module.zip"]
            
            # if not Quntization_Option:
            #     pathList = [str(ip), f"/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom560m_unquantized_seq/device{index}/module{index}/module.zip"]
            # else:
            #     pathList = [str(ip), f"/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom560m_quantized_int8_seq/device{index}/module{index}/module.zip"]
            
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
            "num_device": len(device_pool_manager.working_devices),
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
        conditions = [threading.Condition() for i in range(len(device_pool_manager.working_devices) + 1)]
        
    

        
        # 创建新的通信套接字，使用monitor_port而不是原来的端口
        try:
            print(f"尝试使用monitor_port: {monitor_port}作为通信端口")
            communication_socket = context.socket(zmq.ROUTER)
            communication_socket.bind(f"tcp://*:{monitor_port}")
            print(f"通信套接字已绑定到monitor_port: {monitor_port}")
        except zmq.error.ZMQError as e:
            print(f"无法绑定到monitor_port: {e}")
            # 尝试使用原始端口
           
               
        
        for i in range(config["num_device"]):
            t = threading.Thread(
                target=root_server.communication_open_close, 
                args=(communication_socket, config, status, conditions, locks)
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
        
        # 主线程等待退出信号，同时定期打印设备池状态
        while running:
            time.sleep(10)  # 每10秒打印一次设备池状态
            print("\n当前设备池状态:")
            device_pool_manager.printInfo()
            print(f"初始化完成: {'是' if device_pool_manager.initialization_complete else '否'}")
            
            
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
            communication_socket.close()
            context.term()
        except Exception as e:
            print(f"关闭资源时出错: {e}")
        print("程序已退出")

if __name__ == "__main__":
    main()
