import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# os.environ["DS_ACCELERATOR"]='cpu'
import time
import zmq
import copy
from SecureConnection import root_server
from SecureConnection import server
from SecureConnection import monitor
from SecureConnection.root_server import int_to_bytes,communication_prepare
import threading
import torch
import uuid
import numpy as np
import heapq
import json
import os
from threading import Event, Condition
from collections import deque
from queue import Queue, Empty
from collections import deque
from util.model_card import available_models, ModelCard, retrieve_sending_dir, retrieve_sending_info, retrieve_file_cfg
from system_pipeline.onnx_backend.optimization import Optimizer
import socket
import traceback
import datetime
from web_monitor import start_web_server

import queue

monitor_receive_interval = 10  # set intervals for receiving monitor info from clients
monitor_port = "34568"  # set server port to receive monitor info
monitor_port_2 = "34567"
active_device_port = "23457"  # port for active device communication
TIMEOUT =10 # Time to wait for new devices to connect to servers
MODEL_EXIST_ON_DEVICE = True  # set True if the model exists on the mobile device, will skip model creation and transmission
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
device_identifiers_map = {}  # 存储设备ID与其ZMQ标识符的映射: {device_id: identifier}
device_identifiers_lock = threading.Lock()  # 标识符映射的线程锁
import global_config
global Quntization_Option

#
# 1. 目前一组设备能够正常推理
# 2. 需要做到两组设备多组设备也能够正常注册并推理（但是目前是不是不好做啊，手机不太够）
    # 2.1 也就是建组线程
    # 2.2 首先需要多两台手机（1h） 连接校园网，登录机器，然后开始传输权重
    # 好慢啊，终于搞定了

    # 2.3 写一下建立多组并且隔绝的机制
    # 现在是建立多组的机制，检查一下建组逻辑
    # 现在建组是隔绝的端口的，每一组有不同的端口，所以需要对端口进行通信啊

# 2. 但是活跃设备池的注册和设备恢复仍然存在问题
# 3. 设备注册的时候，是不区分状态的，需要让客户端等待
# 4. 设备要建组的时候，才将这一部分设备的状态设置为 working，需要重构一下逻辑
# 5. 有设备故障的时候，将这一部分设备的状态置为恢复中，然后通信相关的IP和端口，加载相关的内容，再恢复通信
# 5.


# 定义活跃设备的通信函数
def communication_open_close_active(sender, config, device_id, status, lock, open=True):
    """
    处理活跃设备的控制信息交换
    类似于communication_open_close但专门用于活跃设备
    """
    client_id = None
    with device_identifiers_lock:
        if device_id in device_identifiers_map:
            client_id = device_identifiers_map[device_id]
    
    if not client_id:
        print(f"错误: 无法找到设备 {device_id} 的标识符")
        return
    
    device_ip = None
    for device in device_pool_manager.active_devices:
        if device.get("device_id") == device_id:
            device_ip = device.get("ip")
            break
    
    if not device_ip:
        print(f"错误: 无法找到设备 {device_id} 的IP地址")
        return
    
    print(f"开始活跃设备 {device_id} ({device_ip}) 的通信线程")
    
    # 设置长超时时间 (30秒)
    original_timeout = sender.getsockopt(zmq.RCVTIMEO)
    sender.setsockopt(zmq.RCVTIMEO, 30000)  # 设置为30秒
    
    try:
        while True:
            print('enter communication open close active')
            with lock[0]:
                # print('开始接收')
                try:
                    info = sender.recv_multipart()  # 使用阻塞方式，但有30秒超时
                except zmq.error.Again:
                    # 超时后继续尝试
                    print(f"活跃设备 {device_id} 接收超时，继续等待...")
                    continue
            
            # 以下是成功接收到消息的处理
            client_id = info[0]
            msg = info[1]
            print(client_id + msg)
            # print("收到信号")
            
            ## Ready
            if open and msg == b'Ready':
                print("Status Ready")
                ## Open
                if len(info) != 3:
                    print("Error")
                
                config["ids"][client_id] = info[2]
                print(config["ids"])
                
                status[client_id] = b'Ready'
                
                sender.send_multipart([client_id, b'Open',
                                      config["graph"],
                                      config["session_index"],
                                      config["task_type"],
                                      config["core_pool_size"],
                                      config["num_sample"],
                                      config["max_length"],
                                      json.dumps(config["dependency"]).encode(),
                                      int_to_bytes(config['num_device']),
                                     ])
                
                status[client_id] = b'Open'
                print(f"Status: Open {config['ids'][client_id]}")
                
                ## Prepare
                sender.send_multipart([client_id, b'Prepare'])
                status[client_id] = b"Prepare"
                communication_prepare(sender, config, client_id, status)
                
                print(f"Status: Prepare {config['ids'][client_id]}")
            
            ## Initialized
            elif msg == b'Initialized':
                status[client_id] = b'Initialized'
                print(f"Status: Initialized {config['ids'][client_id]}")
                
                ## WaitingRecovery
                sender.send_multipart([client_id, b"WaitingRecovery"])
                status[client_id] = b'WaitingRecovery'
                
                print(f"Status: WaitingRecovery {config['ids'][client_id]}")
              
            elif msg == b'Finish':
                status[client_id] = b'Close'
                
                sender.send_multipart([client_id, b"Close"])
                print(f"Close {config['ids'][client_id]}")
                break
            
            elif msg == b'Recovery':
                status[client_id] = b'Recovery'
                print(f"Status: Recovery {config['ids'][client_id]}")
                sender.send_multipart([client_id,
                                      config["graph"],
                                      config["session_index"],
                                      json.dumps(config["dependency"]).encode(),
                                    ])
            
            elif msg == b'WaitingStart':
                status[client_id] = b'WaitingStart'
                status["status"] = b'WaitingStart'
                print(f"Status: WaitingStart {config['ids'][client_id]}")
                while True:
                    time.sleep(1)
                    if "status" in global_config.working_device_status.keys() and global_config.working_device_status["status"] == b'WaitingStart':
                       
                    # 发送开始推理信号
                        sender.send_multipart([client_id, b'Start'])
                        status[client_id] = b'Start'
                        print(f"活跃设备Status: Start {config['ids'][client_id]}")
                        break
    except Exception as e:
        print(f"活跃设备 {device_id} 通信线程出错: {e}")
        traceback.print_exc()
    finally:
        # 恢复原来的超时设置
        try:
            sender.setsockopt(zmq.RCVTIMEO, original_timeout)
        except:
            pass
        print(f"活跃设备 {device_id} 通信线程结束")

# 添加设备池管理类
class DevicePoolManager:
    def __init__(self):
        # 使用线程安全的数据结构
        self.device_pool = deque()            # 全部已注册活跃设备池
        self.active_devices = deque() # 尚未工作的设备
        self.working_devices = deque() # 正在工作的设备(包含候选工作设备）
        self.alt_working_devices = deque # 候选工作的设备
        # self.single_working_group = dict() # 每一组包含一个header，worker，待恢复worker
        self.working_groups = dict() # 包含single_working_group
        self.failed_active_devices = deque() # 包含所有的活跃设备池故障
        self.failed_working_devices = deque()  # 包含所有的活跃设备池故障


        self.task_counter = 0
        
        # 使用原子操作来管理设备状态
        self.device_status = {}  # {device_id: {status, last_heartbeat, info}}
        self.device_heartbeats = {}           # 记录设备最后心跳时间
        self.heartbeat_timeout = 3          # 心跳超时时间(秒)
        self.heartbeat_check_interval = 1   # 心跳检查间隔(秒)
        self.create_group_check_interval = 10  # 心跳检查间隔(秒)
        self.initialization_complete = False  # 标记是否完成初始化阶段
        self.active_device_threads = {}       # 存储活跃设备通信线程


    
    def register_device(self, device_info):
        """注册新设备到设备池"""
        try:
            device_id = device_info.get("device_id")
            ip = device_info.get("ip")
            role = device_info.get("role")
            status = device_info.get("status")
            status = "working" # 先强制设置为工作状态
            if not device_id or not ip:
                print("错误: 设备注册没有提供ID或IP地址")
                return False
            
            current_time = time.time()
            
            # 更新设备心跳时间和状态（原子操作）
            self.device_heartbeats[device_id] = current_time
            
            # 检查设备是否已存在
            device_exists = False
            device_in_working_pool = False

            # 设备不存在，需要添加



            self.active_devices.append(device_info)



            print(f"新设备已注册为活跃设备: ID={device_id}, IP={ip},status = {status}")


            # 更新设备状态（原子操作）
            self.device_status[device_id] = {
                "status": status,
                "last_heartbeat": current_time,
                "info": device_info.copy()
            }

            # 注册心跳，只要放到设备池中，就会有心跳，不过APP端需要修改，让心跳响应立即运行

            # 打印设备池状态
            self.printInfo()
            return status
            
        except Exception as e:
            print(f"设备注册时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start_create_device_group_thread(self, single_working_group,registration_socket,port):
        """创建设备组"""
        global  Quntization_Option, requested_model
        group_id = str(uuid.uuid4())[:8]  # 唯一组ID
        print(f"启动独立组 {group_id}")
        try:
            header_device = single_working_group.get()  # 取出header（阻塞直到有）
            worker_device0 = single_working_group.get()  # 取出worker
            list_of_devices = [header_device,worker_device0]
            # 首先获取原始的ip_module信息
            header_device_ip = header_device.get("ip", "")
            worker_device0_ip = worker_device0.get("ip", "")

            # 需要设定一下端口号，尽量可以和IP进行一个映射，这样可以进行通信，会有多个设备，每个设备都有一个端口号

            group_config = {
                "num_sample": b'1000',
                "max_length": b'100',
                "task_type": "generation".encode('utf-8'),
                "core_pool_size": b'1',
                "skip_model_transmission": MODEL_EXIST_ON_DEVICE,
                "model_name": requested_model,
                "reload_sampleId": None,
                "onnx": True,
                "ids": {},
                "dependency": {},  # 后面填充
                "group_id": None  # 组特定，后面设
            }
            print("初始化global config模板")

            # 深拷贝为组局部（完整独立）

            group_config["group_id"] = group_id
            group_status = {}  # 局部状态
            # 要进行模型分割和发送，加载等
            # ============== 模型分割和发送部分 ==============
            if requested_model:
                # 确定模型和量化选项
                if requested_model == "bloom560m":
                    global Quntization_Option
                    Quntization_Option = False

                elif requested_model == "bloom560m-int8":
                    Quntization_Option = True
                    requested_model = "bloom560m"  # 内部使用非量化名称

                else:
                    print(f"使用默认模型: bloom560m")

                    Quntization_Option = False

                    requested_model = "bloom560m"

                # 检索模型发送目录
                to_send_path = retrieve_sending_dir(root_dir, requested_model,
                                                    quantization_option=Quntization_Option,
                                                    residual_connection=residual_connection_option)

                # 检查模型目录是否存在
                if os.path.isdir(to_send_path):
                    print('模型目录已存在，使用现有模型')
                    # 加载现有的IP模块映射和会话信息

                    # 创建修改后的ip_module，将第二个IP替换为活跃设备的IP
                    modified_ip_module = [
                        [header_device_ip,
                         f"/Users/amstroy/Downloads/Linked-small/onnx_model__/to_send/bloom560m_unquantized_res/device0/module0/module.zip"],
                        [worker_device0_ip,
                         f"/Users/amstroy/Downloads/Linked-small/onnx_model__/to_send/bloom560m_unquantized_res/device1/module1/module.zip"]
                    ]

                    print(f"为活跃设备 {header_device_ip}   {worker_device0_ip} 创建修改后的ip_module:")
                    print(modified_ip_module)




                    with open(os.path.join(to_send_path, 'session.json'), 'r') as file:
                        session_index_json = file.read()


                    global session
                    session = json.loads(session_index_json)
                    file_cfg = retrieve_file_cfg(modified_ip_module)

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
                    final_sorted_device_module = [[0]] * len(
                        sorted_device_module_order)  # [[ip, [0]], [ip, [1]], [ip, [2]]]
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

                    to_send_model_path = retrieve_sending_dir(root_dir, requested_model,
                                                              quantization_option=Quntization_Option,
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

            for index, device in enumerate(list_of_devices):
                ip = device.get("ip")
                role = device.get("role")

                if not Quntization_Option:
                    print(f"使用非量化模型: bloom560m")
                    pathList = [str(ip),
                                f"/Users/amstroy/Downloads/Linked-small/onnx_model__/to_send/bloom560m_unquantized_res/device{index}/module{index}/module.zip"]
                else:
                    pathList = [str(ip),
                                f"/Users/amstroy/Downloads/Linked-small/onnx_model__/to_send/bloom560m_quantized_int8_res/device{index}/module{index}/module.zip"]


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
            group_config.update({
                "file_path": file_cfg,
                "num_device": len(list_of_devices),
                "head_node": ip_graph[0],
                "tail_node": ip_graph[-1],
                "dependency": dependencyMap,
                "session_index": ";".join(session).encode('utf-8'),
                "graph": ",".join(ip_graph).encode('utf-8'),
            })
            # 读取依赖关系JSON文件
            for idx, fPath in dependencyMap.items():
                file = open(fPath, "r")
                data = json.load(file)
                group_config["dependency"][idx] = data

            print(f"组{group_id} config: {group_config}")
            print("配置完成，准备发送模型...")
            # 启动通信线程

            # 通信线程准备
            comm_locks = [threading.Lock(), threading.Lock()]  # 组锁
            comm_conditions = [Condition(comm_locks[0]), Condition(comm_locks[1]), Condition(comm_locks[0])]  # 3个条件
            # 创建新的通信套接字，使用monitor_port而不是原来的端口
            context_communication = zmq.Context()

            # registration_socket.send_multipart([ip, b"True"])
            try:
                print(f"尝试使用monitor_port: {monitor_port}作为通信端口")
                communication_socket = context_communication.socket(zmq.ROUTER)
                communication_socket.bind(f"tcp://*:{monitor_port}")
                communication_socket.setsockopt(zmq.RCVTIMEO, 1000)
                communication_socket.setsockopt(zmq.SNDTIMEO, 1000)
                print(f"✅ 通信套接字已绑定到monitor_port: {monitor_port}")


            except zmq.error.ZMQError as e:
                print(f"❌ 无法绑定到monitor_port: {e}")

                # 清理第一个socket
                if 'communication_socket' in locals():
                    communication_socket.close()

                try:
                    print(f"尝试使用备用端口monitor_port_2: {monitor_port_2}")
                    communication_socket = context_communication.socket(zmq.ROUTER)
                    communication_socket.bind(f"tcp://*:{monitor_port_2}")
                    communication_socket.setsockopt(zmq.RCVTIMEO, 1000)
                    communication_socket.setsockopt(zmq.SNDTIMEO, 1000)
                    print(f"✅ 通信套接字已绑定到备用端口: {monitor_port_2}")

                except zmq.error.ZMQError as e2:
                    print(f"❌ 备用端口monitor_port_2也绑定失败: {e2}")
                    context_communication.term()
                    raise Exception(f"所有端口绑定失败: {monitor_port}, {monitor_port_2}") from e2

            print("group_status:",group_status)
            comm_threads = []
            for i in range(len(list_of_devices)):
                t = threading.Thread(
                    target=root_server.communication_open_close,
                    args=(communication_socket, group_config, group_status, comm_conditions, comm_locks )
                )
                t.daemon = True  # daemon：并行，非阻塞
                comm_threads.append(t)
                t.start()

            # 非阻塞监控（防卡）
            # 无限监控（长期循环版：无timeout，定期打印状态）
            print(f"组 {group_id} 进入长期运行模式（监控中）")
            while True:  # 改为无限
                current_statuses = list(group_status.values())
                if current_statuses and all(v == b'Finish' for v in current_statuses):
                    print(f"组 {group_id} 完成（罕见）")
                    break  # 如果有Finish，退出
                elif current_statuses:
                    running_count = sum(1 for v in current_statuses if v == b'Running')
                    print(f"组 {group_id} 运行中: {running_count}/{len(current_statuses)} 设备Running")
                else:
                    print(f"组 {group_id} 等待初始状态...")

                time.sleep(60)  # 延长到30s，减日志噪音（或60s）
            # 清理
            for t in comm_threads:
                if t.is_alive():
                    t.join(timeout=10)
            communication_socket.close()
            context_communication.term()



        except KeyboardInterrupt:
            print("\n用户中断，程序退出...")

        except Exception as e:
            print(f"通信线程出错: {e}")
            import traceback
            traceback.print_exc()

    def start_active_device_thread(self, device_id):
        """为活跃设备启动控制通信线程"""
        try:
            # 检查线程是否已经存在
            if device_id in self.active_device_threads and self.active_device_threads[device_id].is_alive():
                print(f"活跃设备 {device_id} 已有通信线程在运行")
                return
            

            
            # 状态将在通信过程中由消息交换决定，不预先设置
            
            # 获取活跃设备信息
            active_device_info = None
            for device in self.active_devices:
                if device.get("device_id") == device_id:
                    active_device_info = device
                    break
            
            if not active_device_info:
                print(f"错误: 无法找到活跃设备 {device_id} 的信息")
                return
            
            # 获取头节点（工作设备池的第一个设备）
            head_device = None
            if self.working_devices:
                head_device = self.working_devices[0]
            else:
                print(f"警告: 工作设备池为空，无法确定头节点")
                return
            
            # 首先获取原始的ip_module信息
            active_device_ip = active_device_info.get("ip", "")
            
            # 获取全局配置
            global config, root_dir, Quntization_Option, requested_model
            
            # 确定模型名称和量化选项
           
            
            
            # 创建修改后的ip_module，将第二个IP替换为活跃设备的IP
            modified_ip_module = [
                [head_device.get("ip", ""), f"onnx_model__/to_send/bloom560m_unquantized_res/device0/module0/module.zip"],
                [active_device_ip, f"onnx_model__/to_send/bloom560m_unquantized_res/device1/module1/module.zip"]
            ]
            
            print(f"为活跃设备 {device_id} 创建修改后的ip_module:")
            print(modified_ip_module)
            
            # 获取送信目录
            to_send_path = retrieve_sending_dir(root_dir, requested_model, 
                                           quantization_option=Quntization_Option,
                                           residual_connection=residual_connection_option)
            
            # 使用retrieve_file_cfg获取文件配置
            file_cfg = retrieve_file_cfg(modified_ip_module)
            
            # 使用retrieve_sending_info获取图和依赖信息
            ip_graph, dependencyMap = retrieve_sending_info(
                root_dir, requested_model, 
                ip_module_list=modified_ip_module,
                quantization_option=Quntization_Option,
                residual_connection=residual_connection_option
            )
            
            # 创建会话索引
            session = ["0", "1"]  # 简单的会话索引
            
            # 创建预填充的device_config
            device_config = {
                "device_id": device_id,
                "ids": {}, 
                "file_path": file_cfg,
                "head_node": ip_graph[0],
                "tail_node": ip_graph[-1],
                "graph": ",".join(ip_graph).encode('utf-8'),
                "session_index": ";".join(session).encode('utf-8'),
                "task_type": b"generation",
                "core_pool_size": b"1",
                "num_sample": b"1000",
                "max_length": b"500",
                "num_device": 2,  # 头节点和活跃设备共2个
                "skip_model_transmission": True,
                "dependency": dependencyMap
            }
            for idx, fPath in dependencyMap.items():
                file = open(fPath, "r")
                data = json.load(file)
                device_config["dependency"][idx] = data
            print(f"device_config: {device_config}")
            # # 添加设备ID映射
            # device_config["ids"][head_device.get("device_id", "")] = head_device.get("ip", "").encode('utf-8')
            # device_config["ids"][device_id] = active_device_ip.encode('utf-8')
            
            # 打印生成的配置信息
            print(f"为活跃设备 {device_id} 创建预填充配置:")
            print(f"  头节点: {device_config['head_node']}")
            print(f"  尾节点: {device_config['tail_node']}")
            print(f"  IP图: {device_config['graph']}")
            print(f"  会话索引: {device_config['session_index']}")
            
            # 创建通信线程
            global active_socket
            thread = threading.Thread(
                target=communication_open_close_active,
                args=(active_socket, device_config, device_id, global_config.active_device_status, [threading.Lock(), threading.Lock()]),
                daemon=True
            )
            thread.name = f"ActiveDevice-{device_id}"
            thread.start()
            
            # 保存线程引用
            self.active_device_threads[device_id] = thread
            print(f"已为活跃设备 {device_id} 启动通信线程")
            
        except Exception as e:
            print(f"为活跃设备 {device_id} 启动通信线程时出错: {e}")
            traceback.print_exc()
    
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
                pass
                # print(f"设备 {device_id} 心跳更新: {time_diff:.1f}秒前")
        else:
            print(f"设备 {device_id} 首次心跳记录")
        
        return True
    
    def printInfo(self):
        print("\n设备池状态:")
        print(f"工作设备: {len(self.working_devices)}个")
        print(f"活跃设备: {len(self.active_devices)}个")
        print(f"工作设备故障: {len(self.failed_working_devices)}个") 
        print(f"活跃设备故障: {len(self.failed_active_devices)}个")

        # 打印活跃设备线程状态
        if hasattr(self, 'active_device_threads') and self.active_device_threads:
            active_threads = sum(1 for t in self.active_device_threads.values() if t.is_alive())
            print(f"活跃设备通信线程: {active_threads}/{len(self.active_device_threads)}个")

# 创建设备池管理器实例
device_pool_manager = DevicePoolManager()

import time  # 假设已import
import uuid  # 假设已import
from collections import deque  # 假设已import
import queue  # 假设已import，用于Queue


def create_group_thread_t(socket, port):
    """建组进程：持续检查活跃设备池，定期创建设备组"""
    print("建组进程线程已启动")

    try:
        while True:
            try:
                # 打印当前组情况
                print("当前组情况\n个数：{}\n详细情况：{}".format(
                    len(device_pool_manager.working_groups.keys()),
                    device_pool_manager.working_groups
                ))

                # 统计数量
                active_header_devices = 0
                active_worker_devices = 0
                for device_info in device_pool_manager.active_devices:
                    if device_info.get("role") == "header":
                        active_header_devices += 1
                    if device_info.get("role") == "worker":
                        active_worker_devices += 1

                need_active_worker_devices = 1
                print(
                    f"活跃设备统计: Header={active_header_devices}, Worker={active_worker_devices}, 所需Worker>={need_active_worker_devices}")

                if active_header_devices >= 1 and active_worker_devices >= need_active_worker_devices:
                    print("条件满足，开始创建新组...")

                    # 创建新组
                    single_working_group = queue.Queue()
                    worker_index = 0
                    header_index = 0
                    devices_to_remove = []
                    active_header_device = None
                    active_worker_device = None

                    # 收集设备
                    for i, device_info in enumerate(device_pool_manager.active_devices):
                        if device_info.get("role") == "header" and header_index == 0:
                            active_header_device = device_info
                            header_index += 1
                            devices_to_remove.append(i)
                        elif device_info.get("role") == "worker" and worker_index < 1:
                            active_worker_device = device_info
                            worker_index += 1
                            devices_to_remove.append(i)

                    if active_header_device and active_worker_device:
                        # 从活跃池移除设备
                        active_list = list(device_pool_manager.active_devices)
                        for index in sorted(devices_to_remove, reverse=True):
                            try:
                                removed_device = active_list.pop(index)
                                print(f"从活跃池移除设备: {removed_device.get('device_id', 'unknown')}")
                            except IndexError:
                                print(f"警告: 索引 {index} 无效，跳过移除")

                        device_pool_manager.active_devices = deque(active_list)

                        # 添加到队列
                        single_working_group.put(active_header_device)
                        single_working_group.put(active_worker_device)

                        # 🔥 关键修改：使用线程启动设备组，避免阻塞
                        group_thread = threading.Thread(
                            target=device_pool_manager.start_create_device_group_thread,
                            args=(single_working_group, socket, port)
                        )
                        group_thread.daemon = True
                        group_thread.start()
                        print(f"已启动设备组线程，线程ID: {group_thread.ident}")

                        # 使用动态 groupID
                        groupID = str(uuid.uuid4())
                        device_pool_manager.working_groups[groupID] = single_working_group

                        print(f"新组创建成功！GroupID={groupID}")
                    else:
                        print("警告: 未找到足够设备，无法创建组")

                else:
                    print("条件不满足，跳过创建（Header<1 或 Worker<所需）")

            except Exception as e:
                print(f"建组线程单轮检查出错: {e}")
                import traceback
                traceback.print_exc()

            # 等待下一次检查
            print(f"等待 {device_pool_manager.create_group_check_interval} 秒后下次检查...")
            time.sleep(device_pool_manager.create_group_check_interval)

    except KeyboardInterrupt:
        print("建组线程被用户中断，退出...")
    except Exception as e:
        print(f"建组线程整体出错: {e}")
        import traceback
        traceback.print_exc()
def heartbeat_check_thread():
    """心跳检查线程"""
    print("心跳检查线程已启动，每 {} 秒检查一次设备心跳状态，超时时间 {} 秒".format(
        device_pool_manager.heartbeat_check_interval, 
        device_pool_manager.heartbeat_timeout
    ))
    
    consecutive_empty_checks = 0
    already_processed_failures = set()  # 用于跟踪已经处理过的故障设备
    
    while True:
        try:
            # print(f"\n正在检查所有设备的心跳状态... 当前时间: {time.time():.2f}")
            current_time = time.time()
            
            # 获取故障前的设备状态
            before_count = {
                'active_header_devices': len(device_pool_manager.active_header_devices),
                'active_worker_devices': len(device_pool_manager.active_worker_devices),
                'working_groups': len(device_pool_manager.working_groups),
                'failed_working_header_devices': len(device_pool_manager.failed_working_header_devices),
                'failed_working_worker_devices': len(device_pool_manager.failed_working_worker_devices),
                'failed_working_alt_worker_devices': len(device_pool_manager.failed_working_alt_worker_devices)
            }
            
            # 不持有锁的情况下收集超时设备
            failed_devices = []
            
            # 检查所有设备的心跳状态
            for device_id, last_heartbeat in list(device_pool_manager.device_heartbeats.items()):
                heartbeat_age = current_time - last_heartbeat
                
                # 获取当前设备状态，确保不会有累积的状态前缀
                current_status = device_pool_manager.device_status.get(device_id, {}).get("status", "unknown")
                # 清除可能的重复 failed_ 前缀
                if current_status.startswith("failed_failed_"):
                    clean_status = "failed_" + current_status.split("failed_")[-1]
                    device_pool_manager.device_status[device_id]["status"] = clean_status
                    current_status = clean_status



                # 如果设备心跳超时且不是已经处理过的故障设备，标记为失败
                if heartbeat_age > device_pool_manager.heartbeat_timeout and device_id not in already_processed_failures:
                    # 确定设备在哪个池
                    device_info = device_pool_manager.device_status.get(device_id, {}).get("info", {})
                    
                    print(f"设备 {device_id} 心跳超时 ({heartbeat_age:.1f}秒)，当前状态: {current_status}")
                    
                    if device_info and not current_status.startswith("failed_"):
                        # 添加失败信息
                        device_info["failure_time"] = current_time
                        device_info["failure_reason"] = f"心跳超时 ({heartbeat_age:.1f}秒)"
                        failed_devices.append((device_id, current_status, device_info.copy()))
                    
                    # 将设备添加到已处理集合中，避免重复处理
                    already_processed_failures.add(device_id)
                elif heartbeat_age <= device_pool_manager.heartbeat_timeout and current_status.startswith("failed_"):
                    # 设备恢复正常，从已处理集合中移除
                    if device_id in already_processed_failures:
                        already_processed_failures.remove(device_id)
                    print(f"设备 {device_id} 已恢复正常 ({heartbeat_age:.1f}秒)，之前状态: {current_status}")
                    # 这里可以添加设备恢复的逻辑
                else:
                    pass
                    # print(f"设备 {device_id} 心跳正常 ({heartbeat_age:.1f}秒)，当前状态: {current_status}")
            
            # 处理超时设备，使用原子操作
            failures_count = 0
            
            for device_id, status, device_info in failed_devices:
                # 根据设备状态处理故障
                if status == "working":
                    # 从工作设备池中移除
                    for groupId, single_working_group in  device_pool_manager.working_groups.items():
                        for role,device in single_working_group.items():
                            if device.get("device_id") == device_id:
                                single_working_group.update({})
                                device_pool_manager.failed_working_devices.append(device_info)
                                print(f"工作设备 {device_id} 已移至故障池")
                                failures_count += 1
                                break
                elif status == "active":
                    # 从活跃设备池中移除
                    for i, device in enumerate(device_pool_manager.active_header_devices):
                        if device.get("device_id") == device_id and device_info.get("role")=="header":
                            device_pool_manager.active_header_devices.remove(device)
                            device_pool_manager.failed_active_devices.append(device_info)
                            print(f"活跃设备 {device_id} 已移至故障池")
                            failures_count += 1
                            break
                
                # 更新设备状态，确保状态前缀不会累积
                if device_id in device_pool_manager.device_status:
                    base_status = status.split("_")[-1] if "_" in status else status
                    device_pool_manager.device_status[device_id]["status"] = f"failed_{base_status}"
            
            # 获取故障后的设备状态
            after_count = {
                'working': len(device_pool_manager.working_devices),
                'active': len(device_pool_manager.active_devices),
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
                    # print("\n设备池状态正常 (无变化):")
                    device_pool_manager.printInfo()
                else:
                    pass
                    # print(f"设备池状态正常 (已连续 {consecutive_empty_checks} 次无变化)")
            
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

        
        # 创建一个标志，表示系统是否正在进行故障处理
        system_handling_failure = False
        # 新增：跟踪注册时间和打印间隔
        last_registration_time = time.time()  # 初始为启动时间
        last_interval_print = 0  # 上次打印的间隔秒数
        last_heartbeat_print = time.time()  # 心跳监听状态打印时间
        
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
                    current_time = time.time()
                    time_since_last_reg = current_time - last_registration_time
                    interval = int(time_since_last_reg // 10) * 10  # 最近的10s倍数（10,20,30...）

                    if interval >= 10 and interval > last_interval_print:
                        print(f"距离上一次设备注册已过 {interval}s，无新注册")
                        last_interval_print = interval
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
                
                # print(f"收到消息: 标识符={id_str}, 动作={action}")
                
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
                        # socket.send_multipart([
                        #     identifier,
                        #     b"REGISTRATION_FAILED",
                        #     b"Missing required fields"
                        # ])
                        continue
                    
                    # 创建设备信息 - 使用唯一标识符的十六进制表示作为设备ID
                    device_id = identifier.hex() if isinstance(identifier, bytes) else str(identifier)
                    
                    # 保存设备的标识符，用于后续通信
                    with device_identifiers_lock:
                        device_identifiers_map[device_id] = identifier
                        print(f"设备标识符已保存: {device_id}")
                    
                    device_info = {
                        "device_id": device_id,
                        "ip": ip,
                        "role": role,
                        "device_type": "mobile",  # 默认设备类型
                        "os": "android",  # 默认操作系统
                        "model": model_request , # 保存请求的模型
                        "status": "Registering"
                    }
                    
                    print(f"处理设备注册: ID={device_id}, IP={ip}, 角色={role}")
                    
                    # 添加到ip_graph_requested以便后续发送模型
                    if identifier not in ip_graph_requested:
                        ip_graph_requested.append(identifier)
                        print(f"将设备标识符添加到ip_graph_requested")
                    
                    # 注册设备
                    status = device_pool_manager.register_device(device_info)
                    print("status:",status)
                    # 新增：更新最后注册时间，并打印重置信息
                    last_registration_time = time.time()
                    time_since_last = last_registration_time - last_interval_print / 10 * 10  # 粗略计算上个间隔
                    if time_since_last > 0:
                        print(f"新设备注册，距离上一次注册间隔约 {int(time_since_last)}s")
                    last_interval_print = 0  # 重置打印标记，避免立即重复
                    # 发送响应消息
                    try:
                        if status=="active":
                            # 发送是否需要监控的信号
                             
                            socket.send_multipart([identifier, b"active"])
                         
                            print("发送 active")
                        if status=="working":
                            socket.send_multipart([identifier, b"working"])
                            
                            print("发送 working")
                    except zmq.error.ZMQError as e:
                        print(f"发送注册响应时出错: {e}")



                # 正常工作心跳
                elif action == "HEARTBEAT" or action == "HeartDetect":
                    # 处理心跳消息 - 使用唯一标识符的十六进制表示作为设备ID
                    device_id = identifier.hex() if isinstance(identifier, bytes) else str(identifier)
                    
                    # 更新设备标识符映射
                    with device_identifiers_lock:
                        device_identifiers_map[device_id] = identifier
                    
                    if not device_id:
                        print("警告: 心跳消息缺少设备ID")
                        socket.send_multipart([identifier, b"HEARTBEAT_FAILED"])
                        continue
                    
                    # 更新心跳时间
                    success = device_pool_manager.update_device_heartbeat(device_id)
                    
                    # 发送响应，包含系统状态信息
                    try:
                        if success:
                            # 检查系统是否有故障，但避免在故障处理过程中重复检测
                            if not system_handling_failure:
                                system_has_failures = (
                                    len(device_pool_manager.failed_working_devices) > 0 or 
                                    len(device_pool_manager.failed_active_devices) > 0
                                )
                                
                                if system_has_failures:
                                    # 设置故障处理标志，避免重复触发
                                    system_handling_failure = True
                                    
                                    # 先通知当前心跳的设备
                                    socket.send_multipart([identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_FAILURE"])
                                    print(f"设备 {device_id} 心跳响应：系统存在故障")
                                    
                                    # 异步触发故障处理，避免阻塞心跳响应线程
                                    def trigger_failure_handling():
                                        try:
                                            # 先通知所有在线设备系统故障状态
                                            notify_all_devices = []
                                            device_identifiers = {}
                                            
                                            # 收集所有在线设备的标识符
                                            with device_identifiers_lock:
                                                for dev_id, dev_identifier in device_identifiers_map.items():
                                                    # 排除当前已通知的设备
                                                    if dev_id != device_id:
                                                        notify_all_devices.append(dev_id)
                                                        device_identifiers[dev_id] = dev_identifier
                                            
                                            print(f"正在通知其他 {len(notify_all_devices)} 个设备系统故障状态...")
                                            
                                            # 发送故障通知给所有收集到的设备
                                            for dev_id in notify_all_devices:
                                                try:
                                                    dev_identifier = device_identifiers[dev_id]
                                                    socket.send_multipart([dev_identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_FAILURE"])
                                                    print(f"通知设备 {dev_id} 系统故障状态")
                                                except Exception as e:
                                                    print(f"通知设备 {dev_id} 失败: {e}")
                                            
                                            # 服务器端进入故障处理流程
                                            handle_system_failure()
                                            
                                            # 故障处理完成后，重置标志
                                            nonlocal system_handling_failure
                                            system_handling_failure = False
                                        except Exception as e:
                                            print(f"故障处理过程中出错: {e}")
                                            system_handling_failure = False  # 确保出错时也重置标志
                                    
                                    # 启动一个线程进行故障处理
                                    failure_thread = threading.Thread(target=trigger_failure_handling)
                                    failure_thread.daemon = True
                                    failure_thread.start()
                                else:
                                    # 系统正常
                                    socket.send_multipart([identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_NORMAL"])
                                    # print(f"设备 {device_id} 心跳响应：系统正常")
                            else:
                                # 系统正在处理故障，告知客户端等待
                                socket.send_multipart([identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_HANDLING_FAILURE"])
                                print(f"设备 {device_id} 心跳响应：系统正在处理故障")
                        else:
                            socket.send_multipart([identifier, b"HEARTBEAT_FAILED"])
                            print(f"设备 {device_id} 心跳更新失败")
                    except zmq.error.ZMQError as e:
                        print(f"发送心跳响应时出错: {e}")

                # 后台+亮屏
                elif action == "HEARTBEAT_InBackground_ScreenOn":
                    # 处理心跳消息 - 使用唯一标识符的十六进制表示作为设备ID
                    device_id = identifier.hex() if isinstance(identifier, bytes) else str(identifier)

                    # 更新设备标识符映射
                    with device_identifiers_lock:
                        device_identifiers_map[device_id] = identifier

                    if not device_id:
                        print("警告: 心跳消息缺少设备ID")
                        socket.send_multipart([identifier, b"HEARTBEAT_FAILED"])
                        continue

                    # 不更新心跳时间，强制进入故障状态
                    success = True

                    # 发送响应，包含系统状态信息
                    try:
                        if success:
                            # 检查系统是否有故障，但避免在故障处理过程中重复检测
                            if not system_handling_failure:
                                system_has_failures = (
                                        len(device_pool_manager.failed_working_devices) > 0 or
                                        len(device_pool_manager.failed_active_devices) > 0
                                )

                                if system_has_failures:
                                    # 设置故障处理标志，避免重复触发
                                    system_handling_failure = True

                                    # 先通知当前心跳的设备
                                    socket.send_multipart([identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_InBackground_ScreenOn"])
                                    print(f"设备 {device_id} 心跳响应：SYSTEM_InBackground_ScreenOn,正在进入重启状态")

                                    # 异步触发故障处理，避免阻塞心跳响应线程
                                    def trigger_failure_handling():
                                        try:
                                            # 先通知所有在线设备系统故障状态
                                            notify_all_devices = []
                                            device_identifiers = {}

                                            # 收集所有在线设备的标识符
                                            with device_identifiers_lock:
                                                for dev_id, dev_identifier in device_identifiers_map.items():
                                                    # 排除当前已通知的设备
                                                    if dev_id != device_id:
                                                        notify_all_devices.append(dev_id)
                                                        device_identifiers[dev_id] = dev_identifier

                                            print(f"正在通知其他 {len(notify_all_devices)} 个设备系统故障状态...")

                                            # 发送故障通知给所有收集到的设备
                                            for dev_id in notify_all_devices:
                                                try:
                                                    dev_identifier = device_identifiers[dev_id]
                                                    socket.send_multipart(
                                                        [dev_identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_FAILURE"])
                                                    print(f"通知设备 {dev_id} 系统故障状态")
                                                except Exception as e:
                                                    print(f"通知设备 {dev_id} 失败: {e}")

                                            # 服务器端进入故障处理流程
                                            handle_system_failure()

                                            # 故障处理完成后，重置标志
                                            nonlocal system_handling_failure
                                            system_handling_failure = False
                                        except Exception as e:
                                            print(f"故障处理过程中出错: {e}")
                                            system_handling_failure = False  # 确保出错时也重置标志

                                    # 启动一个线程进行故障处理
                                    failure_thread = threading.Thread(target=trigger_failure_handling)
                                    failure_thread.daemon = True
                                    failure_thread.start()
                                else:
                                    # 系统正常
                                    socket.send_multipart([identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_NORMAL"])
                                    # print(f"设备 {device_id} 心跳响应：系统正常")
                            else:
                                # 系统正在处理故障，告知客户端等待
                                socket.send_multipart(
                                    [identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_HANDLING_FAILURE"])
                                print(f"设备 {device_id} 心跳响应：系统正在处理故障")
                        else:
                            socket.send_multipart([identifier, b"HEARTBEAT_FAILED"])
                            print(f"设备 {device_id} 心跳更新失败")
                    except zmq.error.ZMQError as e:
                        print(f"发送心跳响应时出错: {e}")
                # 处理故障恢复确认
                elif action == "FAILURE_RECOVERY_ACK":
                    device_id = identifier.hex() if isinstance(identifier, bytes) else str(identifier)
                    client_ip = config["ids"].get(identifier, b"unknown").decode() if isinstance(config["ids"].get(identifier), bytes) else config["ids"].get(identifier, "unknown")
                    print(f"设备 {client_ip} (ID: {device_id}) 确认收到故障恢复信号")
                    
                    # 记录确认状态
                    if "recovery_acks" not in config:
                        config["recovery_acks"] = {}
                    config["recovery_acks"][device_id] = True
                    
                    # 检查是否所有设备都已确认
                    expected_count = len(config["ids"]) - sum(1 for ip in config["ids"].values() 
                                                    if (ip.decode() if isinstance(ip, bytes) else ip) in config.get("failed_ips", []))
                    ack_count = len(config["recovery_acks"])
                    
                    print(f"故障恢复进度: {ack_count}/{expected_count} 设备已确认")
                    
                    # 当所有预期的设备都已确认时，可以清除故障状态
                    if ack_count >= expected_count:
                        print("所有设备已确认故障恢复，清除系统故障状态")
                        if "system_status" in config:
                            del config["system_status"]
                        if "recovery_status" in config:
                            del config["recovery_status"]
                        if "failed_ips" in config:
                            del config["failed_ips"]
                        if "recovery_acks" in config:
                            del config["recovery_acks"]
                    
                    # 回复确认收到
                    socket.send_multipart([identifier, b"RECOVERY_ACK_RECEIVED"])
                
                # 处理无替代设备情况的确认
                elif action == "SYSTEM_FAILURE_NO_REPLACEMENT_ACK":
                    device_id = identifier.hex() if isinstance(identifier, bytes) else str(identifier)
                    client_ip = config["ids"].get(identifier, b"unknown").decode() if isinstance(config["ids"].get(identifier), bytes) else config["ids"].get(identifier, "unknown")
                    print(f"设备 {client_ip} (ID: {device_id}) 确认收到系统故障通知(无替代设备)")
                    
                    # 记录确认状态
                    if "suspended_acks" not in config:
                        config["suspended_acks"] = {}
                    config["suspended_acks"][device_id] = True
                    
                    # 检查是否所有设备都已确认
                    expected_count = len(config["ids"]) - sum(1 for ip in config["ids"].values() 
                                                    if (ip.decode() if isinstance(ip, bytes) else ip) in config.get("failed_ips", []))
                    ack_count = len(config["suspended_acks"])
                    
                    print(f"系统暂停进度: {ack_count}/{expected_count} 设备已确认")
                    
                    # 回复确认收到
                    socket.send_multipart([identifier, b"SUSPENSION_ACK_RECEIVED"])
                
                else:
                    print(f"未知的消息类型: {action}")
                    try:
                        socket.send_multipart([identifier, b"UNKNOWN_ACTION"])
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

def handle_system_failure():
    """
    系统故障处理函数
    当检测到系统中有设备故障时调用此函数
    """
    print("系统故障处理开始...")
    global config, ip_graph, session
    
    # 确保traceback已导入
    import traceback
    
    # 记录开始处理时间
    start_time = datetime.datetime.now()
    print(f"故障处理开始时间: {start_time}")
    
    # 不再创建新的communication_socket，而是使用现有的套接字
    # 现有套接字会在communication_open_close函数的循环中使用
    
    try:
        # 1.确定是哪一个设备故障了
        failed_devices = []
        
        # 检查工作设备池中的故障设备
        for device in list(device_pool_manager.failed_working_devices):
            failed_devices.append(device)
            print(f"发现故障工作设备: ID={device.get('device_id')}, IP={device.get('ip')}, 角色={device.get('role')}")
        
        if not failed_devices:
            print("没有检测到故障设备，不需要进行故障处理")
            return
        
        # 如果没有config或者主要变量，表示系统尚未完成初始化
        if not 'config' in globals() or config is None:
            print("系统尚未完成初始化，无法进行故障处理")
            return
            
        print(f"开始处理 {len(failed_devices)} 个故障设备...")
        
        # 构建故障IP列表，保存到config中
        failed_ips = [device.get('ip') for device in failed_devices]
        config["failed_ips"] = failed_ips
        
        # 2.使用活跃设备池中的设备代替故障设备
        replacement_mapping = {}  # {故障设备IP: 替代设备信息}
        
        # 检查活跃设备池是否为空
        if not device_pool_manager.active_devices:
            print("警告: 活跃设备池为空，无法提供替代设备")
            # 虽然无法替换设备，但仍然通知所有运行中的设备有设备故障发生
            try:
                # 设置config中的状态为需要恢复
                config["system_status"] = "RECOVERY_NEEDED"
                config["recovery_status"] = "NO_REPLACEMENT"
                print("系统已标记为需要恢复但无替代设备，通过通信循环通知客户端")
                
                # 即使无法替换设备，也清空故障池，避免反复触发故障处理
                print("清空故障设备池，防止重复处理...")
                device_pool_manager.failed_working_devices.clear()
                device_pool_manager.failed_active_devices.clear()
                
                # 重置设备状态
                for device in failed_devices:
                    device_id = device.get("device_id")
                    if device_id in device_pool_manager.device_status:
                        print(f"重置设备 {device_id} 的状态")
                        device_pool_manager.device_status[device_id]["status"] = "inactive"
                
                return
            except Exception as e:
                print(f"设置故障恢复状态时出错: {e}")
                traceback.print_exc()
                return
        
        for failed_device in failed_devices:
            failed_ip = failed_device.get("ip")
            failed_role = failed_device.get("role")
            failed_idx = -1
            
            # 找出故障设备在IP图中的位置
            for i, ip in enumerate(ip_graph):
                if ip == failed_ip:
                    failed_idx = i
                    break
            
            if failed_idx == -1:
                print(f"警告: 故障设备 {failed_ip} 不在IP图中，跳过")
                continue
            
            # 从活跃设备池中选择一个设备作为替代
            replacement_device = None
            
            if device_pool_manager.active_devices:
                # 从活跃设备池中选择第一个设备
                replacement_device = device_pool_manager.active_devices.popleft()
                print(f"使用活跃设备 {replacement_device.get('ip')} 替代故障设备 {failed_ip}")
                
                # 更新替代设备的角色与故障设备一致
                replacement_device["role"] = failed_role
                
                # 添加到替换映射
                replacement_mapping[failed_ip] = replacement_device
            else:
                print(f"错误: 活跃设备池为空，无法为故障设备 {failed_ip} 找到替代设备")
                # 设置系统需要恢复但无替代设备
                config["system_status"] = "RECOVERY_NEEDED"
                config["recovery_status"] = "NO_REPLACEMENT"
                return
        
        if not replacement_mapping:
            print("没有可用的替代设备，故障处理失败")
            config["system_status"] = "RECOVERY_NEEDED"
            config["recovery_status"] = "NO_REPLACEMENT"
            return
        
        # 3.修改config等的信息
        new_ip_graph = ip_graph.copy()
        new_session = session.copy()
        
        # 更新IP图
        for old_ip, new_device in replacement_mapping.items():
            new_ip = new_device.get("ip")
            
            # 在IP图中替换
            for i, ip in enumerate(new_ip_graph):
                if ip == old_ip:
                    new_ip_graph[i] = new_ip
                    print(f"IP图替换: {old_ip} -> {new_ip} 在位置 {i}")
            
            # 在config中的ids中替换
            for client_id, device_ip in config["ids"].items():
                if device_ip.decode() if isinstance(device_ip, bytes) else device_ip == old_ip:
                    config["ids"][client_id] = new_ip.encode() if isinstance(device_ip, bytes) else new_ip
                    print(f"配置IDs替换: {old_ip} -> {new_ip}")
            
            # 更新头尾节点（如果需要）
            if config["head_node"] == old_ip:
                config["head_node"] = new_ip
                print(f"头节点替换: {old_ip} -> {new_ip}")
                
            if config["tail_node"] == old_ip:
                config["tail_node"] = new_ip
                print(f"尾节点替换: {old_ip} -> {new_ip}")
        
        # 构建新的配置更新
        config["graph"] = ",".join(new_ip_graph).encode('utf-8')
        config["session_index"] = ";".join(new_session).encode('utf-8')
        
        print("配置已更新:")
        print(f"新IP图: {new_ip_graph}")
        print(f"新会话索引: {new_session}")
        print(f"新配置: {config}")
        # 4.准备通过现有通信循环发送故障控制信息
        config["system_status"] = "RECOVERY_NEEDED"
        config["recovery_status"] = "HAS_REPLACEMENT"
        config["new_graph"] = new_ip_graph
        config["new_session"] = new_session
        print("系统已标记为需要恢复，将通过通信循环通知客户端")
        print(f"config(故障恢复后): {config}")
        # 5.添加替代设备到工作设备池
        for old_ip, replacement_device in replacement_mapping.items():
            device_pool_manager.working_devices.append(replacement_device)
            print(f"替代设备 {replacement_device.get('ip')} 添加到工作设备池")
        
        # 清空故障设备池，防止重复触发故障处理
        print("清空故障设备池...")
        device_pool_manager.failed_working_devices.clear()
        device_pool_manager.failed_active_devices.clear()
        
        # 重置已处理的设备标志
        if 'already_processed_failures' in globals():
            already_processed_failures.clear()
            print("重置故障设备处理标记")
            
        print("系统故障处理准备完成，等待发送故障恢复信号")
        
       
        
    except Exception as e:
        print(f"系统故障处理出错: {e}")
        traceback.print_exc()



        """
        
        
            # 创建修改后的ip_module，将第二个IP替换为活跃设备的IP
            modified_ip_module = [
                [head_device.get("ip", ""),
                 f"onnx_model__/to_send/bloom560m_unquantized_res/device0/module0/module.zip"],
                [active_device_ip,
                 f"onnx_model__/to_send/bloom560m_unquantized_res/device1/module1/module.zip"]
            ]

            print(f"为活跃设备 {device_id} 创建修改后的ip_module:")
            print(modified_ip_module)

            # 获取送信目录
            to_send_path = retrieve_sending_dir(root_dir, requested_model,
                                                quantization_option=Quntization_Option,
                                                residual_connection=residual_connection_option)

            # 使用retrieve_file_cfg获取文件配置
            file_cfg = retrieve_file_cfg(modified_ip_module)

            # 使用retrieve_sending_info获取图和依赖信息
            ip_graph, dependencyMap = retrieve_sending_info(
                root_dir, requested_model,
                ip_module_list=modified_ip_module,
                quantization_option=Quntization_Option,
                residual_connection=residual_connection_option
            )

            # 创建会话索引
            session = ["0", "1"]  # 简单的会话索引

            # 创建预填充的device_config
            device_config = {
                "device_id": device_id,
                "ids": {},
                "file_path": file_cfg,
                "head_node": ip_graph[0],
                "tail_node": ip_graph[-1],
                "graph": ",".join(ip_graph).encode('utf-8'),
                "session_index": ";".join(session).encode('utf-8'),
                "task_type": b"generation",
                "core_pool_size": b"1",
                "num_sample": b"1000",
                "max_length": b"500",
                "num_device": 2,  # 头节点和活跃设备共2个
                "skip_model_transmission": True,
                "dependency": dependencyMap
            }
            for idx, fPath in dependencyMap.items():
                file = open(fPath, "r")
                data = json.load(file)
                device_config["dependency"][idx] = data
            print(f"device_config: {device_config}")
            # # 添加设备ID映射
            # device_config["ids"][head_device.get("device_id", "")] = head_device.get("ip", "").encode('utf-8')
            # device_config["ids"][device_id] = active_device_ip.encode('utf-8')

            # 打印生成的配置信息
            print(f"为活跃设备 {device_id} 创建预填充配置:")
            print(f"  头节点: {device_config['head_node']}")
            print(f"  尾节点: {device_config['tail_node']}")
            print(f"  IP图: {device_config['graph']}")
            print(f"  会话索引: {device_config['session_index']}")

            # 创建通信线程
            global active_socket
            thread = threading.Thread(
                target=communication_open_close_active,
                args=(active_socket, device_config, device_id, global_config.active_device_status,
                      [threading.Lock(), threading.Lock()]),
                daemon=True
            )
            thread.name = f"ActiveDevice-{device_id}"
            thread.start()

            # 保存线程引用
            self.active_device_threads[device_id] = thread
            print(f"已为活跃设备 {device_id} 启动通信线程")

        except Exception as e:
            print(f"为活跃设备 {device_id} 启动通信线程时出错: {e}")
            traceback.print_exc()
        
        """

def main():
    """主函数，包含设备注册、模型分割和发送功能"""
    global devices        # 引用全局变量
    global ip_graph_requested
    global ip_graph  # 添加全局声明
    global active_socket  # 添加全局声明
    
    try:
        start = time.time()
        context = zmq.Context()
        
        # 启动Web监控服务器
        # web_thread = threading.Thread(
        #     target=start_web_server,
        #     args=(device_pool_manager,),
        #     daemon=True
        # )
        # web_thread.start()
        # print("Web监控服务器已启动，访问 http://localhost:34568 查看状态")
        #
        # 创建一个单一的注册/通信/心跳套接字
        PORT = 23456  # 设置统一的服务器端口
        registration_socket = context.socket(zmq.ROUTER)
        registration_socket.bind(f"tcp://*:{PORT}")
        
        # 创建活跃设备通信套接字
        active_socket = context.socket(zmq.ROUTER)
        active_socket.bind(f"tcp://*:{active_device_port}")
        print(f"活跃设备通信套接字已绑定到端口: {active_device_port}")
        
        # 设置注册套接字的超时，只用于注册和心跳
        registration_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1秒接收超时
        registration_socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1秒发送超时

        # 设置活跃设备套接字的超时
        active_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1秒接收超时
        active_socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1秒发送超时
        
        # 设置默认模型，防止未定义错误
        global requested_model
        requested_model = "bloom560m"  # 默认模型
        
        # 定义常量
        running = True  # 控制主线程运行的标志
        
        # 初始化全局设备集合
        devices = deque()
        ip_graph_requested = []  # 存储所有请求设备的IP地址
        ip_graph = []  # 初始化ip_graph列表
        
        print("==== 分布式推理系统启动 ====")

        print(f"正在监听端口: {PORT}")
        
        # 启动设备注册和心跳服务线程
        registration_thread = threading.Thread(
            target=handle_device_registration_and_heartbeat,
            args=(registration_socket, PORT),  # 传递套接字和端口
            daemon=True
        )
        registration_thread.start()





        # # 启动心跳检查线程
        # heartbeat_thread = threading.Thread(
        #     target=heartbeat_check_thread,
        #     daemon=True
        # )
        # heartbeat_thread.start()

        # 启动建组线程
        create_group_thread = threading.Thread(
            target=create_group_thread_t,
            args=(registration_socket, PORT),  # 传递套接字和端口
            daemon=True
        )
        create_group_thread.start()


        print("主线程进入等待模式，按 Ctrl+C 退出...")

        running = True  # 移到这里，确保global

        try:
            while running:  # 用running控制循环
                time.sleep(1)  # 每秒检查一次，避免CPU 100%

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
            print("正在关闭 ZeroMQ socket 和上下文...")
            registration_socket.close()
            active_socket.close()
            context.term()
        except Exception as e:
            print(f"关闭资源时出错: {e}")
        print("程序已退出")

if __name__ == "__main__":
    main()
