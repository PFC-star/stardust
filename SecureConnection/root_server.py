''''''
import json
import time
from threading import Thread
import zmq
import global_config
import uuid
from bidict import bidict
"""
    R: Ready
        client -> root; Show the status of edge device
    O: Open
        root -> client; The information of this task, e.g: Training/Inference/Task name, etc. 
    P: Prepare
        root -> client; Send the decentralized model and training/Inference code to clients. 
    I:  Initialized
        client -> root; Models are initialized and training/Inference is ready
    S: Start
        root -> client; Start training/Inference and Data transmission)    
    F: Finish
        client -> root; Finish training/Inference
    C: Close
        root -> client; Close the connection
"""
def int_to_bytes(num):
    # 如果 num 是 0-9 之间的数字，返回对应字符的字节串
    if 0 <= num <= 9:
        return chr(num + ord('0')).encode('utf-8')  # 将整数转为字符 '0' - '9'
    # 其他整数映射为对应的字符（比如映射大写字母等）
    elif 32 <= num <= 126:
        return chr(num).encode('utf-8')  # 对应可打印字符
    else:
        # 对于其他值，返回原始的字节串
        return bytes([num])


def send_model_file(path, sock, client_id, chunked=True, chunk_size=10*1024*1024):
    if not chunked:
        with open(path, 'rb') as f:
            data = f.read()
            sock.send_multipart([client_id, data])
            print("Data is sent")
    else:
        with open(path, 'rb') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    sock.send_multipart([client_id, b''])
                    break
                sock.send_multipart([client_id, chunk])
            print("Data is sent")


def trans_model(sender, config, client_id, status,ip):

    # send "Prepare" (带前缀)
    sender.send_multipart([client_id, b'Prepare'])

    node_ip = config["ids"][client_id]  # 用real_client_id (IP bytes)
    skip_flag = b'True' if config["skip_model_transmission"] else b'False'  # bool转bytes
    sender.send_multipart([client_id, skip_flag])

    if not config["skip_model_transmission"]:
        print(f"send {config['file_path'][node_ip]} to {node_ip}")

        # onnx sends multiple files, which should be a zip
        if config["onnx"]:
            send_model_file(config["file_path"][node_ip], sender, client_id)  # 传带前缀ID
        else:
            send_model_file(config["file_path"][node_ip], sender, client_id)

        # # transmit tokenizer to header (如果需要，同样加前缀)
        # if config["head_node"] == node_ip:
        #     print(f"send tokenizer to {node_ip}")
        #     send_model_file(config['file_path'][b'tokenizer'], sender, prefixed_client_id)

    status[ip]['status'] = "Prepare"  # status用real_client_id (无前缀)

def communication_prepare(sender, config, client_id, status):
    group_id = config.get("group_id", "")  # 从config取组ID
    group_prefix = group_id.encode('utf-8') if group_id else b''
    prefixed_client_id = group_prefix + client_id  # 带前缀ID，用于send路由

    print(f"Prepare for  (prefixed: {prefixed_client_id.hex()[:16]}...)")  # 日志

    # send "Prepare" (带前缀)
    sender.send_multipart([prefixed_client_id, b'Prepare'])

    node_ip = config["ids"][client_id]  # 用real_client_id (IP bytes)
    skip_flag = b'True' if config["skip_model_transmission"] else b'False'  # bool转bytes
    sender.send_multipart([prefixed_client_id, skip_flag])

    if not config["skip_model_transmission"]:
        print(f"send {config['file_path'][node_ip]} to {node_ip}")

        # onnx sends multiple files, which should be a zip
        if config["onnx"]:
            send_model_file(config["file_path"][node_ip], sender, prefixed_client_id)  # 传带前缀ID
        else:
            send_model_file(config["file_path"][node_ip], sender, prefixed_client_id)

        # # transmit tokenizer to header (如果需要，同样加前缀)
        # if config["head_node"] == node_ip:
        #     print(f"send tokenizer to {node_ip}")
        #     send_model_file(config['file_path'][b'tokenizer'], sender, prefixed_client_id)

    status[client_id] = b"Prepare"  # status用real_client_id (无前缀)

def communication_data_transmission(sender, num_devices, head_client_id, status):
    while check_status(status, num_devices, b"Start"):
        pass


def communication_result_transmission(sender, result, num_devices, tail_client_id, status):
    # while check_status(status, num_devices, b"Finish"):
    sender.send_multipart([b"res", result])
    pass


all_status = {b"Init":  -1,
              b"Ready":  0,
              b"Open":   1,
              b"Prepare":2,
              b"Initialized": 3,
              b"Start":  4,
              b"Running":5,
              b"Finish": 6,
              b"Close": 7}




def ConfigCreator(Config, client_id):
    ## Based on the monitor situation
    return Config["graph"]
def bytes_to_ip(ip_bytes):
    """将4字节的IP地址转换为字符串"""
    try:
        if len(ip_bytes) == 4:
            return '.'.join(str(byte) for byte in ip_bytes)
        else:
            return ip_bytes.hex()
    except:
        return ip_bytes.hex()


def check_status(status, config, mode ):
    """检查所有设备是否达到指定状态"""
    if len(status) != config["num_device"]:
        print(f"状态检查: 设备数量不匹配 {len(status)} != {config['num_device']}")
        return False

    all_ready = True
    for device_id, current_status in status.items():
        device_ip = bytes_to_ip(device_id)
        # 假设all_status是一个全局的状态优先级映射
        if all_status.get(current_status, 0) < all_status.get(mode, 0):
            print(f"设备 {device_ip} 状态 {current_status} 未达到 {mode}")
            all_ready = False

    print(f"状态检查 {mode}: {'通过' if all_ready else '未通过'}")
    return all_ready
def check_ipAlign(status, config,mode, list_of_devices_ip):
    """检查所有设备是否达到指定状态"""

    all_ready = True
    for device_ip in list_of_devices_ip:
        current_status = status[device_ip]["status"]
        if  status[device_ip]["status"] != mode:
            print(f"设备 {device_ip} 状态 {current_status} 未达到 {mode}")
            all_ready = False

    print(f"状态检查 {mode}: {'通过' if all_ready else '未通过'}")
    return all_ready

def communication_open_close(sender, config, status, conditions, lock, open=True):
    group_id = config.get("group_id", "")
    while True:
        try:
            print(f'组{group_id} enter communication open close')
            info = None
            try:
                with lock[0]:
                    info = sender.recv_multipart()
            except zmq.Again:
                time.sleep(0.1)
                continue

            if info is None or len(info) < 2:
                continue
            client_id = info[0]
            msg = info[1]
            data = info[2].decode(errors='ignore') if len(info) > 2 and isinstance(info[2], bytes) else str(
                info[2] if len(info) > 2 else '')  # 安全decode
            print(
                f"Recv raw: client_id={client_id.hex()}, msg={msg}, data={info[2].decode(errors='ignore') if len(info) > 2 else 'None'}")

            # 注册分支
            if msg == b'RegisterGroupID':
                if len(info) < 3:
                    print("注册消息不完整，忽略")
                    continue

                ip_str = data
                print(f"注册设备IP: {ip_str}")

                group_id_str = config.get("group_id", str(uuid.uuid4())[:8])
                group_id_bytes = group_id_str.encode('utf-8')

                sender.send_multipart([
                    info[0],
                    b"GROUP_ASSIGNED",
                    group_id_bytes
                ])

                print(f"分配group_id {group_id_str} 到设备 {ip_str} (via {info[0].hex()})")
                continue

            group_prefix = group_id.encode()
            real_client_id = client_id
            if group_prefix and client_id.startswith(group_prefix):
                real_client_id = client_id[len(group_prefix):]
                client_ip = bytes_to_ip(real_client_id)

                if open and msg == b'Ready':
                    print("Status Ready")
                    if len(info) != 3:
                        print("Error")
                        continue

                    config["ids"][real_client_id] = info[2]
                    print(f"当前注册设备: {config['ids']}")

                    status[real_client_id] = b'Ready'

                    # 发送Open和相关配置
                    sender.send_multipart([group_prefix + real_client_id, b'Open',
                                           config["graph"],
                                           config["session_index"],
                                           config["task_type"],
                                           config["core_pool_size"],
                                           config["num_sample"],
                                           config["max_length"],
                                           json.dumps(config["dependency"]).encode(),
                                           int_to_bytes(config['num_device']),
                                           ])

                    status[real_client_id] = b'Open'
                    print(f"Status: Open {config['ids'][real_client_id]}")

                    with conditions[0]:
                        while not check_status(status, config, b"Open"):
                            conditions[0].wait(timeout=60)
                        print("所有设备Open完成，通知所有线程")
                        conditions[0].notify_all()

                    with lock[1]:
                        communication_prepare(sender, config, real_client_id, status)

                    print(f"Status: Prepare {config['ids'][real_client_id]}")

                elif msg == b'Initialized':
                    print(f"=== 开始处理设备 {client_ip} 的Initialized状态 ===")

                    # 关键修复：在锁保护下处理状态更新和检查
                    with lock[1]:
                        status[real_client_id] = b'Initialized'
                        print(f"设置状态: 设备 {client_ip} -> Initialized")

                        # 检查是否所有设备都Initialized了
                        all_initialized = check_status(status, config, b"Initialized")

                        if all_initialized:
                            print("所有设备都已Initialized，开始发送Start消息")
                            # 向所有设备发送Start消息
                            for device_id in config["ids"]:
                                target_device = group_prefix + device_id
                                device_ip = bytes_to_ip(device_id)
                                print(f"向设备 {device_ip} 发送Start")
                                sender.send_multipart([target_device, b"Start"])
                                status[device_id] = b'Start'
                                print(f"设备 {device_ip} 状态设置为Start")
                        else:
                            print(f"设备 {client_ip} 已Initialized，等待其他设备...")

                elif msg == b'Running':
                    print(f"Status: Running {config['ids'][real_client_id]}")
                    status[real_client_id] = b'Running'
                    break

                elif msg == b'Finish':
                    status[real_client_id] = b'Close'
                    with conditions[2]:
                        while not check_status(status, config, b"Close"):
                            conditions[2].wait(timeout=60)
                        conditions[2].notify_all()

                    sender.send_multipart([group_prefix + real_client_id, b"Close"])
                    print(f"Close {config['ids'][real_client_id]}")
                    break

                elif msg == b'Recovery':
                    status[real_client_id] = b'Recovery'
                    print(f"Status: Recovery {config['ids'][real_client_id]}")
                    sender.send_multipart([group_prefix + real_client_id,
                                           config["graph"],
                                           config["session_index"],
                                           json.dumps(config["dependency"]).encode(),
                                           ])

                elif msg == b'WaitingStart':
                    status[real_client_id] = b'WaitingStart'
                    if "status" not in status:
                        status["status"] = b'WaitingStart'

                    print(f"Status: WaitingStart {config['ids'][real_client_id]}")
                    inner_timeout = 60
                    start_inner = time.time()
                    while time.time() - start_inner < inner_timeout:
                        time.sleep(1)
                        if status.get("status", b'') == b'WaitingStart':
                            sender.send_multipart([group_prefix + real_client_id, b'ResumeStart'])
                            status[real_client_id] = b'ResumeStart'
                            break
                    else:
                        print("WaitingStart timeout")
            else:
                print(f"忽略非本组: {client_id.hex()}")
                continue

        except zmq.Again:
            time.sleep(0.1)
            continue

        except Exception as e:
            print(f"通信异常: {e}")
            import traceback
            traceback.print_exc()
            break


# 传输权重，计算图等->加载权重->IP对齐-> 启动推理（恢复推理）
def communication_prepare_model(sender, context_communication,config, status, open=True):
    group_id = config.get("group_id", "")
    client_id_ip = {}
    try:
        while True:
            try:

                info = None
                try:

                    info = sender.recv_multipart()
                except zmq.Again:
                    time.sleep(0.1)
                    continue

                if info is None or len(info) < 2:
                    continue
                client_id = info[0]
                msg = info[1]
                data = info[2].decode(errors='ignore') if len(info) > 2 and isinstance(info[2], bytes) else str(
                    info[2] if len(info) > 2 else '')  # 安全decode
                print(
                    f"Recv raw: client_id={client_id.hex()}, msg={msg}, data={info[2].decode(errors='ignore') if len(info) > 2 else 'None'}")
                if data:
                    client_id_ip[client_id] = data
                if open and msg == b'Ready':

                    communication_information_prepare_(sender, config, status,client_id,info)
                    # 已经信息准备完毕（传输权重，IP计算图等），并且客户端已经开始进行初始化了


                elif msg == b'Initialized':
                    print(f"=== 开始处理设备 {client_id} 的Initialized状态 ===")



                    status[client_id_ip[client_id]]["status"] = 'Initialized'
                    print(f"设置状态: 设备 {status[client_id_ip[client_id]]} -> Initialized")


                    print("设备已Initialized，等待建组")

                    break



            except zmq.Again:
                time.sleep(0.1)
                continue

            except Exception as e:
                print(f"通信异常: {e}")
                import traceback
                traceback.print_exc()
                break


    finally:
        # 子线程自己关闭资源
        try:
            sender.close()
            print("communication_socket 已关闭")
        except:
            pass
        try:
            context_communication.term()  # ← 必须是全局变量！
            print("context_communication 已销毁")
        except:
            pass

def communication_information_prepare(sender, config, status, conditions, lock, group_prefix, real_client_id, client_ip):
    print("Status Ready")
    if len(info) != 3:
        print("Error")
        return

    config["ids"][real_client_id] = info[2]
    print(f"当前注册设备: {config['ids']}")

    status[real_client_id] = b'Ready'

    # 发送Open和相关配置
    sender.send_multipart([group_prefix + real_client_id, b'Open',
                           config["graph"],
                           config["session_index"],
                           config["task_type"],
                           config["core_pool_size"],
                           config["num_sample"],
                           config["max_length"],
                           json.dumps(config["dependency"]).encode(),
                           int_to_bytes(config['num_device']),
                           ])

    status[real_client_id] = b'Open'
    print(f"Status: Open {config['ids'][real_client_id]}")


#   TODO 要不要是个问题？？？
    with conditions[0]:
        while not check_status(status, config, b"Open"):
            conditions[0].wait(timeout=60)
        print("所有设备Open完成，通知所有线程")
        conditions[0].notify_all()

    with lock[1]:
        communication_prepare(sender, config, real_client_id, status)

    print(f"Status: Prepare {config['ids'][real_client_id]}")

    return
def communication_information_prepare_(sender, config, status,   real_client_id,info ):
    print("Status Ready")
    if len(info) != 3:
        print("Error")
        return

    config["ids"][real_client_id] = info[2]
    ip =  info[2].decode('utf-8')
    status[ip]["info"].update({"identifier_2":real_client_id})
    print(f"当前注册设备: {config['ids']}")

    status[ip]['status'] = 'Ready'


    # 发送Open和相关配置
    sender.send_multipart([real_client_id, b'Open',
                           config["graph"],
                           config["session_index"],
                           config["task_type"],
                           config["core_pool_size"],
                           config["num_sample"],
                           config["max_length"],
                           json.dumps(config["dependency"]).encode(),
                           int_to_bytes(config['num_device']),
                           ])

    status[ip]['status'] = 'Open'
    print(f"Status: Open {config['ids'][real_client_id]}")



    trans_model(sender, config, real_client_id, status,ip)

    print(f"Status: Prepare {config['ids'][real_client_id]}")

    return
def communication_information_prepare_ip(sender, config, status,   real_client_id,info ):
    print("Status Ready")
    if len(info) != 3:
        print("Error")
        return

    config["ids"][real_client_id] = info[2]
    ip =  info[2].decode('utf-8')
    status[ip]["info"].update({"identifier_2":real_client_id})
    print(f"当前注册设备: {config['ids']}")

    status[ip]['status'] = 'Ready'


    # 发送Open和相关配置
    sender.send_multipart([real_client_id, b'IPAligning',
                           config["graph"],
                           config["session_index"],
                           config["task_type"],
                           config["core_pool_size"],
                           config["num_sample"],
                           config["max_length"],
                           json.dumps(config["dependency"]).encode(),
                           int_to_bytes(config['num_device']),
                           ])

    status[ip]['status'] = 'IPAligning'
    print(f"Status: IPAligning {config['ids'][real_client_id]}")



    trans_model(sender, config, real_client_id, status,ip)

    print(f"Status: Prepare {config['ids'][real_client_id]}")

    return



def communication_IPAlign(sender, context_communication,config, status):
    print("start communication_IPAlign")
    group_id = config.get("group_id", "")
    client_id_ip = bidict()

    while True:
        try:

            info = None
            try:

                info = sender.recv_multipart()
            except zmq.Again:
                time.sleep(0.1)
                continue

            if info is None or len(info) < 2:
                continue
            client_id = info[0]
            msg = info[1]
            data = info[2].decode(errors='ignore') if len(info) > 2 and isinstance(info[2], bytes) else str(
                info[2] if len(info) > 2 else '')  # 安全decode
            print(
                f"Recv raw: client_id={client_id.hex()}, msg={msg}, data={info[2].decode(errors='ignore') if len(info) > 2 else 'None'}")
            if data:
                client_id_ip[client_id] = data




            # 准备建组了
            if  msg == b'WaitIPAligning' :
                # 开始对齐IP

                # 该设备发送  IPAligning 消息

                print(f"向设备 {client_id_ip[client_id]} 发送 IPAligning")
                # sender.send_multipart([client_id, b"IPAligning"])



                communication_information_prepare_ip(sender, config, status, client_id, info)
            elif msg == b'IPAlignedPrepare' :
                status[client_id_ip[client_id]]["status"] = 'IPAlignedPrepare'
                list_of_devices_ip = config["list_of_devices_ip"]
                # 要找到是哪两个设备
                while True:
                    all_IPAlign = check_ipAlign(status, config, mode="IPAlignedPrepare",list_of_devices_ip=list_of_devices_ip)
                    if  all_IPAlign:
                        break

                if all_IPAlign:
                    print("所有设备都已IPAlign，开始发送Start消息")
                    # 向所有设备发送Start消息



                    print(f"向设备 { client_id_ip[client_id] } 发送Start")
                    sender.send_multipart([client_id, b"Start"])


                    status[client_id_ip[client_id]]["status"] = b'Start'
                    print(f"设备  { client_id_ip[client_id] }  状态设置为Start")
                else:
                    print(f"设备  { client_id_ip[client_id] }  IPAlignedPrepare，等待其他设备...")

            elif msg == b'Running':
                print(f"Status: Running {config['ids'][client_id]}")
                status[client_id_ip[client_id]]["status"] = b'Running'


            elif msg == b'Finish':
                status[real_client_id] = b'Close'
                with conditions[2]:
                    while not check_status(status, config, b"Close"):
                        conditions[2].wait(timeout=60)
                    conditions[2].notify_all()

                sender.send_multipart([group_prefix + real_client_id, b"Close"])
                print(f"Close {config['ids'][real_client_id]}")
                break

            elif msg == b'Recovery':
                status[real_client_id] = b'Recovery'
                print(f"Status: Recovery {config['ids'][real_client_id]}")

                sender.send_multipart([group_prefix + real_client_id,
                                       config["graph"],
                                       config["session_index"],
                                       json.dumps(config["dependency"]).encode(),
                                       ])

            elif msg == b'WaitingStart':
                status[real_client_id] = b'WaitingStart'
                if "status" not in status:
                    status["status"] = b'WaitingStart'

                print(f"Status: WaitingStart {config['ids'][real_client_id]}")
                inner_timeout = 60
                start_inner = time.time()
                while time.time() - start_inner < inner_timeout:
                    time.sleep(1)
                    if status.get("status", b'') == b'WaitingStart':
                        sender.send_multipart([group_prefix + real_client_id, b'ResumeStart'])
                        status[real_client_id] = b'ResumeStart'
                        break
                else:
                    print("WaitingStart timeout")
            else:
                print(f"忽略非本组: {client_id.hex()}")
                continue


        except zmq.Again:
            time.sleep(0.1)
            continue

        except Exception as e:
            print(f"通信异常: {e}")
            import traceback
            traceback.print_exc()
            break

