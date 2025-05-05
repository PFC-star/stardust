''''''
import json
import time
from threading import Thread
import zmq

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

def communication_open_close(sender, config, status, conditions, lock, open=True):
    ## Status: Ready, Open, Prepare, Initialized, Start, Running, Finish
    while True:
        print('enter communication open close')
        with lock[0]:
            print('开始接收')
            info = sender.recv_multipart()

        client_id = info[0]
        msg = info[1]
        print(client_id + msg)
        print("收到信号")
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

            with conditions[0]:
                while not check_status(status, config, b"Open"):
                    conditions[0].wait()
                conditions[0].notify_all()

            ## Prepare
            with lock[1]:
                communication_prepare(sender, config, client_id, status)

            print(f"Status: Prepare {config['ids'][client_id]}")

        ## Initialized
        elif msg == b'Initialized':
            status[client_id] = b'Initialized'
            print(f"Status: Initialized {config['ids'][client_id]}")

            with conditions[1]:
                while not check_status(status, config, b"Initialized"):
                    conditions[1].wait()
                conditions[1].notify_all()

            ## Start
                sender.send_multipart([client_id, b"Start"])
                status[client_id] = b'Start'

                print(f"Status: Start {config['ids'][client_id]}")
        
        ## 添加故障恢复处理 - 检查是否存在需要恢复的状态
        elif msg == b'Running' or msg == b'HEARTBEAT':
            # 检查系统是否需要恢复
            if "system_status" in config and config["system_status"] == "RECOVERY_NEEDED":
                client_ip = config["ids"][client_id].decode() if isinstance(config["ids"][client_id], bytes) else config["ids"][client_id]
                print(f"检测到系统需要恢复，向设备 {client_ip} 发送握手请求")
                
                # 发送握手请求
                try:
                    sender.send_multipart([client_id, b"HANDSHAKE_REQUEST"])
                    print(f"已向设备 {client_ip} 发送握手请求，等待响应")
                    
                    # 等待握手响应 - 设置超时
                    original_timeout = sender.getsockopt(zmq.RCVTIMEO)
                    sender.setsockopt(zmq.RCVTIMEO, 10000)  # 10秒超时
                    
                    try:
                        response = sender.recv_multipart()
                        if len(response) >= 2 and response[0] == client_id:
                            response_msg = response[1].decode('utf-8', errors='ignore')
                            print(f"收到设备 {client_ip} 握手响应: {response_msg}")
                            
                            if response_msg == "HANDSHAKE_READY":
                                print(f"设备 {client_ip} 握手成功，开始发送故障恢复信号")
                                
                                # 根据故障恢复状态发送不同消息
                                if config["recovery_status"] == "HAS_REPLACEMENT":
                                    # 有替代设备，发送完整的故障恢复信息
                                    print(f"向设备 {client_ip} 发送故障恢复信号 (有替代设备)")
                                    
                                    # 1. 发送故障恢复信号
                                    sender.send_multipart([client_id, b"FAILURE_RECOVERY"])
                                    print(f"故障恢复信号发送完成")
                                    
                                    # 等待客户端处理
                                    import time
                                    time.sleep(3)
                                    
                                    # 2. 发送新IP图
                                    sender.send_multipart([client_id, config["graph"]])
                                    print(f"新IP图发送完成")
                                    
                                    # 等待客户端处理
                                    time.sleep(3)
                                    
                                    # 3. 发送会话索引
                                    sender.send_multipart([client_id, config["session_index"]])
                                    print(f"会话索引发送完成")
                                    
                                    # 设置设备状态为恢复中
                                    status[client_id] = b'Recovering'
                                    print(f"设备 {client_ip} 进入恢复状态")
                                    
                                elif config["recovery_status"] == "NO_REPLACEMENT":
                                    # 无替代设备，发送系统故障通知
                                    print(f"向设备 {client_ip} 发送系统故障通知 (无替代设备)")
                                    sender.send_multipart([client_id, b"SYSTEM_FAILURE_NO_REPLACEMENT", 
                                                         json.dumps(config.get("failed_ips", [])).encode('utf-8')])
                                    
                                    # 设置设备状态为暂停
                                    status[client_id] = b'Suspended'
                                    print(f"设备 {client_ip} 进入暂停状态")
                            else:
                                print(f"设备 {client_ip} 握手响应不符合预期: {response_msg}")
                        else:
                            print(f"接收到格式异常的握手响应")
                            
                    except zmq.error.Again:
                        print(f"等待设备 {client_ip} 握手响应超时")
                    except Exception as e:
                        print(f"处理设备 {client_ip} 握手响应时出错: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        # 恢复原来的超时设置
                        sender.setsockopt(zmq.RCVTIMEO, original_timeout)
                    
                except Exception as e:
                    print(f"向设备 {client_ip} 发送握手请求时出错: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 保持正常的Running状态处理
            status[client_id] = b'Running'
            # 如果客户端没有明确告知running状态，这里可以发送一个正常响应
            if msg == b'HEARTBEAT':
                sender.send_multipart([client_id, b"HEARTBEAT_RECEIVED", b"SYSTEM_NORMAL"])
                print(f"设备心跳响应：系统正常")

        ## 添加故障恢复处理逻辑
        elif msg == b"FAILURE_RECOVERY_ACK":
            print(f"设备 {config['ids'][client_id]} 确认收到故障恢复信号")
            status[client_id] = b'Recovery'
            
            # 记录已恢复的设备数量
            recovery_count = sum(1 for s in status.values() if s == b'Recovery')
            expected_count = len(config["ids"]) - sum(1 for ip in config["ids"].values() 
                                                     if ip.decode() in config.get("failed_ips", []))
            
            print(f"故障恢复进度: {recovery_count}/{expected_count} 设备已恢复")
            
            # 当所有预期的设备都已恢复时，通知继续运行
            if recovery_count >= expected_count:
                print("所有设备已恢复，继续推理")
                for cid in config["ids"]:
                    if status.get(cid) == b'Recovery':
                        sender.send_multipart([cid, b"RESUME_INFERENCE"])
                        status[cid] = b'Running'
                        
                # 清除故障标记
                if "system_status" in config:
                    del config["system_status"]
                if "recovery_status" in config:
                    del config["recovery_status"]
                if "failed_ips" in config:
                    del config["failed_ips"]
                
                print("系统已成功从故障中恢复")
        
        ## 处理无替代设备情况下的故障确认
        elif msg == b"SYSTEM_FAILURE_NO_REPLACEMENT_ACK":
            print(f"设备 {config['ids'][client_id]} 确认收到系统故障通知(无替代设备)")
            status[client_id] = b'Suspended'
            
            # 记录已暂停的设备数量
            suspended_count = sum(1 for s in status.values() if s == b'Suspended')
            expected_count = len(config["ids"]) - sum(1 for ip in config["ids"].values()
                                                     if ip.decode() in config.get("failed_ips", []))
            
            print(f"系统暂停进度: {suspended_count}/{expected_count} 设备已暂停")
            
            # 当所有预期的设备都已暂停时，等待人工干预
            if suspended_count >= expected_count:
                print("所有设备已暂停，等待人工干预")
                
                # 这里可以添加人工干预的通知机制，如发送邮件或短信
                print("需要人工干预：系统因设备故障且无可用替代设备而暂停")
                print(f"故障设备列表: {config.get('failed_ips', [])}")
                
                # 保持系统在suspended状态
                for cid in config["ids"]:
                    if status.get(cid) == b'Suspended':
                        # 可以添加定期发送保持暂停状态的消息
                        pass

        elif msg == b"Running":
            # Todo simulate load balance
            # time.sleep(10)
            # print(f"{config['ids'][client_id]} Start Load Balance")
            # # config["session_index"] = ";".join(["0,1", "2,3,4,5,6", "7,8,9"]).encode('utf-8')
            # sender.send_multipart([client_id, b"re-balance",
            #                                   config["session_index"],
            #                                   json.dumps(config["dependency"]).encode()])
            
            # if (config["ids"][client_id] == config["head_node"].encode()):
            #     client_id, msg = sender.recv_multipart()
            #     config["reload_sampleId"] = msg.decode()
            #     print(f"The Reload Sample starts from {config['reload_sampleId']}")
            #     assert config["reload_sampleId"].isdigit(), f"reload sampleId is not an integer string"
            # else:
            #     while (config["reload_sampleId"] == None):
            #         print("Wait the resample ID")
            #         time.sleep(0.1)
            
            #     print(f"Send Reload Sample id : {config['reload_sampleId']} to {config['ids'][client_id]}")
            #     sender.send_multipart([client_id, "id".encode(), config["reload_sampleId"].encode()])
            pass
        elif msg == b'Finish':
            status[client_id] = b'Close'
            with conditions[2]:
                while not check_status(status, config, b"Close"):
                    conditions[2].wait()
                conditions[2].notify_all()

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
        elif msg == b'RecoveryInference':
            status[client_id] = b'RecoveryInference'
            print(f"Status: RecoveryInference {config['ids'][client_id]}")
        elif msg == b'RecoveryReady':
            status[client_id] = b'RecoveryReady'
            print(f"Status: RecoveryReady {config['ids'][client_id]}")
            # 判断active设备是否也RecoveryReady了
            
 
                     
            

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

def communication_prepare(sender, config, client_id, status):
    sender.send_multipart([client_id, b'Prepare'])
    node_ip = config["ids"][client_id]
    sender.send_multipart([client_id, str(config["skip_model_transmission"]).encode()])

    if not config["skip_model_transmission"]:   ## Assume data is received on the machine

        print(f"send {config['file_path'][node_ip]} to {node_ip}")

        # onnx sends multiple files, which should be a zip
        if config["onnx"]:
            send_model_file(config["file_path"][node_ip], sender, client_id)
        else:
            send_model_file(config["file_path"][node_ip], sender, client_id)

        # # transmit tokenizer to header
        # if config["head_node"] == node_ip:
        #     print(f"send {config['file_path'][b'tokenizer']} to {node_ip}")
        #     send_model_file(config["file_path"][b"tokenizer"], sender, client_id)

    status[client_id] = b"Prepare"

def communication_data_transmission(sender, num_devices, head_client_id, status):
    while check_status(status, num_devices, b"Start"):
        pass


def communication_result_transmission(sender, result, num_devices, tail_client_id, status):
    # while check_status(status, num_devices, b"Finish"):
    sender.send_multipart([b"res", result])
    pass


all_status = {b"Ready":  0,
              b"Open":   1,
              b"Prepare":2,
              b"Initialized": 3,
              b"Start":  4,
              b"Running":5,
              b"Finish": 6,
              b"Close": 7}

def check_status(status, config, mode):
    if len(status) != config["num_device"]:
        return False
    for v in status.values():
        if all_status[v] < all_status[mode]:
            return False
    return True


def ConfigCreator(Config, client_id):
    ## Based on the monitor situation
    return Config["graph"]