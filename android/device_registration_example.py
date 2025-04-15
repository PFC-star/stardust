#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
设备动态注册示例客户端
用于演示如何在任何时刻将设备注册到服务器的设备池中
"""

import zmq
import json
import time
import argparse
import socket
import sys
import threading
from datetime import datetime
import platform
import uuid

def get_local_ip():
    """获取本地IP地址"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 不需要真正连接
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def send_heartbeat(server_ip, port, device_id):
    """向服务器发送心跳"""
    try:
        # 创建ZMQ上下文和套接字
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.identity = device_id.encode('utf-8')
        socket.connect(f"tcp://{server_ip}:{port}")
        
        # 准备心跳数据
        heartbeat_data = {
            'device_id': device_id,
            'timestamp': time.time(),
            'status': 'active'
        }
        
        # 发送ZMQ格式的心跳消息
        socket.send_multipart([
            b"HEARTBEAT",
            json.dumps(heartbeat_data).encode('utf-8')
        ])
        
        # 等待响应，设置超时
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时
        try:
            response = socket.recv_multipart()
            print(f"心跳响应: {response[0].decode()}")
            return True
        except zmq.error.Again:
            print("心跳响应超时")
            return False
        
    except Exception as e:
        print(f"发送心跳时出错: {e}")
        return False
    finally:
        if 'socket' in locals():
            socket.close()
        if 'context' in locals():
            context.term()

def start_heartbeat_thread(server_ip, port, device_id, interval=30):
    """启动心跳线程"""
    def heartbeat_loop():
        while True:
            send_heartbeat(server_ip, port, device_id)
            time.sleep(interval)
    
    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()
    print(f"已启动心跳线程，间隔 {interval} 秒")
    return heartbeat_thread

def register_device(server_ip, port, device_role, model_request=None, device_id=None, virtual_ip=None):
    """
    向服务器注册设备
    
    参数:
        server_ip (str): 服务器IP地址
        port (int): 服务器端口
        device_role (str): 设备角色 (worker/client)
        model_request (str, 可选): 客户端请求的模型
        device_id (str, 可选): 设备唯一标识符
        virtual_ip (str, 可选): 虚拟IP地址
    
    返回:
        tuple: (成功状态, 设备ID, 服务器分配的IP)
    """
    if device_id is None:
        device_id = f"{platform.node()}-{uuid.uuid4().hex[:8]}"
    
    try:
        # 创建ZMQ上下文和套接字
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.identity = device_id.encode('utf-8')
        socket.connect(f"tcp://{server_ip}:{port}")
        
        # 获取设备信息
        device_info = {
            'ip': virtual_ip or get_local_ip(),
            'role': device_role,
            'device_id': device_id,
            'device_type': platform.machine(),
            'os': platform.system(),
            'timestamp': time.time()
        }
        
        # 添加模型请求信息(如果是客户端)
        if device_role == 'client' and model_request:
            device_info['model'] = model_request
        
        # 发送ZMQ格式的注册消息
        socket.send_multipart([
            b"RegisterIP",
            json.dumps(device_info).encode('utf-8')
        ])
        
        # 等待响应，设置超时
        socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10秒超时
        try:
            response = socket.recv_multipart()
            action = response[0].decode()
            msg = response[1].decode() if len(response) > 1 else ""
            
            if action == "REGISTRATION_SUCCESSFUL":
                print(f"设备 {device_id} 注册成功")
                print(f"服务器响应: {msg}")
                
                assigned_ip = device_info['ip']  # 使用发送的IP作为分配的IP
                
                # 启动心跳线程
                start_heartbeat_thread(server_ip, port, device_id, 5)  # 5秒心跳间隔
                
                return True, device_id, assigned_ip
            else:
                print(f"注册失败: {action} - {msg}")
                return False, None, None
        
        except zmq.error.Again:
            print("注册响应超时")
            return False, None, None
            
    except Exception as e:
        print(f"注册过程中出错: {e}")
        return False, None, None
    finally:
        if 'socket' in locals():
            socket.close()
        if 'context' in locals():
            context.term()

def query_device_pool_status(server_ip, port):
    """
    查询服务器上的设备池状态
    
    参数:
        server_ip (str): 服务器IP地址
        port (int): 服务器端口
        
    返回:
        dict: 包含设备池状态信息的字典，如果查询失败则返回None
    """
    try:
        # 创建ZMQ上下文和套接字
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.identity = f"status_query_{int(time.time())}".encode('utf-8')
        socket.connect(f"tcp://{server_ip}:{port}")
        
        # 发送ZMQ格式的状态查询消息
        socket.send_multipart([
            b"GET_STATUS",
            json.dumps({"query": "device_pool_status"}).encode('utf-8')
        ])
        
        # 等待响应，设置超时
        socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10秒超时
        try:
            response = socket.recv_multipart()
            
            if len(response) < 2:
                print(f"警告: 响应不完整，只有 {len(response)} 个部分")
                return None
            
            action = response[0].decode()
            msg_content = response[1].decode()
            
            if action == "STATUS_INFO":
                try:
                    status_info = json.loads(msg_content)
                    # 显示状态信息
                    display_device_pool_status(status_info)
                    return status_info
                except json.JSONDecodeError as e:
                    print(f"无法解析状态响应JSON: {e}")
                    return None
            else:
                print(f"查询设备池状态失败: 收到 {action} 响应")
                return None
            
        except zmq.error.Again:
            print("状态查询响应超时")
            return None
            
    except Exception as e:
        print(f"查询设备池状态时出错: {e}")
        return None
    finally:
        if 'socket' in locals():
            socket.close()
        if 'context' in locals():
            context.term()

def send_inference_request(socket, device_id, input_text):
    """发送推理请求到服务器"""
    try:
        inference_data = {
            "device_id": device_id,
            "input_text": input_text
        }
        socket.send_multipart([b"INFERENCE", json.dumps(inference_data).encode('utf-8')])
        
        # 添加更健壮的响应处理
        response = socket.recv_multipart()
        if not response:
            print("警告: 服务器返回空响应")
            return {"status": "error", "message": "服务器返回空响应"}
            
        if len(response) < 2:
            print(f"警告: 服务器返回不完整响应，只有 {len(response)} 个部分")
            # 尝试解析可能的错误消息
            if len(response) > 0:
                print(f"收到响应: {response[0].decode() if isinstance(response[0], bytes) else response[0]}")
                return {"status": "incomplete", "message": response[0].decode() if isinstance(response[0], bytes) else str(response[0])}
            return {"status": "error", "message": "服务器返回不完整响应"}
            
        print(f"收到推理响应: {response[0].decode()}")
        
        # 尝试解析JSON响应
        try:
            result = json.loads(response[1].decode())
            return result
        except json.JSONDecodeError as je:
            print(f"JSON解析错误: {je}")
            return {"status": "error", "message": f"无法解析服务器响应: {response[1].decode()}"}
    except Exception as e:
        print(f"发送推理请求失败: {e}")
        return {"status": "error", "message": str(e)}

def display_device_pool_status(status_info):
    """
    显示设备池状态信息
    
    参数:
        status_info (dict): 包含设备池状态的字典
    """
    if not status_info:
        print("没有可用的设备池状态信息")
        return
        
    # 打印状态信息
    print("\n设备池状态:")
    print(f"  总设备数: {status_info.get('total_devices', 0)}")
    
    # 检查并打印工作设备数量
    working_devices = status_info.get('working_devices', 0)
    print(f"  工作设备数: {working_devices}")
    
    # 检查并打印活跃设备数量
    active_devices = status_info.get('active_devices', 0)
    print(f"  活跃设备数: {active_devices}")
    
    # 检查并打印工作设备故障数量
    failed_working_devices = len(status_info.get('failed_working_devices', []))
    print(f"  工作设备故障数: {failed_working_devices}")
    
    # 检查并打印活跃设备故障数量
    failed_active_devices = len(status_info.get('failed_active_devices', []))
    print(f"  活跃设备故障数: {failed_active_devices}")
    
    # 检查并打印活跃任务数量
    active_tasks = status_info.get('active_tasks', 0)
    print(f"  活跃任务数: {active_tasks}")
    
    # 检查并打印初始化状态
    initialization_complete = status_info.get('initialization_complete', False)
    print(f"  初始化完成: {'是' if initialization_complete else '否'}")
    
    # 打印工作设备详情
    working_devices_list = status_info.get('working_devices_list', [])
    if working_devices_list:
        print("\n工作设备列表:")
        for i, device in enumerate(working_devices_list):
            device_id = device.get('id', 'N/A')
            ip = device.get('ip', 'N/A')
            role = device.get('role', 'N/A')
            last_heartbeat = device.get('last_heartbeat', 0)
            last_heartbeat_time = datetime.fromtimestamp(last_heartbeat).strftime('%Y-%m-%d %H:%M:%S') if last_heartbeat > 0 else 'N/A'
            print(f"  {i+1}. ID: {device_id}, IP: {ip}, 角色: {role}, 最近心跳: {last_heartbeat_time}")
    
    # 打印活跃设备详情
    active_devices_list = status_info.get('active_devices_list', [])
    if active_devices_list:
        print("\n活跃设备列表:")
        for i, device in enumerate(active_devices_list):
            device_id = device.get('id', 'N/A')
            ip = device.get('ip', 'N/A')
            role = device.get('role', 'N/A')
            last_heartbeat = device.get('last_heartbeat', 0)
            last_heartbeat_time = datetime.fromtimestamp(last_heartbeat).strftime('%Y-%m-%d %H:%M:%S') if last_heartbeat > 0 else 'N/A'
            print(f"  {i+1}. ID: {device_id}, IP: {ip}, 角色: {role}, 最近心跳: {last_heartbeat_time}")
    
    # 打印工作设备故障详情
    failed_working_devices_list = status_info.get('failed_working_devices', [])
    if failed_working_devices_list:
        print("\n工作设备故障列表:")
        for i, device in enumerate(failed_working_devices_list):
            device_id = device.get('id', 'N/A')
            ip = device.get('ip', 'N/A')
            role = device.get('role', 'N/A')
            failure_time = device.get('failure_time', 0)
            failure_time_str = datetime.fromtimestamp(failure_time).strftime('%Y-%m-%d %H:%M:%S') if failure_time > 0 else 'N/A'
            failure_reason = device.get('failure_reason', 'N/A')
            print(f"  {i+1}. ID: {device_id}, IP: {ip}, 角色: {role}, 故障时间: {failure_time_str}, 原因: {failure_reason}")
    
    # 打印活跃设备故障详情
    failed_active_devices_list = status_info.get('failed_active_devices', [])
    if failed_active_devices_list:
        print("\n活跃设备故障列表:")
        for i, device in enumerate(failed_active_devices_list):
            device_id = device.get('id', 'N/A')
            ip = device.get('ip', 'N/A')
            role = device.get('role', 'N/A')
            failure_time = device.get('failure_time', 0)
            failure_time_str = datetime.fromtimestamp(failure_time).strftime('%Y-%m-%d %H:%M:%S') if failure_time > 0 else 'N/A'
            failure_reason = device.get('failure_reason', 'N/A')
            print(f"  {i+1}. ID: {device_id}, IP: {ip}, 角色: {role}, 故障时间: {failure_time_str}, 原因: {failure_reason}")

def main():
    parser = argparse.ArgumentParser(description='设备动态注册客户端')
    parser.add_argument('--server', default='localhost', help='服务器地址')
    parser.add_argument('--port', type=int, default=23456, help='服务器通信端口')
    parser.add_argument('--role', choices=['header', 'worker', 'tail'], default='worker', 
                        help='设备角色 (header: 头节点, worker: 工作节点, tail: 尾节点)')
    parser.add_argument('--model',default='bloom560m-int8', help='请求的模型名称 (例如: bloom560m, bloom560m-int8)')
    parser.add_argument('--query', action='store_true', help='只查询设备池状态')
    parser.add_argument('--device-id', help='指定唯一设备ID（用于测试多设备场景）')
    parser.add_argument('--virtual-ip', help='指定虚拟IP地址（用于测试多设备场景）')
    parser.add_argument('--heartbeat-interval', type=int, default=5, help='心跳发送间隔（秒）')
    parser.add_argument('--inference', help='发送推理请求（提供输入文本）')
    
    args = parser.parse_args()
    
    # 将服务器地址和端口设为全局变量，以便心跳线程使用
    global server_address, server_port
    server_address = args.server
    server_port = args.port
    
    if args.query:
        query_device_pool_status(args.server, args.port)
    elif args.inference:
        # 如果指定了推理参数，发送推理请求而不是注册设备
        device_id = args.device_id or f"inference_client_{int(time.time())}"
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.identity = device_id.encode('utf-8')
        socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10秒超时
        socket.connect(f"tcp://{args.server}:{args.port}")
        
        try:
            result = send_inference_request(socket, device_id, args.inference)
            print("\n推理结果:")
            print(json.dumps(result, indent=2))
        finally:
            socket.close()
            context.term()
    else:
        success, device_id, assigned_ip = register_device(
            args.server, args.port, args.role, args.model, args.device_id, args.virtual_ip
        )
        
        if success:
            print(f"\n设备已成功注册为 '{args.role}'")
            if args.model:
                print(f"已请求模型: {args.model}")
            
            # 注册后，查询一下当前设备池状态
            time.sleep(1)
            status_info = query_device_pool_status(args.server, args.port)
            
            # 如果初始化已完成，检查设备状态
            if status_info and status_info.get("initialization_complete"):
                device_status = "working" if any(d.get("id") == device_id and d.get("status") == "working" for d in status_info.get("devices", [])) else "active"
                print(f"\n当前设备状态: {device_status}")
            
            try:
                print("\n开始持续发送心跳...")
                print("按 Ctrl+C 终止程序")
                
                # 启动心跳线程
                heartbeat_t = threading.Thread(
                    target=start_heartbeat_thread,
                    args=(args.server, args.port, device_id, args.heartbeat_interval),
                    daemon=True
                )
                heartbeat_t.start()
                
                # 主线程保持运行，直到用户中断
                while True:
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n用户中断，正在关闭连接...")
            finally:
                if assigned_ip:
                    print(f"服务器分配的IP: {assigned_ip}")
                print("客户端已关闭")
        else:
            print("设备注册失败")
            sys.exit(1)

if __name__ == "__main__":
    main() 