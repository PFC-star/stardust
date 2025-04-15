#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
设备动态注册客户端
用于演示如何将设备注册到服务器的设备池中并保持心跳
"""

import zmq
import json
import time
import argparse
import socket
import sys
import threading
import platform
import uuid

# 全局变量
VERBOSE = False  # 详细日志模式

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
        
        if VERBOSE:
            print(f"发送心跳数据: {heartbeat_data}")
            
        # 发送ZMQ格式的心跳消息
        socket.send_multipart([
            b"HEARTBEAT",
            json.dumps(heartbeat_data).encode('utf-8')
        ])
        
        if VERBOSE:
            print(f"心跳数据已发送，等待响应...")
            
        # 等待响应，设置超时
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时
        try:
            response = socket.recv_multipart()
            if VERBOSE:
                print(f"收到心跳响应: {response}")
            print(f"心跳响应: {response[0].decode()}")
            return True
        except zmq.error.Again:
            if VERBOSE:
                print("等待心跳响应超时(5秒)")
            print("心跳响应超时")
            return False
        
    except Exception as e:
        if VERBOSE:
            print(f"心跳发送过程中出错: {e}")
            import traceback
            traceback.print_exc()
        print(f"发送心跳时出错: {e}")
        return False
    finally:
        if 'socket' in locals():
            socket.close()
        if 'context' in locals():
            context.term()

def start_heartbeat_thread(server_ip, port, device_id, interval=30, stop_event=None):
    """启动心跳线程"""
    def heartbeat_loop():
        consecutive_failures = 0
        max_failures = 3  # 连续失败3次后报告设备可能故障
        
        while True:
            if stop_event and stop_event.is_set():
                print("心跳线程收到停止信号")
                break
                
            success = send_heartbeat(server_ip, port, device_id)
            
            if not success:
                consecutive_failures += 1
                print(f"心跳失败次数: {consecutive_failures}/{max_failures}")
                
                if consecutive_failures >= max_failures:
                    print("警告: 连续多次心跳失败，设备可能已断开连接")
                    
                # 如果心跳失败，减少等待时间以便更快重试
                wait_time = min(interval, 2)
            else:
                consecutive_failures = 0
                wait_time = interval
                
            time.sleep(wait_time)
    
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
        device_role (str): 设备角色 (header/worker/tail)
        model_request (str, 可选): 客户端请求的模型
        device_id (str, 可选): 设备唯一标识符
        virtual_ip (str, 可选): 虚拟IP地址
    
    返回:
        tuple: (成功状态, 设备ID, 服务器分配的IP)
    """
    if device_id is None:
        device_id = f"{platform.node()}-{uuid.uuid4().hex[:8]}"
    
    try:
        if VERBOSE:
            print(f"创建ZMQ连接到 {server_ip}:{port}")
            
        # 创建ZMQ上下文和套接字
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.identity = device_id.encode('utf-8')
        socket.connect(f"tcp://{server_ip}:{port}")
        
        if VERBOSE:
            print(f"连接成功，套接字身份: {device_id}")
            
        # 获取设备信息
        device_info = {
            'ip': virtual_ip or get_local_ip(),
            'role': device_role,
            'device_id': device_id,
            'device_type': platform.machine(),
            'os': platform.system(),
            'timestamp': time.time()
        }
        
        if VERBOSE:
            print(f"准备注册设备，信息: {device_info}")
        
        # 添加模型请求信息(如果是客户端)
        if device_role == 'client' and model_request:
            device_info['model'] = model_request
        
        if VERBOSE:
            print("打包注册消息...")
            
        # 发送ZMQ格式的注册消息
        message = [
            b"RegisterIP",
            json.dumps(device_info).encode('utf-8')
        ]
        
        if VERBOSE:
            print(f"发送注册消息: {message}")
            
        socket.send_multipart(message)
        
        if VERBOSE:
            print("注册消息已发送，等待响应...")
        
        # 等待响应，设置超时
        socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10秒超时
        try:
            if VERBOSE:
                print("准备接收响应...")
                
            response = socket.recv_multipart()
            
            if VERBOSE:
                print(f"收到服务器响应: {response}")
                
            action = response[0].decode()
            msg = response[1].decode() if len(response) > 1 else ""
            
            if action == "REGISTRATION_SUCCESSFUL":
                print(f"设备 {device_id} 注册成功")
                print(f"服务器响应: {msg}")
                
                assigned_ip = device_info['ip']  # 使用发送的IP作为分配的IP
                
                return True, device_id, assigned_ip
            else:
                print(f"注册失败: {action} - {msg}")
                return False, None, None
        
        except zmq.error.Again:
            if VERBOSE:
                print("等待注册响应超时(10秒)")
                print("检查服务器是否运行且端口是否正确")
            print("注册响应超时")
            return False, None, None
            
    except Exception as e:
        if VERBOSE:
            print(f"注册过程中发生异常: {e}")
            import traceback
            traceback.print_exc()
        print(f"注册过程中出错: {e}")
        return False, None, None
    finally:
        if VERBOSE:
            print("关闭套接字和上下文")
        if 'socket' in locals():
            socket.close()
        if 'context' in locals():
            context.term()

def main():
    parser = argparse.ArgumentParser(description='设备动态注册客户端')
    parser.add_argument('--server', default='localhost', help='服务器地址')
    parser.add_argument('--port', type=int, default=23456, help='服务器通信端口')
    parser.add_argument('--role', choices=['header', 'worker', 'tail'], default='worker', 
                        help='设备角色 (header: 头节点, worker: 工作节点, tail: 尾节点)')
    parser.add_argument('--model', default='bloom560m-int8', help='请求的模型名称 (例如: bloom560m, bloom560m-int8)')
    parser.add_argument('--device-id', help='指定唯一设备ID（用于测试多设备场景）')
    parser.add_argument('--virtual-ip', help='指定虚拟IP地址（用于测试多设备场景）')
    parser.add_argument('--heartbeat-interval', type=int, default=5, help='心跳发送间隔（秒）')
    parser.add_argument('-v', '--verbose', action='store_true', help='启用详细日志')
    
    args = parser.parse_args()
    
    # 设置详细模式
    global VERBOSE
    VERBOSE = args.verbose
    
    # 创建停止事件，用于安全终止心跳线程
    heartbeat_stop_event = threading.Event()
    
    try:
        if VERBOSE:
            print(f"开始注册设备... 服务器: {args.server}:{args.port}")
            
        success, device_id, assigned_ip = register_device(
            args.server, args.port, args.role, args.model, args.device_id, args.virtual_ip
        )
        
        if success:
            print(f"\n设备已成功注册为 '{args.role}'")
            if args.model:
                print(f"已请求模型: {args.model}")
            
            print("\n开始持续发送心跳...")
            print("按 Ctrl+C 终止程序")
            
            # 启动心跳线程
            heartbeat_thread = start_heartbeat_thread(
                args.server, args.port, device_id, args.heartbeat_interval, heartbeat_stop_event
            )
            
            # 主线程保持运行，直到用户中断
            while True:
                time.sleep(1)
                
        else:
            print("设备注册失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n用户中断，正在关闭连接...")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        # 停止心跳线程
        if 'heartbeat_stop_event' in locals():
            heartbeat_stop_event.set()
            print("已发送心跳线程停止信号")
        
        if 'assigned_ip' in locals() and assigned_ip:
            print(f"服务器分配的IP: {assigned_ip}")
        print("客户端已关闭")

if __name__ == "__main__":
    main() 