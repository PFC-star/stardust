from flask import Flask, render_template, jsonify
import threading
import json
from datetime import datetime
import time
import logging

# 禁用Flask的标准日志输出
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# 全局变量存储设备状态
device_status = {
    "working_devices": [],
    "active_devices": [],
    "failed_working_devices": [],
    "failed_active_devices": [],
    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "uptime": "0:00:00",
    "total_devices": 0,
    "online_devices": 0,
    "failed_devices": 0,
    "history": [],
    "start_time": time.time()
}

# 保持最近100条历史记录
MAX_HISTORY = 100

def add_history_event(message):
    """添加历史事件"""
    global device_status
    event = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "message": message
    }
    device_status["history"].insert(0, event)
    if len(device_status["history"]) > MAX_HISTORY:
        device_status["history"].pop()

def update_device_status(device_pool_manager):
    """更新设备状态"""
    global device_status
    
    # 获取当前设备状态的快照
    current_working = len(device_status["working_devices"])
    current_active = len(device_status["active_devices"])
    current_failed_working = len(device_status["failed_working_devices"])
    current_failed_active = len(device_status["failed_active_devices"])
    
    # 更新设备列表
    device_status["working_devices"] = [
        {
            "ip": device.get("ip"),
            "id": device.get("device_id"),
            "role": device.get("role", "unknown"),
            "last_heartbeat": datetime.fromtimestamp(
                device_pool_manager.device_heartbeats.get(device.get("device_id"), 0)
            ).strftime("%H:%M:%S")
        }
        for device in device_pool_manager.working_devices
    ]
    
    device_status["active_devices"] = [
        {
            "ip": device.get("ip"),
            "id": device.get("device_id"),
            "role": device.get("role", "unknown"),
            "last_heartbeat": datetime.fromtimestamp(
                device_pool_manager.device_heartbeats.get(device.get("device_id"), 0)
            ).strftime("%H:%M:%S")
        }
        for device in device_pool_manager.active_devices
    ]
    
    device_status["failed_working_devices"] = [
        {
            "ip": device.get("ip"),
            "id": device.get("device_id"),
            "role": device.get("role", "unknown"),
            "failure_time": datetime.fromtimestamp(
                device.get("failure_time", time.time())
            ).strftime("%H:%M:%S"),
            "failure_reason": device.get("failure_reason", "Unknown")
        }
        for device in device_pool_manager.failed_working_devices
    ]
    
    device_status["failed_active_devices"] = [
        {
            "ip": device.get("ip"),
            "id": device.get("device_id"),
            "role": device.get("role", "unknown"),
            "failure_time": datetime.fromtimestamp(
                device.get("failure_time", time.time())
            ).strftime("%H:%M:%S"),
            "failure_reason": device.get("failure_reason", "Unknown")
        }
        for device in device_pool_manager.failed_active_devices
    ]
    
    # 添加状态变化历史记录
    new_working = len(device_status["working_devices"])
    new_active = len(device_status["active_devices"])
    new_failed_working = len(device_status["failed_working_devices"])
    new_failed_active = len(device_status["failed_active_devices"])
    
    if new_working != current_working:
        add_history_event(f"工作设备数量变化: {current_working} -> {new_working}")
    
    if new_active != current_active:
        add_history_event(f"活跃设备数量变化: {current_active} -> {new_active}")
    
    if new_failed_working != current_failed_working:
        add_history_event(f"工作设备故障数量变化: {current_failed_working} -> {new_failed_working}")
    
    if new_failed_active != current_failed_active:
        add_history_event(f"活跃设备故障数量变化: {current_failed_active} -> {new_failed_active}")
    
    # 更新统计信息
    device_status["total_devices"] = (
        len(device_status["working_devices"]) +
        len(device_status["active_devices"]) +
        len(device_status["failed_working_devices"]) +
        len(device_status["failed_active_devices"])
    )
    
    device_status["online_devices"] = (
        len(device_status["working_devices"]) +
        len(device_status["active_devices"])
    )
    
    device_status["failed_devices"] = (
        len(device_status["failed_working_devices"]) +
        len(device_status["failed_active_devices"])
    )
    
    # 更新运行时间
    uptime_seconds = int(time.time() - device_status["start_time"])
    hours = uptime_seconds // 3600
    minutes = (uptime_seconds % 3600) // 60
    seconds = uptime_seconds % 60
    device_status["uptime"] = f"{hours}:{minutes:02d}:{seconds:02d}"
    
    # 更新时间戳
    device_status["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html', status=device_status)

@app.route('/api/status')
def api_status():
    """API端点，返回JSON格式的设备状态"""
    return jsonify(device_status)

def start_web_server(device_pool_manager):
    """启动Web服务器"""
    def update_status():
        while True:
            update_device_status(device_pool_manager)
            threading.Event().wait(1)  # 每秒更新一次
    
    # 启动状态更新线程
    update_thread = threading.Thread(target=update_status, daemon=True)
    update_thread.start()
    
    # 启动Flask服务器（关闭日志输出）
    app.run(host='127.0.0.1', port=32333, debug=True, use_reloader=False)