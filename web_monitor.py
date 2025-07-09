from flask import Flask, render_template, jsonify
import threading
import json
from datetime import datetime
import time
import logging

# Disable Flask's standard log output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# Global variable to store device status
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

# Keep the most recent 100 history records
MAX_HISTORY = 100

def add_history_event(message):
    """Add a history event"""
    global device_status
    event = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "message": message
    }
    device_status["history"].insert(0, event)
    if len(device_status["history"]) > MAX_HISTORY:
        device_status["history"].pop()

def update_device_status(device_pool_manager):
    """Update device status"""
    global device_status
    
    # Get a snapshot of the current device status
    current_working = len(device_status["working_devices"])
    current_active = len(device_status["active_devices"])
    current_failed_working = len(device_status["failed_working_devices"])
    current_failed_active = len(device_status["failed_active_devices"])
    
    # Update device lists
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
    
    # Add status change history records
    new_working = len(device_status["working_devices"])
    new_active = len(device_status["active_devices"])
    new_failed_working = len(device_status["failed_working_devices"])
    new_failed_active = len(device_status["failed_active_devices"])
    
    if new_working != current_working:
        add_history_event(f"Working device count changed: {current_working} -> {new_working}")
    
    if new_active != current_active:
        add_history_event(f"Active device count changed: {current_active} -> {new_active}")
    
    if new_failed_working != current_failed_working:
        add_history_event(f"Failed working device count changed: {current_failed_working} -> {new_failed_working}")
    
    if new_failed_active != current_failed_active:
        add_history_event(f"Failed active device count changed: {current_failed_active} -> {new_failed_active}")
    
    # Update statistics
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
    
    # Update uptime
    uptime_seconds = int(time.time() - device_status["start_time"])
    hours = uptime_seconds // 3600
    minutes = (uptime_seconds % 3600) // 60
    seconds = uptime_seconds % 60
    device_status["uptime"] = f"{hours}:{minutes:02d}:{seconds:02d}"
    
    # Update timestamp
    device_status["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@app.route('/')
def index():
    """Render homepage"""
    return render_template('index.html', status=device_status)

@app.route('/api/status')
def api_status():
    """API endpoint, returns device status in JSON format"""
    return jsonify(device_status)

def start_web_server(device_pool_manager):
    """Start web server"""
    def update_status():
        while True:
            update_device_status(device_pool_manager)
            threading.Event().wait(1)  # Update every second
    
    # Start status update thread
    update_thread = threading.Thread(target=update_status, daemon=True)
    update_thread.start()
    
    # Start Flask server (disable log output)
    app.run(host='127.0.0.1', port=32333, debug=True, use_reloader=False)