import zmq
import os
import threading
import json

LOG_DIR = "device_logs12"
os.makedirs(LOG_DIR, exist_ok=True)
log_locks = {}

def get_log_lock(device_ip):
    if device_ip not in log_locks:
        log_locks[device_ip] = threading.Lock()
    return log_locks[device_ip]

def main():
    context = zmq.Context()
    # DEALER/ROUTER模式都可以，推荐ROUTER更通用
    socket = context.socket(zmq.ROUTER)
    socket.bind("tcp://*:9889")
    print("ZeroMQ log server started on port 9889...")

    while True:
        # ROUTER模式需先接收identity
        identity = socket.recv()
        msg = socket.recv()
        try:
            data = json.loads(msg.decode("utf-8"))
        except Exception as e:
            print("Failed to parse JSON:", e)
            continue

        device_ip = data.get("deviceIP", "unknown")
        print(f"\n=== Log from device {device_ip} ===")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("=== End of log ===\n")

        # 存储到以IP命名的文件
        log_file = os.path.join(LOG_DIR, f"{device_ip}.log")
        lock = get_log_lock(device_ip)
        with lock:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()