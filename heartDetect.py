import zmq
import json
import time

def main():
    # 创建 ZeroMQ 上下文
    context = zmq.Context()

    # 创建 DEALER 套接字
    socket = context.socket(zmq.DEALER)

    # 绑定服务器端口
    socket.bind("tcp://0.0.0.0:34567")
    print("Server is listening on tcp://0.0.0.0:34567...")

    # 初始化设备列表，存储设备的 IP 和角色
    device_list = {}

    # 设备的心跳时间间隔，单位为秒
    heartbeat_timeout = 10  # 例如，心跳包超时 10 秒

    while True:
        try:
            # 接收来自客户端的消息
            message = socket.recv_string()
            print(f"Received message: {message}")

            # 假设收到的是 JSON 格式的心跳包
            try:
                # 将消息解析为 JSON 对象
                data = json.loads(message)
                print(f"Received JSON data: {data}")

                # 判断心跳包内容
                if "ip" in data and "role" in data:
                    ip = data["ip"]
                    role = data["role"]
                    print(f"Heartbeat received from IP: {ip} with role: {role}")

                    # 判断设备是否已经存在
                    if ip not in device_list:
                        # 如果设备不在列表中，表示是新设备加入
                        device_list[ip] = {"role": role, "last_heartbeat": time.time()}
                        print(f"Device with IP {ip} has joined the network with role: {role}")
                    else:
                        # 如果设备已经在列表中，更新其最后的心跳时间
                        device_list[ip]["last_heartbeat"] = time.time()

                else:
                    print("Received invalid heartbeat message")

            except json.JSONDecodeError:
                print("Error decoding JSON message")

            # 移除超时的设备
            current_time = time.time()
            for ip in list(device_list.keys()):
                if current_time - device_list[ip]["last_heartbeat"] > heartbeat_timeout:
                    # 如果设备超时未发送心跳包，表示设备已经退出
                    role = device_list[ip]["role"]
                    print(f"Device with IP {ip} and role {role} has exited (no heartbeat received in {heartbeat_timeout} seconds).")
                    del device_list[ip]

        except KeyboardInterrupt:
            break

    # 关闭 socket 和上下文
    socket.close()
    context.term()

if __name__ == "__main__":
    main()