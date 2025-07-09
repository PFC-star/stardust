import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# os.environ["DS_ACCELERATOR"]='cpu'
import time
import zmq
from SecureConnection import root_server
from SecureConnection import server
from SecureConnection import monitor
from SecureConnection.root_server import int_to_bytes,communication_prepare
import threading
import torchxq
import numpy as np
import heapq
import json
import os
from collections import deque
from util.model_card import available_models, ModelCard, retrieve_sending_dir, retrieve_sending_info, retrieve_file_cfg
from system_pipeline.onnx_backend.optimization import Optimizer
import socket
import traceback
import datetime
from web_monitor import start_web_server
monitor_receive_interval = 10  # set intervals for receiving monitor info from clients
monitor_port = "34567"  # set server port to receive monitor info
active_device_port = "23457"  # port for active device communication
TIMEOUT =10 # Time to wait for new devices to connect to servers
MODEL_EXIST_ON_DEVICE = True  # set True if the model exists on the mobile device, will skip model creation and transmission
runtime_option = False  # set True if the load balance is runtime
split_size = 2
device_number =2
task = "Generation"
root_dir = os.path.dirname(os.path.abspath(__file__))
residual_connection_option = True

# Add global device pool and related locks
all_devices_pool = deque()  # Global device pool, stores all registered devices
active_tasks = {}  # Format: {task_id: {"devices": devices_list, "status": status}}
devices_pool_lock = threading.Lock()  # Thread lock for device pool
device_identifiers_map = {}  # Stores the mapping between device ID and its ZMQ identifier: {device_id: identifier}
device_identifiers_lock = threading.Lock()  # Thread lock for identifier mapping
import global_config
global Quntization_Option
 
# Define the communication function for active devices
def communication_open_close_active(sender, config, device_id, status, lock, open=True):
    """
    Handle control information exchange for active devices
    Similar to communication_open_close but specifically for active devices
    """
    client_id = None
    with device_identifiers_lock:
        if device_id in device_identifiers_map:
            client_id = device_identifiers_map[device_id]
    
    if not client_id:
        print(f"Error: Cannot find identifier for device {device_id}")
        return
    
    device_ip = None
    for device in device_pool_manager.active_devices:
        if device.get("device_id") == device_id:
            device_ip = device.get("ip")
            break
    
    if not device_ip:
        print(f"Error: Cannot find IP address for device {device_id}")
        return
    
    print(f"Starting communication thread for active device {device_id} ({device_ip})")
    
    # Set a long timeout (30 seconds)
    original_timeout = sender.getsockopt(zmq.RCVTIMEO)
    sender.setsockopt(zmq.RCVTIMEO, 30000)  # Set to 30 seconds
    
    try:
        while True:
            print('enter communication open close active')
            with lock[0]:
                # print('Start receiving')
                try:
                    info = sender.recv_multipart()  # Blocking mode, but with 30s timeout
                except zmq.error.Again:
                    # Continue trying after timeout
                    print(f"Active device {device_id} receive timeout, waiting...")
                    continue
            
            # The following is the processing after successfully receiving a message
            client_id = info[0]
            msg = info[1]
            print(client_id + msg)
            # print("Signal received")
            
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
                       
                    # Send start inference signal
                        sender.send_multipart([client_id, b'Start'])
                        status[client_id] = b'Start'
                        print(f"Active device Status: Start {config['ids'][client_id]}")
                        break
    except Exception as e:
        print(f"Active device {device_id} communication thread error: {e}")
        traceback.print_exc()
    finally:
        # Restore the original timeout setting
        try:
            sender.setsockopt(zmq.RCVTIMEO, original_timeout)
        except:
            pass
        print(f"Active device {device_id} communication thread ended")

# Add device pool manager class
class DevicePoolManager:
    def __init__(self):
        # Use thread-safe data structures
        self.device_pool = deque()            # All registered active device pool (non-working devices)
        self.working_devices = deque()        # Working device pool (devices registered in the initial stage)
        self.active_devices =    deque()         # {task_id: device_list} Devices used by current active tasks
        self.failed_working_devices = deque() # Working device failure pool
        self.failed_active_devices = deque()  # Active device failure pool
        self.task_counter = 0
        
        # Use atomic operations to manage device status
        self.device_status = {}  # {device_id: {status, last_heartbeat, info}}
        self.device_heartbeats = {}           # Record the last heartbeat time of the device
        self.heartbeat_timeout = 3          # Heartbeat timeout (seconds)
        self.heartbeat_check_interval = 1   # Heartbeat check interval (seconds)
        self.initialization_complete = False  # Flag whether the initialization phase is complete
        self.active_device_threads = {}       # Store active device communication threads

    def set_initialization_complete(self):
        """Mark the initialization phase as complete, set the current devices in the device pool as working devices"""
        # Use atomic operations to update the device pool
        # self.working_devices = deque(self.device_pool)
        # self.device_pool.clear()
        self.initialization_complete = True
        
        print(f"Initialization phase complete! Total {len(self.working_devices)} working devices")
        # Print detailed information of all working devices
        for i, device in enumerate(self.working_devices):
            device_id = device.get("device_id", "N/A")
            ip = device.get("ip", "N/A")
            role = device.get("role", "N/A")
            print(f"  Working device {i+1}: ID={device_id}, IP={ip}, Role={role}")
        
        # Update the status of each working device
        for device in self.working_devices:
            device_id = device.get("device_id")
            if device_id:
                self.device_status[device_id] = {
                    "status": "working",
                    "last_heartbeat": time.time(),
                    "info": device.copy()
                }
        
        # Ensure all device statuses are updated
        self.printInfo()
    
    def register_device(self, device_info):
        """Register a new device to the device pool"""
        try:
            device_id = device_info.get("device_id")
            ip = device_info.get("ip")
            
            if not device_id or not ip:
                print("Error: Device registration did not provide ID or IP address")
                return False
            
            current_time = time.time()
            
            # Update device heartbeat time and status (atomic operation)
            self.device_heartbeats[device_id] = current_time
            
            # Check if the device already exists
            device_exists = False
            device_in_working_pool = False
            
            # Check working device pool
            for device in self.working_devices:
                if device.get("device_id") == device_id:
                    device.update(device_info)
                    device_exists = True
                    device_in_working_pool = True
                    print(f"Device already in working device pool, updated: ID={device_id}, IP={ip}")
                    break
            
            # If not in working device pool, check active device pool
            if not device_exists:
                for device in self.active_devices:
                    if device.get("device_id") == device_id:
                        device.update(device_info)
                        device_exists = True
                        print(f"Device already in active device pool, updated: ID={device_id}, IP={ip}")
                        break
            
            # If already exists, update device status
            if device_exists:
                status = "working" if device_in_working_pool else "active"
                self.device_status[device_id] = {
                    "status": status,
                    "last_heartbeat": current_time,
                    "info": device_info.copy()
                }
                print(f"Update device status: ID={device_id}, Status={status}")
                return status
            
            # Device does not exist, need to add
            if self.initialization_complete:
                # After initialization, new devices are directly added to the active device pool
                self.active_devices.append(device_info)
                status = "active"
                print(f"Running phase - new device registered as active device: ID={device_id}, IP={ip}")
                
                # Create a communication thread for the newly registered active device
                self.start_active_device_thread(device_id)
            else:
                # Initialization phase, add to working device pool
                self.working_devices.append(device_info)
                status = "working"
                print(f"Initialization phase - new device registered as working device: ID={device_id}, IP={ip}, Role={device_info.get('role')}")
            
            # Update device status (atomic operation)
            self.device_status[device_id] = {
                "status": status,
                "last_heartbeat": current_time,
                "info": device_info.copy()
            }
            
            # Print device pool status
            self.printInfo()
            return status
            
        except Exception as e:
            print(f"Error during device registration: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_active_device_thread(self, device_id):
        """Start a control communication thread for the active device"""
        try:
            # Check if the thread already exists
            if device_id in self.active_device_threads and self.active_device_threads[device_id].is_alive():
                print(f"Active device {device_id} already has a communication thread running")
                return
            
            # Create status dictionary
           
                
            # Status will be determined by message exchange during communication, not set in advance
            
            # Get active device information
            active_device_info = None
            for device in self.active_devices:
                if device.get("device_id") == device_id:
                    active_device_info = device
                    break
            
            if not active_device_info:
                print(f"Error: Cannot find information for active device {device_id}")
                return
            
            # Get the head node (the first device in the working device pool)
            head_device = None
            if self.working_devices:
                head_device = self.working_devices[0]
            else:
                print(f"Warning: Working device pool is empty, cannot determine head node")
                return
            
            # First get the original ip_module information
            active_device_ip = active_device_info.get("ip", "")
            
            # Get global configuration
            global config, root_dir, Quntization_Option, requested_model
            
            # Determine model name and quantization option
           
            
            
            # Create a modified ip_module, replacing the second IP with the active device's IP
            modified_ip_module = [
                [head_device.get("ip", ""), f"/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom560m_unquantized_res/device0/module0/module.zip"],
                [active_device_ip, f"/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom560m_unquantized_res/device1/module1/module.zip"]
            ]
            
            print(f"Creating modified ip_module for active device {device_id}:")
            print(modified_ip_module)
            
            # Get sending directory
            to_send_path = retrieve_sending_dir(root_dir, requested_model, 
                                           quantization_option=Quntization_Option,
                                           residual_connection=residual_connection_option)
            
            # Use retrieve_file_cfg to get file configuration
            file_cfg = retrieve_file_cfg(modified_ip_module)
            
            # Use retrieve_sending_info to get graph and dependency information
            ip_graph, dependencyMap = retrieve_sending_info(
                root_dir, requested_model, 
                ip_module_list=modified_ip_module,
                quantization_option=Quntization_Option,
                residual_connection=residual_connection_option
            )
            
            # Create session index
            session = ["0", "1"]  # Simple session index
            
            # Create pre-filled device_config
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
                "num_device": 2,  # Head node and active device, total 2
                "skip_model_transmission": True,
                "dependency": dependencyMap
            }
            for idx, fPath in dependencyMap.items():
                file = open(fPath, "r")
                data = json.load(file)
                device_config["dependency"][idx] = data
            print(f"device_config: {device_config}")
            # # Add device ID mapping
            # device_config["ids"][head_device.get("device_id", "")]= head_device.get("ip", "").encode('utf-8')
            # device_config["ids"][device_id] = active_device_ip.encode('utf-8')
            
            # Print generated configuration information
            print(f"Creating pre-filled configuration for active device {device_id}:")
            print(f"  Head node: {device_config['head_node']}")
            print(f"  Tail node: {device_config['tail_node']}")
            print(f"  IP graph: {device_config['graph']}")
            print(f"  Session index: {device_config['session_index']}")
            
            # Create communication thread
            global active_socket
            thread = threading.Thread(
                target=communication_open_close_active,
                args=(active_socket, device_config, device_id, global_config.active_device_status, [threading.Lock(), threading.Lock()]),
                daemon=True
            )
            thread.name = f"ActiveDevice-{device_id}"
            thread.start()
            
            # Save thread reference
            self.active_device_threads[device_id] = thread
            print(f"Started communication thread for active device {device_id}")
            
        except Exception as e:
            print(f"Error starting communication thread for active device {device_id}: {e}")
            traceback.print_exc()
    
    def update_device_heartbeat(self, device_id):
        """Update device heartbeat time, using atomic operation"""
        if not device_id:
            print("Warning: Attempt to update heartbeat for invalid device ID")
            return False
        
        current_time = time.time()
        old_time = self.device_heartbeats.get(device_id, 0)
        
        # Use atomic operation to update heartbeat time
        self.device_heartbeats[device_id] = current_time
        
        # If device status exists, update the heartbeat time in the status
        if device_id in self.device_status:
            self.device_status[device_id]["last_heartbeat"] = current_time
        
        # Record the heartbeat time difference for monitoring
        if old_time > 0:
            time_diff = current_time - old_time
            if time_diff > self.heartbeat_timeout / 2:
                print(f"Warning: Device {device_id} heartbeat interval is long: {time_diff:.1f}s")
            else:
                pass
                # print(f"Device {device_id} heartbeat updated: {time_diff:.1f}s ago")
        else:
            print(f"Device {device_id} first heartbeat recorded")
        
        return True
    
    def printInfo(self):
        print("\nDevice pool status:")
        print(f"Working devices: {len(self.working_devices)}")
        print(f"Active devices: {len(self.active_devices)}")
        print(f"Failed working devices: {len(self.failed_working_devices)}") 
        print(f"Failed active devices: {len(self.failed_active_devices)}")
        print(f"Initialization status: {'Complete' if self.initialization_complete else 'Incomplete'}")
        # Print active device thread status
        if hasattr(self, 'active_device_threads') and self.active_device_threads:
            active_threads = sum(1 for t in self.active_device_threads.values() if t.is_alive())
            print(f"Active device communication threads: {active_threads}/{len(self.active_device_threads)}")

 
device_pool_manager = DevicePoolManager()

def heartbeat_check_thread():
    """Heartbeat check thread"""
    print("Heartbeat check thread started, checking device heartbeat status every {} seconds, timeout {} seconds".format(
        device_pool_manager.heartbeat_check_interval, 
        device_pool_manager.heartbeat_timeout
    ))
    
    consecutive_empty_checks = 0
    already_processed_failures = set()  # Used to track already processed failed devices
    
    while True:
        try:
            # print(f"\nChecking all device heartbeat status... Current time: {time.time():.2f}")
            current_time = time.time()
            
            # Get device status before failure
            before_count = {
                'working': len(device_pool_manager.working_devices),
                'active': len(device_pool_manager.active_devices),
                'failed_working': len(device_pool_manager.failed_working_devices),
                'failed_active': len(device_pool_manager.failed_active_devices)
            }
            
            # Collect timed-out devices without holding the lock
            failed_devices = []
            
            # Check all device heartbeat status
            for device_id, last_heartbeat in list(device_pool_manager.device_heartbeats.items()):
                heartbeat_age = current_time - last_heartbeat
                
                # Get current device status, ensure no accumulated status prefix
                current_status = device_pool_manager.device_status.get(device_id, {}).get("status", "unknown")
                # Clear possible duplicate failed_ prefix
                if current_status.startswith("failed_failed_"):
                    clean_status = "failed_" + current_status.split("failed_")[-1]
                    device_pool_manager.device_status[device_id]["status"] = clean_status
                    current_status = clean_status
                
                # If device heartbeat times out and is not already processed, mark as failed
                if heartbeat_age > device_pool_manager.heartbeat_timeout and device_id not in already_processed_failures:
                    # Determine which pool the device is in
                    device_info = device_pool_manager.device_status.get(device_id, {}).get("info", {})
                    
                    print(f"Device {device_id} heartbeat timeout ({heartbeat_age:.1f}s), current status: {current_status}")
                    
                    if device_info and not current_status.startswith("failed_"):
                        # Add failure info
                        device_info["failure_time"] = current_time
                        device_info["failure_reason"] = f"Heartbeat timeout ({heartbeat_age:.1f}s)"
                        failed_devices.append((device_id, current_status, device_info.copy()))
                    
                    # Add device to processed set to avoid duplicate processing
                    already_processed_failures.add(device_id)
                elif heartbeat_age <= device_pool_manager.heartbeat_timeout and current_status.startswith("failed_"):
                    # Device recovered, remove from processed set
                    if device_id in already_processed_failures:
                        already_processed_failures.remove(device_id)
                    print(f"Device {device_id} has recovered ({heartbeat_age:.1f}s), previous status: {current_status}")
                    # Here you can add device recovery logic
                else:
                    pass
                    # print(f"Device {device_id} heartbeat normal ({heartbeat_age:.1f}s), current status: {current_status}")
            
            # Handle timed-out devices, use atomic operation
            failures_count = 0
            
            for device_id, status, device_info in failed_devices:
                # Handle failure based on device status
                if status == "working":
                    # Remove from working device pool
                    for i, device in enumerate(device_pool_manager.working_devices):
                        if device.get("device_id") == device_id:
                            device_pool_manager.working_devices.remove(device)
                            device_pool_manager.failed_working_devices.append(device_info)
                            print(f"Working device {device_id} moved to failure pool")
                            failures_count += 1
                            break
                elif status == "active":
                    # Remove from active device pool
                    for i, device in enumerate(device_pool_manager.active_devices):
                        if device.get("device_id") == device_id:
                            device_pool_manager.active_devices.remove(device)
                            device_pool_manager.failed_active_devices.append(device_info)
                            print(f"Active device {device_id} moved to failure pool")
                            failures_count += 1
                            break
                
                # Update device status, ensure status prefix does not accumulate
                if device_id in device_pool_manager.device_status:
                    base_status = status.split("_")[-1] if "_" in status else status
                    device_pool_manager.device_status[device_id]["status"] = f"failed_{base_status}"
            
            # Get device status after failure
            after_count = {
                'working': len(device_pool_manager.working_devices),
                'active': len(device_pool_manager.active_devices),
                'failed_working': len(device_pool_manager.failed_working_devices),
                'failed_active': len(device_pool_manager.failed_active_devices)
            }
            
            # Check for changes
            status_changed = (
                before_count['working'] != after_count['working'] or
                before_count['active'] != after_count['active'] or
                before_count['failed_working'] != after_count['failed_working'] or
                before_count['failed_active'] != after_count['failed_active']
            )
            
            # If there are changes, print details
            if failures_count > 0 or status_changed:
                print("\n⚠️ Device pool status changed:")
                print(f"  Working devices: {before_count['working']} -> {after_count['working']}")
                print(f"  Active devices: {before_count['active']} -> {after_count['active']}")
                print(f"  Failed working devices: {before_count['failed_working']} -> {after_count['failed_working']}") 
                print(f"  Failed active devices: {before_count['failed_active']} -> {after_count['failed_active']}")
                
                if failures_count > 0:
                    print(f"\nDetected {failures_count} new failed devices in this check")
                
                consecutive_empty_checks = 0
            else:
                consecutive_empty_checks += 1
                if consecutive_empty_checks <= 2:
                    # print("\nDevice pool status normal (no change):")
                    device_pool_manager.printInfo()
                else:
                    pass
                    # print(f"Device pool status normal (no change for {consecutive_empty_checks} consecutive checks)")
            
            # Reprint status every 5 checks with no change
            if consecutive_empty_checks > 0 and consecutive_empty_checks % 5 == 0:
                print("\nPeriodic status update:")
                device_pool_manager.printInfo()
                
        except Exception as e:
            print(f"Heartbeat check thread error: {e}")
            import traceback
            traceback.print_exc()
            
        # Wait for next check
        time.sleep(device_pool_manager.heartbeat_check_interval)

def handle_device_registration_and_heartbeat(socket, port):
    """Handle device registration, heartbeat, and status query in a separate thread"""
    global ip_graph_requested  # Add global declaration
    
    try:
        print(f"Device registration and heartbeat service started, listening on port {port}")
        # Configure socket timeout to prevent blocking operations
        socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second receive timeout
        socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second send timeout
        
        # Create a flag to indicate whether the system is handling a failure
        system_handling_failure = False
        
        while True:
            try:
                # Check if socket is closed
                if socket.closed:
                    print("Socket closed, exiting registration and heartbeat service")
                    break
                    
                # Receive message
                try:
                    message = socket.recv_multipart()
                except zmq.error.Again:
                    # Receive timeout, continue loop
                    continue
                
                if not message or len(message) < 2:
                    print("Warning: Received empty or incomplete message")
                    continue
                
                # Parse message
                identifier = message[0]  # Device identifier
                action = message[1].decode()  # Action type
                
                # Safely display identifier to avoid decode errors
                if isinstance(identifier, bytes):
                    try:
                        id_str = identifier.decode('utf-8')
                    except UnicodeDecodeError:
                        # If cannot decode as UTF-8, use hex representation
                        id_str = identifier.hex()
                else:
                    id_str = str(identifier)
                
                # print(f"Received message: identifier={id_str}, action={action}")
                
                # Get data based on message type
                if len(message) > 2:
                    data_raw = message[2]
                    try:
                        data = json.loads(data_raw.decode())
                    except:
                        data = {}
                else:
                    data = {}
                
                # Handle message based on action type
                if action == "RegisterIP":
                    # Handle device registration
                    ip = data.get("ip")
                    role = data.get("role")
                    model_request = data.get("model", None)  # Only header device will send model
                    
                    if not all([ip, role]):
                        print(f"Warning: Device registration info incomplete: {data}")
                        # socket.send_multipart([
                        #     identifier,
                        #     b"REGISTRATION_FAILED",
                        #     b"Missing required fields"
                        # ])
                        continue
                    
                    # Create device info - use unique identifier's hex as device ID
                    device_id = identifier.hex() if isinstance(identifier, bytes) else str(identifier)
                    
                    # Save device identifier for later communication
                    with device_identifiers_lock:
                        device_identifiers_map[device_id] = identifier
                        print(f"Device identifier saved: {device_id}")
                    
                    device = {
                        "device_id": device_id,
                        "ip": ip,
                        "role": role,
                        "device_type": "mobile",  # Default device type
                        "os": "android",  # Default OS
                        "model": model_request  # Save requested model
                    }
                    
                    print(f"Processing device registration: ID={device_id}, IP={ip}, Role={role}")
                    
                    # Add to ip_graph_requested for later model sending
                    if identifier not in ip_graph_requested:
                        ip_graph_requested.append(identifier)
                        print(f"Device identifier added to ip_graph_requested")
                    
                    # Register device
                    status = device_pool_manager.register_device(device)
                    print("status:",status)
                    # Send response message
                    try:
                        if status=="active":
                            # Send signal whether monitoring is needed
                             
                            socket.send_multipart([identifier, b"active"])
                         
                            print("Sent active")
                        if status=="working":
                            socket.send_multipart([identifier, b"working"])
                            
                            print("Sent working")
                    except zmq.error.ZMQError as e:
                        print(f"Error sending registration response: {e}")

                # Normal heartbeat
                elif action == "HEARTBEAT" or action == "HeartDetect":
                    # Handle heartbeat message - use unique identifier's hex as device ID
                    device_id = identifier.hex() if isinstance(identifier, bytes) else str(identifier)
                    
                    # Update device identifier mapping
                    with device_identifiers_lock:
                        device_identifiers_map[device_id] = identifier
                    
                    if not device_id:
                        print("Warning: Heartbeat message missing device ID")
                        socket.send_multipart([identifier, b"HEARTBEAT_FAILED"])
                        continue
                    
                    # Update heartbeat time
                    success = device_pool_manager.update_device_heartbeat(device_id)
                    
                    # Send response, including system status information
                    try:
                        if success:
                            # Check if system has failures, but avoid repeating detection during failure handling
                            if not system_handling_failure:
                                system_has_failures = (
                                    len(device_pool_manager.failed_working_devices) > 0 or 
                                    len(device_pool_manager.failed_active_devices) > 0
                                )
                                
                                if system_has_failures:
                                    # Set failure handling flag to avoid repeating trigger
                                    system_handling_failure = True
                                    
                                    # First notify current heartbeat device
                                    socket.send_multipart([identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_FAILURE"])
                                    print(f"Device {device_id} heartbeat response: system has failures")
                                    
                                    # Asynchronously trigger failure handling, to avoid blocking heartbeat response thread
                                    def trigger_failure_handling():
                                        try:
                                            # First notify all online devices about system failure status
                                            notify_all_devices = []
                                            device_identifiers = {}
                                            
                                            # Collect all online devices' identifiers
                                            with device_identifiers_lock:
                                                for dev_id, dev_identifier in device_identifiers_map.items():
                                                    # Exclude current notified devices
                                                    if dev_id != device_id:
                                                        notify_all_devices.append(dev_id)
                                                        device_identifiers[dev_id] = dev_identifier
                                            
                                            print(f"Notifying other {len(notify_all_devices)} devices about system failure status...")
                                            
                                            # Send failure notification to all collected devices
                                            for dev_id in notify_all_devices:
                                                try:
                                                    dev_identifier = device_identifiers[dev_id]
                                                    socket.send_multipart([dev_identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_FAILURE"])
                                                    print(f"Notifying device {dev_id} about system failure status")
                                                except Exception as e:
                                                    print(f"Failed to notify device {dev_id}: {e}")
                                            
                                            # Server enters failure handling process
                                            handle_system_failure()
                                            
                                            # After failure handling, reset flag
                                            nonlocal system_handling_failure
                                            system_handling_failure = False
                                        except Exception as e:
                                            print(f"Error during failure handling: {e}")
                                            system_handling_failure = False  # Ensure flag reset even on error
                                    
                                    # Start a thread for failure handling
                                    failure_thread = threading.Thread(target=trigger_failure_handling)
                                    failure_thread.daemon = True
                                    failure_thread.start()
                                else:
                                    # System normal
                                    socket.send_multipart([identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_NORMAL"])
                                    # print(f"Device {device_id} heartbeat response: system normal")
                            else:
                                # System is handling failure, inform client to wait
                                socket.send_multipart([identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_HANDLING_FAILURE"])
                                print(f"Device {device_id} heartbeat response: system is handling failure")
                        else:
                            socket.send_multipart([identifier, b"HEARTBEAT_FAILED"])
                            print(f"Device {device_id} heartbeat update failed")
                    except zmq.error.ZMQError as e:
                        print(f"Error sending heartbeat response: {e}")

                # Background+ScreenOn
                elif action == "HEARTBEAT_InBackground_ScreenOn":
                    # Handle heartbeat message - use unique identifier's hex as device ID
                    device_id = identifier.hex() if isinstance(identifier, bytes) else str(identifier)

                    # Update device identifier mapping
                    with device_identifiers_lock:
                        device_identifiers_map[device_id] = identifier

                    if not device_id:
                        print("Warning: Heartbeat message missing device ID")
                        socket.send_multipart([identifier, b"HEARTBEAT_FAILED"])
                        continue

                    # Do not update heartbeat time, force into failure state
                    success = True

                    # Send response, including system status information
                    try:
                        if success:
                            # Check if system has failures, but avoid repeating detection during failure handling
                            if not system_handling_failure:
                                system_has_failures = (
                                        len(device_pool_manager.failed_working_devices) > 0 or
                                        len(device_pool_manager.failed_active_devices) > 0
                                )

                                if system_has_failures:
                                    # Set failure handling flag to avoid repeating trigger
                                    system_handling_failure = True

                                    # First notify current heartbeat device
                                    socket.send_multipart([identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_InBackground_ScreenOn"])
                                    print(f"Device {device_id} heartbeat response: SYSTEM_InBackground_ScreenOn, entering restart state")

                                    # Asynchronously trigger failure handling, to avoid blocking heartbeat response thread
                                    def trigger_failure_handling():
                                        try:
                                            # First notify all online devices about system failure status
                                            notify_all_devices = []
                                            device_identifiers = {}

                                            # Collect all online devices' identifiers
                                            with device_identifiers_lock:
                                                for dev_id, dev_identifier in device_identifiers_map.items():
                                                    # Exclude current notified devices
                                                    if dev_id != device_id:
                                                        notify_all_devices.append(dev_id)
                                                        device_identifiers[dev_id] = dev_identifier

                                            print(f"Notifying other {len(notify_all_devices)} devices about system failure status...")

                                            # Send failure notification to all collected devices
                                            for dev_id in notify_all_devices:
                                                try:
                                                    dev_identifier = device_identifiers[dev_id]
                                                    socket.send_multipart(
                                                        [dev_identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_FAILURE"])
                                                    print(f"Notifying device {dev_id} about system failure status")
                                                except Exception as e:
                                                    print(f"Failed to notify device {dev_id}: {e}")

                                            # Server enters failure handling process
                                            handle_system_failure()

                                            # After failure handling, reset flag
                                            nonlocal system_handling_failure
                                            system_handling_failure = False
                                        except Exception as e:
                                            print(f"Error during failure handling: {e}")
                                            system_handling_failure = False  # Ensure flag reset even on error

                                    # Start a thread for failure handling
                                    failure_thread = threading.Thread(target=trigger_failure_handling)
                                    failure_thread.daemon = True
                                    failure_thread.start()
                                else:
                                    # System normal
                                    socket.send_multipart([identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_NORMAL"])
                                    # print(f"Device {device_id} heartbeat response: system normal")
                            else:
                                # System is handling failure, inform client to wait
                                socket.send_multipart(
                                    [identifier, b"HEARTBEAT_RECEIVED", b"SYSTEM_HANDLING_FAILURE"])
                                print(f"Device {device_id} heartbeat response: system is handling failure")
                        else:
                            socket.send_multipart([identifier, b"HEARTBEAT_FAILED"])
                            print(f"Device {device_id} heartbeat update failed")
                    except zmq.error.ZMQError as e:
                        print(f"Error sending heartbeat response: {e}")
                # Handle failure recovery confirmation
                elif action == "FAILURE_RECOVERY_ACK":
                    device_id = identifier.hex() if isinstance(identifier, bytes) else str(identifier)
                    client_ip = config["ids"].get(identifier, b"unknown").decode() if isinstance(config["ids"].get(identifier), bytes) else config["ids"].get(identifier, "unknown")
                    print(f"Device {client_ip} (ID: {device_id}) confirmed receipt of failure recovery signal")
                    
                    # Record confirmation status
                    if "recovery_acks" not in config:
                        config["recovery_acks"] = {}
                    config["recovery_acks"][device_id] = True
                    
                    # Check if all devices have confirmed
                    expected_count = len(config["ids"]) - sum(1 for ip in config["ids"].values() 
                                                    if (ip.decode() if isinstance(ip, bytes) else ip) in config.get("failed_ips", []))
                    ack_count = len(config["recovery_acks"])
                    
                    print(f"Failure recovery progress: {ack_count}/{expected_count} devices confirmed")
                    
                    # When all expected devices have confirmed, can clear failure status
                    if ack_count >= expected_count:
                        print("All devices confirmed failure recovery, clear system failure status")
                        if "system_status" in config:
                            del config["system_status"]
                        if "recovery_status" in config:
                            del config["recovery_status"]
                        if "failed_ips" in config:
                            del config["failed_ips"]
                        if "recovery_acks" in config:
                            del config["recovery_acks"]
                    
                    # Reply to confirmation received
                    socket.send_multipart([identifier, b"RECOVERY_ACK_RECEIVED"])
                
                # Handle confirmation for no replacement device
                elif action == "SYSTEM_FAILURE_NO_REPLACEMENT_ACK":
                    device_id = identifier.hex() if isinstance(identifier, bytes) else str(identifier)
                    client_ip = config["ids"].get(identifier, b"unknown").decode() if isinstance(config["ids"].get(identifier), bytes) else config["ids"].get(identifier, "unknown")
                    print(f"Device {client_ip} (ID: {device_id}) confirmed receipt of system failure notification (no replacement device)")
                    
                    # Record confirmation status
                    if "suspended_acks" not in config:
                        config["suspended_acks"] = {}
                    config["suspended_acks"][device_id] = True
                    
                    # Check if all devices have confirmed
                    expected_count = len(config["ids"]) - sum(1 for ip in config["ids"].values() 
                                                    if (ip.decode() if isinstance(ip, bytes) else ip) in config.get("failed_ips", []))
                    ack_count = len(config["suspended_acks"])
                    
                    print(f"System suspension progress: {ack_count}/{expected_count} devices confirmed")
                    
                    # Reply to confirmation received
                    socket.send_multipart([identifier, b"SUSPENSION_ACK_RECEIVED"])
                
                else:
                    print(f"Unknown message type: {action}")
                    try:
                        socket.send_multipart([identifier, b"UNKNOWN_ACTION"])
                    except zmq.error.ZMQError as e:
                        print(f"Error sending unknown action response: {e}")
                    
            except zmq.error.ZMQError as e:
                print(f"ZMQ error: {e}")
                if socket.closed:
                    print("Socket closed, exiting registration and heartbeat service")
                    break
                continue
            except Exception as e:
                print(f"Error processing message: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    except Exception as e:
        print(f"Error during device registration and heartbeat service: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Device registration and heartbeat service stopped")

def handle_system_failure():
  
    print("System failure handling started...")
    global config, ip_graph, session
    
    # Ensure traceback is imported
    import traceback
    
    # Record start time of processing
    start_time = datetime.datetime.now()
    print(f"Failure handling start time: {start_time}")
    
    # No longer create new communication_socket, instead use existing socket
    # Existing socket will be used in the loop of communication_open_close function
    
    try:
        # 1. Determine which device failed
        failed_devices = []
        
        # Check for failed devices in working device pool
        for device in list(device_pool_manager.failed_working_devices):
            failed_devices.append(device)
            print(f"Found failed working device: ID={device.get('device_id')}, IP={device.get('ip')}, Role={device.get('role')}")
        
        if not failed_devices:
            print("No failure detected in devices, no failure handling needed")
            return
        
        # If there is no config or main variables, it means system initialization is not complete
        if not 'config' in globals() or config is None:
            print("System not fully initialized, cannot handle failure")
            return
            
        print(f"Processing {len(failed_devices)} failed devices...")
        
        # Build failure IP list and save to config
        failed_ips = [device.get('ip') for device in failed_devices]
        config["failed_ips"] = failed_ips
        
        # 2. Use devices from active device pool instead of failed devices
        replacement_mapping = {}  # {failed device IP: replacement device information}
        
        # Check if active device pool is empty
        if not device_pool_manager.active_devices:
            print("Warning: Active device pool is empty, cannot provide replacement device")
            # Although unable to replace device, still notify all running devices of device failure
            try:
                # Set status in config to need recovery
                config["system_status"] = "RECOVERY_NEEDED"
                config["recovery_status"] = "NO_REPLACEMENT"
                print("System marked as needing recovery but without replacement, notifying clients through communication loop")
                
                # Even if unable to replace device, clear failure pool to prevent repeated triggering of failure handling
                print("Clearing failure device pool to prevent repeated handling...")
                device_pool_manager.failed_working_devices.clear()
                device_pool_manager.failed_active_devices.clear()
                
                # Reset device status
                for device in failed_devices:
                    device_id = device.get("device_id")
                    if device_id in device_pool_manager.device_status:
                        print(f"Resetting status of device {device_id}")
                        device_pool_manager.device_status[device_id]["status"] = "inactive"
                
                return
            except Exception as e:
                print(f"Error setting recovery status: {e}")
                traceback.print_exc()
                return
        
        for failed_device in failed_devices:
            failed_ip = failed_device.get("ip")
            failed_role = failed_device.get("role")
            failed_idx = -1
            
            # Find position of failed device in IP graph
            for i, ip in enumerate(ip_graph):
                if ip == failed_ip:
                    failed_idx = i
                    break
            
            if failed_idx == -1:
                print(f"Warning: Failed device {failed_ip} not in IP graph, skipping")
                continue
            
            # Select a device from active device pool as replacement
            replacement_device = None
            
            if device_pool_manager.active_devices:
                # Select first device from active device pool
                replacement_device = device_pool_manager.active_devices.popleft()
                print(f"Using active device {replacement_device.get('ip')} as replacement for failed device {failed_ip}")
                
                # Update role of replacement device to match failed device
                replacement_device["role"] = failed_role
                
                # Add to replacement mapping
                replacement_mapping[failed_ip] = replacement_device
            else:
                print(f"Error: Active device pool is empty, cannot find replacement for failed device {failed_ip}")
                # Set system to need recovery but without replacement
                config["system_status"] = "RECOVERY_NEEDED"
                config["recovery_status"] = "NO_REPLACEMENT"
                return
        
        if not replacement_mapping:
            print("No available replacement device, failure handling failed")
            config["system_status"] = "RECOVERY_NEEDED"
            config["recovery_status"] = "NO_REPLACEMENT"
            return
        
        # 3. Modify config and related information
        new_ip_graph = ip_graph.copy()
        new_session = session.copy()
        
        # Update IP graph
        for old_ip, new_device in replacement_mapping.items():
            new_ip = new_device.get("ip")
            
            # Replace in IP graph
            for i, ip in enumerate(new_ip_graph):
                if ip == old_ip:
                    new_ip_graph[i] = new_ip
                    print(f"IP graph replacement: {old_ip} -> {new_ip} at position {i}")
            
            # Replace in config's ids
            for client_id, device_ip in config["ids"].items():
                if device_ip.decode() if isinstance(device_ip, bytes) else device_ip == old_ip:
                    config["ids"][client_id] = new_ip.encode() if isinstance(device_ip, bytes) else new_ip
                    print(f"Config IDs replacement: {old_ip} -> {new_ip}")
            
            # Update head and tail nodes (if needed)
            if config["head_node"] == old_ip:
                config["head_node"] = new_ip
                print(f"Head node replacement: {old_ip} -> {new_ip}")
                
            if config["tail_node"] == old_ip:
                config["tail_node"] = new_ip
                print(f"Tail node replacement: {old_ip} -> {new_ip}")
        
        # Build new configuration update
        config["graph"] = ",".join(new_ip_graph).encode('utf-8')
        config["session_index"] = ";".join(new_session).encode('utf-8')
        
        print("Configuration updated:")
        print(f"New IP graph: {new_ip_graph}")
        print(f"New session index: {new_session}")
        print(f"New config: {config}")
        # 4. Prepare to send failure control information through existing communication loop
        config["system_status"] = "RECOVERY_NEEDED"
        config["recovery_status"] = "HAS_REPLACEMENT"
        config["new_graph"] = new_ip_graph
        config["new_session"] = new_session
        print("System marked as needing recovery, will notify clients through communication loop")
        print(f"config(after failure recovery): {config}")
        # 5. Add replacement device to working device pool
        for old_ip, replacement_device in replacement_mapping.items():
            device_pool_manager.working_devices.append(replacement_device)
            print(f"Replacement device {replacement_device.get('ip')} added to working device pool")
        
        # Clear failure device pool to prevent repeated triggering of failure handling
        print("Clearing failure device pool...")
        device_pool_manager.failed_working_devices.clear()
        device_pool_manager.failed_active_devices.clear()
        
        # Reset flag for processed devices
        if 'already_processed_failures' in globals():
            already_processed_failures.clear()
            print("Reset failure device processing flag")
            
        print("System failure handling preparation completed, waiting to send failure recovery signal")
        
       
        
    except Exception as e:
        print(f"Error handling system failure: {e}")
        traceback.print_exc()

def main():
    """Main function, includes device registration, model splitting, and sending"""
    global devices        # Reference global variable
    global ip_graph_requested
    global ip_graph  # Add global declaration
    global active_socket  # Add global declaration
    
    try:
        start = time.time()
        context = zmq.Context()
        
        # Start Web monitoring server
        web_thread = threading.Thread(
            target=start_web_server,
            args=(device_pool_manager,),
            daemon=True
        )
        web_thread.start()
        print("Web monitoring server started, access http://localhost:34568 to view status")
        
        # Create a single registration/communication/heartbeat socket
        PORT = 23456  # Set uniform server port
        registration_socket = context.socket(zmq.ROUTER)
        registration_socket.bind(f"tcp://*:{PORT}")
        
        # Create active device communication socket
        active_socket = context.socket(zmq.ROUTER)
        active_socket.bind(f"tcp://*:{active_device_port}")
        print(f"Active device communication socket bound to port: {active_device_port}")
        
        # Set timeout for registration socket, only for registration and heartbeat
        registration_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second receive timeout
        registration_socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second send timeout
        
        # Set timeout for active device socket
        active_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second receive timeout
        active_socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second send timeout
        
        # Set default model to prevent undefined error
        global requested_model
        requested_model = "bloom560m"  # Default model
        
        # Define constant
        running = True  # Control flag for main thread
        
        # Initialize global device set
        devices = deque()
        ip_graph_requested = []  # Store IP addresses of all requested devices
        ip_graph = []  # Initialize ip_graph list
        
        print("==== Distributed inference system started ====")
        print(f"Waiting for device registration, timeout for initialization: {TIMEOUT} seconds")
        print(f"Listening on port: {PORT}")
        
        # Start device registration and heartbeat service thread
        registration_thread = threading.Thread(
            target=handle_device_registration_and_heartbeat,
            args=(registration_socket, PORT),  # Pass socket and port
            daemon=True
        )
        registration_thread.start()
        
        # Start heartbeat check thread
        heartbeat_thread = threading.Thread(
            target=heartbeat_check_thread,
            daemon=True
        )
        heartbeat_thread.start()
        
        # Wait for initialization phase to complete
        print("Waiting for initialization phase to complete...")
        initialization_complete = False
        last_registration_time = time.time()  # Record last device registration time
        device_count = 0  # Record current device count
        
        # Use timeout check in main loop to avoid permanent blocking
        while not initialization_complete:
            current_device_count = 0
            
            # Get current device count and check for changes - hold lock for minimal time
            with devices_pool_lock:
                current_device_count = len(device_pool_manager.working_devices)
                initialization_complete = device_pool_manager.initialization_complete
            
            # If device count changes, update last registration time
            if current_device_count > device_count:
                last_registration_time = time.time()
                device_count = current_device_count
                print(f"New device registered, current device count: {device_count}")
            
            # Check if more than 10 seconds have passed without new device registration
            if time.time() - last_registration_time >= TIMEOUT and not initialization_complete:
                if device_count > 0:  # Ensure at least one device
                    # End initialization phase, set current devices as working devices
                    with devices_pool_lock:
                        if not device_pool_manager.initialization_complete:  # Double check to avoid race condition
                            device_pool_manager.set_initialization_complete()
                            # Add devices to legacy-compatible device set
                            devices.clear()  # Clear existing devices
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
                    print(f"Initialization complete, collected {device_count} working devices")
                else:
                    print("Warning: Initialization timeout, but no devices registered")
                    return  # Exit directly if no devices
            
            # Periodically print status
            if time.time() - last_registration_time > 0 and int(time.time() - last_registration_time) % 2 == 0:
                print(f"Waiting for initialization... {int(time.time() - last_registration_time)} seconds since last device registration")
                print(f"Currently collected {device_count} devices")
            
            time.sleep(0.5)  # Reduce wait time, check more frequently
        
        if device_count == 0:
            print("Initialization failed: no devices registered")
            return
        
        print(f"Initialization phase ended, number of working devices: {len(device_pool_manager.working_devices)}")
        print(f"Preparing to split and send model...")
      
        # ============== Model splitting and sending section ==============
        if requested_model:
        # Determine model and quantization option
            if requested_model == "bloom560m":
                global Quntization_Option
                Quntization_Option = False
            elif requested_model == "bloom560m-int8":
               
                Quntization_Option = True
               
                requested_model = "bloom560m"  # Use non-quantized name internally
            else:
                print(f"Using default model: bloom560m")
               
                Quntization_Option = False
             
                requested_model = "bloom560m"
            
            # Retrieve model sending directory
            to_send_path = retrieve_sending_dir(root_dir, requested_model, 
                                            quantization_option=Quntization_Option,
                                            residual_connection=residual_connection_option)
            
            # Check if model directory exists
            if os.path.isdir(to_send_path):
                print('Model directory exists, using existing model')
                # Load existing IP module mapping and session info
                with open(os.path.join(to_send_path, 'ip_module.json'), 'r') as file:
                    ip_module_json = file.read()
                
                with open(os.path.join(to_send_path, 'session.json'), 'r') as file:
                    session_index_json = file.read()
                
                ip_module = json.loads(ip_module_json)
                global session 
                session = json.loads(session_index_json)
                file_cfg = retrieve_file_cfg(ip_module)
                
                # Send monitor initialization signal to devices (False means use existing model)
                for ip in ip_graph_requested:
                    registration_socket.send_multipart([ip, b"False"])
            else:
                print('Model directory does not exist, preparing model...')
                # Send monitor initialization signal to devices (True means need to prepare new model)
                for ip in ip_graph_requested:
                    registration_socket.send_multipart([ip, b"True"])
                
                # Create model card object
                model_card = ModelCard(requested_model, 
                                    quantization_option=Quntization_Option, 
                                    task_type=task,
                                    residual_connection=residual_connection_option, 
                                    load_balancing_option=False,
                                    split_size=split_size)
                
                # Prepare optimization info
                mem_util, out_size_map, bytearray_path, flop_module_path, num_flop, module_flop_map, num_modules = model_card.prepare_optimization_info()
                tokenizer_dir = model_card.retreive_tokenizer_path()
                directory_path = os.path.dirname(bytearray_path)

                print(f'bytearray_path: {bytearray_path}')
                print(f'flop_module_path: {flop_module_path}')
                print(f'num_flop: {num_flop}')
                print(f'out_size_map: {out_size_map}')
            
                print(f"Model split size: {model_card.split_size}")
                print("Using Round-Robin allocation method")
                for ip in ip_graph_requested:
                    send.send_multipart([ip, b"ready for monitor"])
                # # start monitor
                monitor_instance = monitor.Monitor(monitor_receive_interval, monitor_port, devices, requested_model, \
                                        bytearray_path, flop_module_path, num_flop, runtime_option)
                thread = threading.Thread(target=monitor_instance.start)
                thread.start()

                num_devices = len(devices)
                monitor_instance.is_monitor_ready.wait()  # Wait for monitor data to be ready

                # Parameters
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
                
                # Allocate modules
                initial_module_arrangement = round_robin_module_arrangement(split_size, split_size)
                overlapping_module_arrangement = initial_module_arrangement
                print(f"Module allocation scheme:\n{initial_module_arrangement}")
                
                # Prepare to send model
                model_dirs = model_card.prepare_model_to_send(module_arrangement=initial_module_arrangement)
                device_module_order = model_card.device_module_arrangement
                device_dir_map = {tuple(device_module_order[i]): model_dirs[i] for i in range(len(model_dirs))}
                ip_device_module_map = {}
                for i in range(len(devices)):
                    ip_device_module_map[devices[i]["ip"].encode("utf-8")] = device_module_order[
                        i]  # .26: [0], .19: [2], ..

                # retrieve session for inference
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
        # Modify IP addresses in file_cfg JSON file
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
                print(f"Using non-quantized model: bloom560m")
                pathList = [str(ip), f"/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom560m_unquantized_res/device{index}/module{index}/module.zip"]
            else:
                pathList = [str(ip), f"/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom560m_quantized_int8_res/device{index}/module{index}/module.zip"]
            
            # if not Quntization_Option:
            #     pathList = [str(ip), f"/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom560m_unquantized_seq/device{index}/module{index}/module.zip"]
            # else:
            #     pathList = [str(ip), f"/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom560m_quantized_int8_seq/device{index}/module{index}/module.zip"]
            
            pathLists.append(pathList)
        
        # Save path list
        with open(os.path.join(to_send_path, 'ip_module.json'), 'w') as file:
            json.dump(pathLists, file)
        
        # Read saved JSON
        with open(os.path.join(to_send_path, 'ip_module.json'), 'r') as file:
            ip_module_json = file.read()
        
        # Process IP module data
        ip_module = json.loads(ip_module_json)
        file_cfg = retrieve_file_cfg(ip_module)
        ip_graph, dependencyMap = retrieve_sending_info(
            root_dir, requested_model, 
            ip_module_list=ip_module,
            quantization_option=Quntization_Option,
            residual_connection=residual_connection_option
        )
        
        print(f'\nGraph: {ip_graph}')
        print(f"Session index: {session}")
        global config
        # Create config
        config = {
            "file_path": file_cfg,
            "num_sample": b'1000',
            "num_device": len(device_pool_manager.working_devices),
            "max_length": b'100',
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
     
        # Read dependency JSON files
        for idx, fPath in dependencyMap.items():
            file = open(fPath, "r")
            data = json.load(file)
            config["dependency"][idx] = data
        
   
        print(f"config(normal case): {config}")
        print("Config complete, preparing to send model...")
        # Start communication threads
       
        threads = []
        lock = threading.Lock()
        locks = [threading.Lock(), threading.Lock()]
        conditions = [threading.Condition() for i in range(len(device_pool_manager.working_devices) + 1)]
        
    

        
        # Create new communication socket, use monitor_port instead of original port
        try:
            print(f"Trying to use monitor_port: {monitor_port} as communication port")
            global communication_socket
            communication_socket = context.socket(zmq.ROUTER)
            communication_socket.bind(f"tcp://*:{monitor_port}")
            print(f"Communication socket bound to monitor_port: {monitor_port}")
        except zmq.error.ZMQError as e:
            print(f"Failed to bind to monitor_port: {e}")
            # Try using original port
           
               
        
        for i in range(config["num_device"]):
            t = threading.Thread(
                target=root_server.communication_open_close, 
                args=(communication_socket, config, global_config.working_device_status, conditions, locks)
            )
            threads.append(t)
        
        # Start all threads
        for i in threads:
            i.start()
        
        # Wait for all threads to finish
        for t in threads:
            t.join()
            if hasattr(t, 'exception') and t.exception:
                print(f"Thread {t.name} encountered exception: {t.exception}")
        
        print("Model loading and allocation complete!")
        
        # Main thread waits for exit signal, periodically prints device pool status
        while running:
            time.sleep(10)  # Print device pool status every 10 seconds
            print("\nCurrent device pool status:")
            device_pool_manager.printInfo()
            print(f"Initialization complete: {'Yes' if device_pool_manager.initialization_complete else 'No'}")
            
            
    except KeyboardInterrupt:
        print("\nUser interrupted, program exiting...")
        running = False
    except Exception as e:
        print(f"Main thread error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        print("Cleaning up resources...")
        try:
            registration_socket.close()
            communication_socket.close()
            active_socket.close()  # Close active device communication socket
            context.term()
        except Exception as e:
            print(f"Error closing resources: {e}")
        print("Program exited")

if __name__ == "__main__":
    main()
