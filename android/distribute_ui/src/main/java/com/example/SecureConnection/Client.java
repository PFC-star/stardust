package com.example.SecureConnection;

import static com.example.distribute_ui.service.BackgroundService.TAG;

import android.content.Context;
import android.util.Log;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import android.content.Context;
import android.os.Environment;
import java.util.ArrayList;
import java.util.Map;
import java.io.File;
import java.io.FileOutputStream;

import org.greenrobot.eventbus.EventBus;
import org.json.JSONException;
import org.json.JSONObject;
import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;
import org.zeromq.ZMQ.Socket;
import org.zeromq.ZMQException;

import java.io.IOException;
import java.util.Objects;

import com.example.distribute_ui.Events;
import com.example.distribute_ui.network.FTPHelper;

import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.lang.reflect.Field;

public class Client {
    /**
     * Establish a TCP connection with the server and return the corresponding Socket object
     * @param context ZeroMQ context
     * @param type Socket type (DEALER, ROUTER, etc.)
     * @param port Target port
     * @param address Target IP address
     * @return Created Socket object
     */
    public Socket establish_connection(ZContext context, SocketType type, int port, String address) {
        Socket socket = context.createSocket(type);
        try {
            Log.d(TAG, "attempt to connect to address: " + address + ", port: " + port);
            socket.connect("tcp://" + address + ":" + port);
            Log.d(TAG, "successfully connected to address: " + address + ", port: " + port);
        } catch (ZMQException e) {
            Log.e(TAG, "connection to address: " + address + ", port: " + port + " failed: " + e.getMessage() + ", error code: " + e.getErrorCode());
            if (e.getErrorCode() == ZMQ.Error.ECONNREFUSED.getCode()) {
                Log.e(TAG, "connection refused, target device may not be started or port not enabled");
            } else if (e.getErrorCode() == ZMQ.Error.ETIMEDOUT.getCode()) {
                Log.e(TAG, "connection timeout, target device may not be accessible");
            }
            // Rethrow exception to maintain original behavior
            throw e;
        } catch (Exception e) {
            Log.e(TAG, "connection to address: " + address + ", port: " + port + " occurred unknown exception: " + e.getMessage());
            throw e;
        }
        return socket;
    }
    
    private Context conText;
    
    /**
     * Main communication handler - Worker device mode
     * Responsible for communication loop with the server, handling device state transitions and fault recovery
     * 
     * State flow:
     * Ready -> Open -> Prepare -> Initialized -> Start -> Running -> Finish
     * Fault recovery flow:
     * Running -> Recovery -> Recovering -> Running
     * 
     * @param cfg Device configuration object
     * @param com Communication object
     * @param receiver Socket for receiving messages
     * @param modelName Model name
     * @param serverIp Server IP
     * @param role Device role
     * @throws Exception Possible exceptions during communication
     */
    public void communicationOpenClose(Config cfg, Communication com, Socket receiver, String modelName, String serverIp, int role) throws Exception {
        Log.d(TAG, "Start communicationOpenClose");
        Communication.Params param = com.param;
        while (true) {
            // When status is Ready, send "Ready" and own IP to the server, wait for server message
            if (param.status.equals("Ready")) {
                Log.d(TAG, "Status: Ready");
                receiver.sendMore("Ready");         // sendMore means more messages to be sent

                receiver.send(Config.local, 0);     // send means complete message
                Log.d(TAG, "waiting for open signal");

                // Open
                String msg = new String(receiver.recv(0));
                Log.d(TAG, "msg: " + msg);
                if (msg.equals("Open")) {   // When msg is "Open", change status and perform a series of preparations
                    param.status = "Open";
                    System.out.println("Status: Open");

                    receiveIPGraph(cfg, receiver);

                    // Receive session index
                    receiveSessionIndex(receiver);

                    // Receive task type (generation or classification)
                    receiveTaskType(param,receiver);

                    // Receive thread pool size configuration
                    receiveThreadPoolSize(param, receiver);

                    // Receive batch size
                    receiveBatchSize(param, receiver);

                    // Receive maximum sequence length
                    receiveSeqLength(param,receiver);

                    // Receive dependency map (module dependencies)
                    receiveDependencyMap(receiver);

                    String num_devices = new String(receiver.recv(0));
                    Log.d(TAG, "num_devices: " + num_devices);


                    Log.d(TAG, "open status receive info finished");
                }

                // Prepare
                msg = new String(receiver.recv(0));
                Log.d(TAG, "prepare msg: " + msg);
                if (msg.equals("Prepare")) {    // When receiving msg as "Prepare", change status and prepare the model file
                    communicationPrepare(receiver, param, modelName, serverIp, role);  // Prepare the unzipped model file
                }

                // Initialize load balancing and model (create new session and tokenizer)
                LoadBalanceInitialization();
                modelInitialization(cfg, param);
                param.status = "Initialized";
                System.out.println("Status: Initialized");
                receiver.send("Initialized", 0);

                msg = new String(receiver.recv(0));
                System.out.println(msg);

                if (msg.equals("Start")) {
                    param.status = "Start";
                    Log.d(TAG, "Status: Start");
                    receiver.send("Running");
                    Log.d(TAG, "Status: Running");
                    param.status = "Running";
                    if (param.status == "Running") {
                        // Send RunningStatusEvent to set runningStatus to true in BackgroundService
                        EventBus.getDefault().post(new Events.RunningStatusEvent(true));
                        Log.d(TAG, "Post Events.RunningStatusEvent(true)");
                    }
                }
            }



            else if (param.status.equals("Finish")) {
                // Task finished, send finish signal
                receiver.send("Finish");
                String msg = new String(receiver.recv(0));
                System.out.println(msg);
                System.out.println("Status: Close");
                Log.d(TAG, "Status: Close");

                if (msg.equals("Close")) {  // When receiving msg as "Close", close all sockets
                    for(ArrayList<Map<Integer, Socket>> s: com.allSockets)
                        closeSockets(s);
//                    com.context.close();
                }
                System.out.println("Finish task");
                Log.d(TAG, "Finish task");
                break;
            }
            else if (param.status.equals("Recovery")){
                // Received fault recovery signal, enter recovery process
                Log.d(TAG, "entering fault recovery state - Recovery");
                
                // Send recovery request and own IP to the server
                receiver.sendMore("Recovery");         
                receiver.send(Config.local, 0);
                Log.d(TAG, "sent recovery request to server");

                // Receive new IP graph and session index
                Log.d(TAG, "receiving fault recovery information");
                receiveIPGraph(cfg, receiver);
                receiveSessionIndex(receiver);
                
                // Receive new dependency graph (if needed)
                receiveDependencyMap(receiver);
                Log.d(TAG, "fault recovery information received");
                
                // Directly call Communication's handleSystemFailure method
                // This method will update Socket configuration without interrupting inference threads
                com.handleSystemFailure();
                
                // Send recovery completion signal to server
                receiver.sendMore("WaitingStart");
                receiver.send(Config.local, 0);

                String msg = new String(receiver.recv(0));
                if (msg.equals("ResumeStart")) {  // When receiving msg as "ResumeStart", can resume start
                    param.status = "ResumeStart";
                    Log.d(TAG, "notified server recovery completed,ResumeStart");
                }

                // State recovery to Running is done in handleSystemFailure

            }

        }
    }
    
    /**
     * Main communication handler - Active device mode
     * Similar to communicationOpenClose, but for devices in the active device pool
     * Active devices are those not yet assigned tasks but can replace faulty devices at any time
     * 
     * @param cfg Device configuration object
     * @param com Communication object
     * @param receiver Socket for receiving messages
     * @param modelName Model name
     * @param serverIp Server IP
     * @param role Device role
     * @throws Exception Possible exceptions during communication
     */
    public void communicationOpenCloseActive(Config cfg, Communication com, Socket receiver, String modelName, String serverIp, int role) throws Exception {
        Log.d(TAG, "Start communicationOpenCloseActive");
        Communication.Params param = com.param;
        while (true) {
            // When status is Ready, send "Ready" and own IP to the server, wait for server message
            if (param.status.equals("Ready")) {
                Log.d(TAG, "Status: Ready");
                receiver.sendMore("Ready");         // sendMore means more messages to be sent

                receiver.send(Config.local, 0);     // send means complete message
                Log.d(TAG, "waiting for open signal");

                // Open
                String msg = new String(receiver.recv(0));
                Log.d(TAG, "msg: " + msg);
                if (msg.equals("Open")) {   // When msg is "Open", change status and perform a series of preparations
                    param.status = "Open";
                    System.out.println("Status: Open");

                    receiveIPGraph(cfg, receiver);

                    receiveSessionIndex(receiver);

                    receiveTaskType(param,receiver);

                    receiveThreadPoolSize(param, receiver);

                    receiveBatchSize(param, receiver);

                    receiveSeqLength(param,receiver);

                    receiveDependencyMap(receiver);

                    String num_devices = new String(receiver.recv(0));
                    Log.d(TAG, "num_devices: " + num_devices);


                    Log.d(TAG, "open status receive info finished");
                }

                // Prepare
                msg = new String(receiver.recv(0));
                Log.d(TAG, "prepare msg: " + msg);
                if (msg.equals("Prepare")) {    // When receiving msg as "Prepare", change status and prepare the model file
//                    Do not prepare model for now

                    communicationPrepare(receiver, param, modelName, serverIp, role);  // Prepare the unzipped model file
                }

                // Initialize load balancing and model (create new session and tokenizer)

                modelInitialization(cfg, param); // Temporarily do not load weights
                param.status = "Initialized";
                System.out.println("Status: Initialized");
                receiver.send("Initialized", 0);
                // First part: client_id
                byte[] clientIdBytes = receiver.recv(0); // Receive first part
                String clientId = new String(clientIdBytes); // Convert to string (assuming client_id is a string)
                Log.d(TAG, "Received client_id: " + clientId);
                byte[] messageBytes = receiver.recv(0); // Receive second part
                msg = new String(messageBytes); // Convert to string
                Log.d(TAG, "msg: " + msg);

                if (msg.equals("WaitingRecovery")) {
                    param.status = "WaitingRecovery";
                    Log.d(TAG, "Status: WaitingRecovery");

                }

            }



            else if (param.status.equals("Finish")) {
                receiver.send("Finish");
                String msg = new String(receiver.recv(0));
                System.out.println(msg);
                System.out.println("Status: Close");
                Log.d(TAG, "Status: Close");

                if (msg.equals("Close")) {  // When receiving msg as "Close", close all sockets
                    for(ArrayList<Map<Integer, Socket>> s: com.allSockets)
                        closeSockets(s);
//                    com.context.close();
                }
                System.out.println("Finish task");
                Log.d(TAG, "Finish task");
                break;
            }
            else if (param.status.equals("Recovery")){
                // Recovery process when an active device is selected to replace a faulty device
                Log.d(TAG, "entering fault recovery state - Recovery (active device)");
                
                // Send recovery request and own IP to the server
                receiver.sendMore("Recovery");         
                receiver.send(Config.local, 0);
                Log.d(TAG, "sent recovery request to server");

                // Receive new IP graph and session index
                Log.d(TAG, "receiving fault recovery information");
                receiveIPGraph(cfg, receiver);
                receiveSessionIndex(receiver);
                
                // Receive dependency graph and other necessary configurations
                receiveDependencyMap(receiver);
                Log.d(TAG, "fault recovery information received");
                
                // Enter recovery processing stage
                param.status = "Recovering";
            }
            else if (param.status.equals("Recovering")){
                Log.d(TAG, "starting fault recovery process - Recovering (active device)");




//
//
//                // Recreate Socket connections
//                // com.updateSockets(param.corePoolSize);
//                // Log.d(TAG, "Socket pool updated according to new communication topology");
//                // If state needs to be synchronized to other devices, implement here
//                syncStateWithNewDevices(com, receiver);
//                // Send recovery ready signal to server
                LoadBalanceInitialization();
                receiver.sendMore("WaitingStart");
                receiver.send(Config.local, 0);
                param.status = "WaitingStart";
                Log.d(TAG, "notified server recovery ready, waiting to start");
                String msg = new String(receiver.recv(0));
                System.out.println(msg);

                if (msg.equals("Start")) {
                    param.status = "Start";
                    Log.d(TAG, "Status: Start");
                    receiver.send("Running");
                    Log.d(TAG, "Status: Running");
                    param.status = "Running";
                    if (param.status == "Running") {
                        // Send RunningStatusEvent to set runningStatus to true in BackgroundService

                        EventBus.getDefault().post(new Events.RunningStatusEvent(true));
                        Log.d(TAG, "Post Events.RunningStatusEvent(true)");
                    }
                }

            }
        }
    }

    /**
     * Receive model file and save to specified path
     * Supports two receiving methods: receive the entire file at once or in chunks
     * 
     * @param path Path to save the file
     * @param receiver Socket for receiving data
     * @param chunked Whether to use chunked transfer
     * @param chunk_size Chunk size (bytes)
     */
    public void receiveModelFile(String path, Socket receiver, boolean chunked, int chunk_size) {
        File file = new File(path);
        if (file.exists() && file.delete()) {   // If file exists, try to delete it
            System.out.println("Deleted the file: " + file.getName());
        } else {
            System.out.println("Failed to delete the file.");
        }

        File parentDir = file.getParentFile();  // Record the directory where the model file is located
        System.out.println("parent dir is: " + parentDir.toString());
        assert parentDir != null;
        if (!parentDir.exists()) {  // If parent directory does not exist, create it
            parentDir.mkdirs();
        }
        System.out.println("Start receiving file");

        file = new File(path);
        if (!chunked) { // When transferring the entire file
            try (FileOutputStream fos = new FileOutputStream(file)) {   // Receive the entire file and then write
                byte[] data = receiver.recv(0);
                fos.write(data);
                System.out.println("Data is written");
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {    // When transferring in chunks
            try (FileOutputStream fos = new FileOutputStream(file)) {
                byte[] chunk;
                int totalSize = 0;
                while ((chunk = receiver.recv()) != null) { // Receive each chunk and write until no more data is received
                    fos.write(chunk);
                    totalSize += chunk.length;
                    if (chunk.length == 0) {
                        break;
                    }
                    System.out.println("Chunk size: " + chunk.length + " Total size: " + totalSize);
                }
                System.out.println("Data is written");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Initialize load balancing system
     * Calls LoadBalance object's reLoadBalance method to recalculate load distribution
     * 
     * @throws Exception Possible exceptions during load balancing
     */
    public void LoadBalanceInitialization() throws Exception {
        Communication.loadBalance.reLoadBalance();  // Load balancing
        Log.d(TAG, "load balance init finished");
    }

    /**
     * Communication handler for preparing model files
     * 1. Receive whether to download the model from the server
     * 2. If needed, receive and save the model file
     * 3. Unzip the model file to the working directory
     * 
     * @param receiver Socket for receiving data
     * @param param Communication parameter object
     * @param modelName Model name
     * @param serverIp Server IP
     * @param role Device role
     */
    public void communicationPrepare(Socket receiver, Communication.Params param, String modelName, String serverIp, int role) {
        param.status = "Prepare";   // Change status to "Prepare"
        boolean chunk = true;       // Default to chunked transfer
        System.out.println("Status: Prepare");
        String skipModelDownload = new String(receiver.recv(0));    // Receive from server whether to skip download phase
        Log.d(TAG, "skipModelDownload: " + skipModelDownload);
        if (skipModelDownload.equals("False")) {
            receiveModelFile(param.modelPath + "/module.zip", receiver, chunk, 100 * 1024 * 1024);  // Receive model file, chunk size 1MB
            System.out.println("Model Received");
        } else {    // Model already exists, skip model download phase
            System.out.println("Model Exists");
        }
        if (skipModelDownload.equals("False")){
            Utils.unzipFile(param.modelPath + "/module.zip");   // Unzip model zip file
        }
    }

    /**
     * Model initialization
     * 1. Load the model and create inference sessions
     * 2. For head and tail nodes, also create a tokenizer
     * 
     * @param cfg Device configuration object
     * @param param Communication parameter object
     * @throws IOException Possible file operation exceptions
     */
    public void modelInitialization(Config cfg, Communication.Params param) throws IOException {
//        for (String i: Communication.sessionIndex) {
////            Communication.sessions.add(createSession(param.modelPath + "/device/module" + i + "/module_" + i + ".onnx"));
//            Communication.sessions.add(createSession(param.modelPath + "/device/module.onnx"));
//            System.out.println("Load module " + i + " successfully");
//            Log.d(TAG, "Load module " + i + " successfully");
//        }
//        File destFile = new File(conText.getFilesDir(), "module.onnx");
//        File sourceFile = new File(Environment.getExternalStorageDirectory(), "device/module.onnx");
//        Files.copy(sourceFile.toPath(), destFile.toPath());


        // Create new session and add to the list
//        String modelPath = destFile.getAbsolutePath();
//        Communication.sessions.add(createSession(modelPath));
        Communication.sessions.add(createSession(param.modelPath + "/device/module.onnx"));
        System.out.println("create session finished");


        // For head and tail nodes, also create a tokenizer based on tokenizer.json
        if (cfg.isHeader() || cfg.isTailer()) {
            Communication.tokenizer = createHuggingFaceTokenizer(param.modelPath + "/device/tokenizer.json");
            // OR SENTENCEPIECE LATER
            System.out.println("Tokenizer created");
            Log.d(TAG, "Tokenizer created");
        }
        System.out.println("model init finished");
    }

    /**
     * Receive IP graph information and build communication graph
     * The IP graph defines the connection topology of each device in distributed inference
     * 
     * @param cfg Device configuration object
     * @param receiver Socket for receiving data
     */
    void receiveIPGraph(Config cfg, Socket receiver){
        byte[] ip_graph = receiver.recv(0);     // Receive IP graph
        String ip_graph_str = new String(ip_graph);
        cfg.buildCommunicationGraph(ip_graph_str);   // Build communication graph based on IP graph, add head and tail node lists for each node
        Log.d(TAG, "Get IP graph: " + ip_graph_str);
        cfg.getDeviceId();  // Get DeviceId
    }

    /**
     * Receive session index information
     * Session index defines how each part of the model is assigned to different devices
     * 
     * @param receiver Socket for receiving data
     */
    void receiveSessionIndex(Socket receiver){
        // Receive Session Index and initial load balance
        String session_indices = receiver.recvStr(0);
        Communication.loadBalance.sessIndices = session_indices.split(";");
        Log.d(TAG, "Get session index: " + session_indices);
    }

    /**
     * Receive task type information
     * Supports two task types: generation and classification
     * 
     * @param param Communication parameter object
     * @param receiver Socket for receiving data
     */
    private void receiveTaskType(Communication.Params param, Socket receiver){
        byte[] task_type = receiver.recv(0);
        param.task_type = new String(task_type);
        Log.d(TAG, "Task: " + param.task_type);
        if (param.task_type.equals("generation")) {
            Log.d(TAG, "Generation with text length: " + param.max_length);
        }else if (param.task_type.equals("classification")){
            Log.d(TAG, "Classification without text length");
        }
    }

    /**
     * Receive thread pool size configuration
     * Thread pool is used to handle multiple inference requests in parallel
     * 
     * @param param Communication parameter object
     * @param receiver Socket for receiving data
     */
    private void receiveThreadPoolSize(Communication.Params param, Socket receiver){
        String pool_size = "";
        try {
            byte[] core_pool_size = receiver.recv(0);
            pool_size = new String(core_pool_size);
            param.corePoolSize = Integer.parseInt(pool_size);   // Convert string parameter to integer
        } catch (NumberFormatException nfe) {
            System.out.println("Core Pool Size is not Integer");
        }
        Log.d(TAG, "Get ThreadPollSize: " + pool_size);
    }

    /**
     * Receive batch size configuration
     * Batch size determines how many input samples are processed at once
     * 
     * @param param Communication parameter object
     * @param receiver Socket for receiving data
     */
    private void receiveBatchSize(Communication.Params param, Socket receiver){
        try {
            byte[] batch = receiver.recv(0);
            param.numSample = Integer.parseInt(new String(batch));
        } catch (NumberFormatException nfe) {
            System.out.println("Num of Batch is not Integer");
        }
        Log.d(TAG, "Num of batch: " + param.numSample);
    }

    /**
     * Receive maximum sequence length configuration
     * For generation tasks, defines the maximum length of generated text
     * For classification tasks, the length is 0
     * 
     * @param param Communication parameter object
     * @param receiver Socket for receiving data
     */
    private void receiveSeqLength(Communication.Params param, Socket receiver) {
        try {
            byte[] max_length = receiver.recv(0);
            param.max_length = Integer.parseInt(new String(max_length));
        } catch (NumberFormatException nfe) {
            Log.d(TAG, "max_length is not Integer");
        }
        Log.d(TAG, "Get Sequence Max Length: " + param.max_length);
    }

    /**
     * Receive dependency map information
     * The dependency map defines the data dependencies between different parts of the model
     * 
     * @param receiver Socket for receiving data
     */
    private void receiveDependencyMap(Socket receiver) {
        String depMap = receiver.recvStr(0);
        Log.d(TAG, "Show Map: " + depMap);
        try {
            Communication.loadBalance.dependencyMap = new JSONObject(depMap);   // Create JSON file from string
        }catch (JSONException e) {
            Log.d(TAG, "Dependency Map JSON EXCEPTION");
        }
        Log.d(TAG, "Get Dependency Map");
    }

    /**
     * Close all Socket connections
     * Called when the task is finished or communication needs to be reset
     * 
     * @param sockets List of Sockets to close
     */
    public void closeSockets(ArrayList<Map<Integer, Socket>> sockets) {
//        releaseSession(Communication.session);
        for (Map<Integer, Socket> sock: sockets) {
            for (Socket socket : sock.values()) {
                socket.close();
            }
        }
    }

    // Native methods for creating ONNX session, releasing session, and creating tokenizer
    public static native long createSession(String inference_model_path);
    public static native long releaseSession(long session);
    public native long createHuggingFaceTokenizer(String tokenizer_path);
    public native long createSentencePieceTokenizer(String tokenizer_path);

    /**
     * Wait for existing inference tasks to complete
     * Monitors inference threads and waits for them to terminate or forcefully terminates after timeout
     * Mainly handles OneStep threads and possibly blocked Socket communication
     * 
     * @param com Communication object
     */
    private void waitForTasksToComplete(Communication com) {
        try {
            Log.d(TAG, "waiting for existing inference tasks to complete...");
            
            // 1. First, try to find and interrupt the actual inference threads (ThreadPoolExecutor in Com.running)
            // instead of shutting down the executor (which is responsible for prepare and the entire communication process)
            Field poolField = null;
            ThreadPoolExecutor inferencePool = null;
            
            try {
                // Use reflection to find the ThreadPoolExecutor instance in Communication
                // This method is more precise, only terminating inference threads without affecting communication threads
                poolField = Communication.class.getDeclaredField("pool");
                if (poolField != null) {
                    poolField.setAccessible(true);
                    inferencePool = (ThreadPoolExecutor) poolField.get(com);
                    
                    if (inferencePool != null && !inferencePool.isShutdown()) {
                        Log.w(TAG, "closing inference thread pool...");
                        // Reject new task submissions
                        inferencePool.shutdown();
                        
                        // Give threads some time to finish current tasks
                        boolean terminated = inferencePool.awaitTermination(3000, TimeUnit.MILLISECONDS);
                        
                        if (!terminated) {
                            // If not terminated in time, force interrupt all threads
                            Log.w(TAG, "inference thread pool not closed in time, forcing interrupt");
                            inferencePool.shutdownNow();
                        }
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "accessing inference thread pool failed: " + e.getMessage());
                // If reflection fails, fallback to direct interruption
            }
            

            
            // 3. If we cannot access the inference thread pool directly, try to close the executor
            // This is the last resort as it may affect communication threads
            if (inferencePool == null && com.executor != null && !com.executor.isTerminated()) {
                Log.w(TAG, "inference thread pool not found, attempting to close executor");
                
                // Set a flag to notify the communication thread to stop running
                // This may require adding a volatile flag in the Communication class
                try {
                    Field runningField = Communication.class.getDeclaredField("isRunning");
                    if (runningField != null) {
                        runningField.setAccessible(true);
                        runningField.set(com, false);
                    }
                } catch (Exception e) {
                    Log.e(TAG, "unable to set running flag: " + e.getMessage());
                }
                
                // Wait a short time for the flag to take effect
                Thread.sleep(1000);
                
                // Avoid shutting down the communication thread, just register a callback to notify the executor we want to terminate
                // Instead of directly calling shutdown
                com.executor.execute(() -> {
                    Log.d(TAG, "requesting termination of running inference tasks");
                });
            }
            
            // Finally, wait a while to ensure all resources are released
            Thread.sleep(2000);
            
            Log.d(TAG, "inference task termination process completed");
        } catch (Exception e) {
            Log.e(TAG, "inference task termination failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Interrupt blocked Socket communication
     * Used to break deadlocks when waiting for faulty nodes during fault recovery
     * 
     * @param com Communication object
     */
    private void interruptBlockedSockets(Communication com) {
        try {
            Log.d(TAG, "releasing blocked Socket communication...");
            
            // Traverse all Socket connections
            if (com.allSockets != null && !com.allSockets.isEmpty()) {
                // Copy queue contents to avoid modifying the original queue structure
                ArrayList<ArrayList<Map<Integer, Socket>>> allSocketsCopy = new ArrayList<>();
                
                // Copy allSockets contents to a temporary list
                com.allSockets.drainTo(allSocketsCopy);
                
                for (ArrayList<Map<Integer, Socket>> socketPair : allSocketsCopy) {
                    for (Map<Integer, Socket> socketMap : socketPair) {
                        for (Socket socket : socketMap.values()) {
                            // Set receive timeout for all Sockets to avoid permanent blocking
                            socket.setReceiveTimeOut(100);
                            // Try to send an "INTERRUPT" message to wake up waiting threads
                            try {
                                socket.send("INTERRUPT", ZMQ.DONTWAIT);
                            } catch (Exception e) {
                                // Ignore send errors, continue processing
                            }
                        }
                    }
                }
                
                // Put the Sockets back into the queue
                for (ArrayList<Map<Integer, Socket>> socketPair : allSocketsCopy) {
                    com.allSockets.put(socketPair);
                }
            }
            
            Log.d(TAG, "Socket communication blocked release completed");
        } catch (Exception e) {
            Log.e(TAG, "Socket blocked release failed: " + e.getMessage());
        }
    }
    
    /**
     * Save intermediate state of inference
     * For head node, save the generated token sequence
     * For worker node, save intermediate computation results
     * 
     * @param com Communication object
     */
    private void saveIntermediateState(Communication com) {
        Log.d(TAG, "saving intermediate state...");
        
        try {
            // The current batch ID has been saved to loadBalance.reSampleId during Recovery
            int currentSampleId = Communication.loadBalance.reSampleId;
            
            // Check if there are intermediate results to save
            if (currentSampleId >= 0) {
                Log.d(TAG, "saving batch " + currentSampleId + " intermediate state");
                
                // For head node, save the current input ID sequence (already generated tokens)
                if (com.cfg.isHeader() && com.InputIds.containsKey(currentSampleId-1)) {
                    // Record the number of generated tokens for verification during recovery
                    Log.d(TAG, "head node: saving input ID sequence, length: " + 
                          com.InputIds.get(currentSampleId-1).size());
                    
                    // InputIds is already a class member variable, it will be accessed automatically during recovery, just ensure it is saved correctly here
                    // If extra backup is needed, implement here
                }
                
                // For intermediate nodes, save intermediate computation results
                if (!com.cfg.isHeader() && !com.cfg.isTailer()) {
                    // Save any intermediate computation state if available
                    if (com.OutputData.containsKey(currentSampleId-1)) {
                        Log.d(TAG, "work node: saving output data");
                        // OutputData is already a class member variable, it will be accessed automatically during recovery
                    }
                }
                
                // If there is residual data to save
                if (com.ResidualDataFromDevice.containsKey(currentSampleId-1) || 
                    com.ResidualDataToDevice.containsKey(currentSampleId-1)) {
                    Log.d(TAG, "saving residual data");
                    // Residual data is already stored in class member variables, it will be accessed automatically during recovery
                }
            } else {
                Log.d(TAG, "no intermediate state to save, current batch ID invalid");
            }
            
            Log.d(TAG, "intermediate state saved");
        } catch (Exception e) {
            Log.e(TAG, "saving intermediate state failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Synchronize state with new devices
     * During fault recovery, synchronize current state to newly joined devices
     * 
     * @param com Communication object
     * @param receiver Socket for communicating with the server
     */
    private void syncStateWithNewDevices(Communication com, Socket receiver) {
        try {
            Log.d(TAG, "starting to synchronize state with new devices...");
            
            // Current batch ID
            int currentSampleId = Communication.loadBalance.reSampleId;
            if (currentSampleId < 0) {
                Log.d(TAG, "no state synchronization needed, no valid recovery batch ID");
                return;
            }
            
            // Get information about newly joined devices from the server (if any)
            // Here we assume the server will send a message containing the new device ID
            if (receiver.hasReceiveMore()) {
                String newDevicesInfo = new String(receiver.recv(0));
                Log.d(TAG, "received new device information: " + newDevicesInfo);
                
                // Parse new device information, format may be "NEW_DEVICE:ip:port"
                if (newDevicesInfo.startsWith("NEW_DEVICE:")) {
                    String[] parts = newDevicesInfo.substring(11).split(":");
                    if (parts.length >= 2) {
                        String newDeviceIp = parts[0];
                        int newDevicePort = Integer.parseInt(parts[1]);
                        
                        // Establish direct communication with the new device
                        Log.d(TAG, "establishing direct communication with new device: " + newDeviceIp + ":" + newDevicePort);
                        
                        // Use existing Socket or create a new Socket to communicate with the new device
                        // Here you may need to find the corresponding Socket according to the new communication topology
                        Socket newDeviceSocket = null;
//                        for (Map.Entry<Integer, String> entry : com.cfg.ipGraph_entry.entrySet()) {
//                            if (entry.getValue().equals(newDeviceIp)) {
//                                // Found new device ID
//                                int newDeviceId = entry.getKey();
//                                Log.d(TAG, "found new device ID: " + newDeviceId);
//
//                                // Get the Socket for communication with the new device
//                                // Here we assume the allSockets structure already has the Socket for the new device
//                                // Actual implementation may need to be adjusted according to the specific code structure
//                                ArrayList<Map<Integer, Socket>> socketPair = com.allSockets.peek();
//                                if (socketPair != null && socketPair.size() >= 2) {
//                                    Map<Integer, Socket> clientSockets = socketPair.get(0);
//                                    if (clientSockets.containsKey(newDeviceId)) {
//                                        newDeviceSocket = clientSockets.get(newDeviceId);
//                                        Log.d(TAG, "found Socket for communication with new device");
//                                    }
//                                }
//                                break;
//                            }
//                        }
//
                        // If the Socket for communication with the new device is found, send state data
                        if (newDeviceSocket != null) {
                            Log.d(TAG, "starting to send state data to new device");
                            
                            // Send handshake signal
                            newDeviceSocket.send("SYNC_STATE");
                            
                            // Wait for the new device to confirm
                            String response = new String(newDeviceSocket.recv(0));
                            if ("READY_FOR_SYNC".equals(response)) {
                                // Send current batch ID
                                newDeviceSocket.sendMore("SAMPLE_ID");
                                newDeviceSocket.send(String.valueOf(currentSampleId));
                                
                                // Send different state data according to device role
                                if (com.cfg.isHeader()) {
                                    // Head node sends the generated token sequence
                                    if (com.InputIds.containsKey(currentSampleId-1)) {
                                        ArrayList<Integer> tokens = com.InputIds.get(currentSampleId-1);
                                        // Convert ArrayList<Integer> to a transferable format
                                        // For simplicity, convert to string here, should use binary format in practice
                                        StringBuilder sb = new StringBuilder();
                                        for (int token : tokens) {
                                            sb.append(token).append(",");
                                        }
                                        newDeviceSocket.sendMore("INPUT_IDS");
                                        newDeviceSocket.send(sb.toString());
                                    }
                                }
                                
                                // Wait for the new device to confirm receipt
                                response = new String(newDeviceSocket.recv(0));
                                if ("SYNC_COMPLETED".equals(response)) {
                                    Log.d(TAG, "state synchronization completed successfully");
                                } else {
                                    Log.e(TAG, "state synchronization exception: " + response);
                                }
                            } else {
                                Log.e(TAG, "new device not ready to receive state: " + response);
                            }
                        } else {
                            Log.e(TAG, "Socket for communication with new device not found");
                        }
                    }
                }
            } else {
                Log.d(TAG, "server did not provide new device information, skipping state synchronization");
            }
            
            Log.d(TAG, "state synchronization process completed");
        } catch (Exception e) {
            Log.e(TAG, "state synchronization failed: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Wait for state synchronization
     * When an active device becomes a worker device, wait for state synchronization information from other devices
     * 
     * @param com Communication object
     */
    private void waitForStateSynchronization(Communication com) {
        Log.d(TAG, "waiting for state synchronization...");
        
        try {
            // Receive state synchronization based on the existing communication structure
            // Find all possible Socket connections
            ArrayList<Map<Integer, Socket>> socketPair = com.allSockets.peek();
            if (socketPair == null || socketPair.size() < 2) {
                Log.e(TAG, "no valid Socket connection found, skipping state synchronization");
                return;
            }
            
            // Get the receiving Socket map
            Map<Integer, Socket> serverSockets = socketPair.get(1);
            
            // Try to receive state data from each connection
            boolean syncReceived = false;
            long startTime = System.currentTimeMillis();
            long timeout = 10000; // 10 seconds timeout
            
            while (!syncReceived && System.currentTimeMillis() - startTime < timeout) {
                // Check all receiving Sockets
                for (Socket socket : serverSockets.values()) {
                    // Non-blocking check for messages
                    byte[] message = socket.recv(ZMQ.DONTWAIT);
                    if (message != null) {
                        String command = new String(message);
                        Log.d(TAG, "received message: " + command);
                        
                        if ("SYNC_STATE".equals(command)) {
                            // Received sync request, send ready response
                            socket.send("READY_FOR_SYNC");
                            Log.d(TAG, "sent ready for synchronization response");
                            
                            // Receive sync data
                            String dataType = new String(socket.recv(0));
                            if ("SAMPLE_ID".equals(dataType)) {
                                // Receive current batch ID
                                String sampleIdStr = new String(socket.recv(0));
                                int sampleId = Integer.parseInt(sampleIdStr);
                                Log.d(TAG, "received batch ID: " + sampleId);
                                
                                // Continue to receive other data
                                dataType = new String(socket.recv(0));
                                if ("INPUT_IDS".equals(dataType)) {
                                    // Receive input ID sequence (important for successor nodes of the head node)
                                    String tokensStr = new String(socket.recv(0));
                                    String[] tokenParts = tokensStr.split(",");
                                    ArrayList<Integer> tokens = new ArrayList<>();
                                    for (String part : tokenParts) {
                                        if (!part.isEmpty()) {
                                            tokens.add(Integer.parseInt(part));
                                        }
                                    }
                                    
                                    // Save the received tokens to the current device's state
                                    if (tokens.size() > 0) {
                                        // Save tokens here if needed
                                        // For example, if this device becomes the new head node, save tokens
                                        if (com.cfg.isHeader() && !com.InputIds.containsKey(sampleId-1)) {
                                            com.InputIds.put(sampleId-1, tokens);
                                            Log.d(TAG, "saving received token sequence, length: " + tokens.size());
                                        }
                                    }
                                }
                                
                                // There may be other data types to receive...
                                
                                // Send sync completion confirmation
                                socket.send("SYNC_COMPLETED");
                                Log.d(TAG, "state synchronization completed successfully");
                                
                                syncReceived = true;
                                break;
                            }
                        }
                    }
                }
                
                // If sync not yet received, sleep a bit and check again
                if (!syncReceived) {
                    Thread.sleep(100);
                }
            }
            
            if (!syncReceived) {
                Log.w(TAG, "waiting for state synchronization timeout, attempting to perform state-less recovery");
            }
        } catch (Exception e) {
            Log.e(TAG, "state synchronization reception failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}