package com.example.SecureConnection;
import static com.example.distribute_ui.service.BackgroundService.TAG;

import android.content.Context;
import android.util.Log;
import android.content.Intent;

import org.json.JSONException;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.*;

import org.zeromq.ZMQ;
import org.zeromq.ZMQ.Socket;
import org.zeromq.ZContext;
import org.zeromq.SocketType;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;

import org.json.JSONObject;
import org.zeromq.ZMQException;

import com.example.SecureConnection.Utils.LBPause;
import com.example.distribute_ui.DataRepository;
import org.greenrobot.eventbus.EventBus;
import com.example.distribute_ui.Events;
import java.util.concurrent.ConcurrentHashMap;
import org.greenrobot.eventbus.Subscribe;
import org.greenrobot.eventbus.ThreadMode;

public class Communication {

    public class Params {
        public String modelPath;
        public String sourcePath;
        public int max_length = 256;
        public String task_type;
        public boolean skip_special_token;
        public int corePoolSize;
        public String[] classes;
        public volatile String status = "Ready";
        public int numSample;
    }

    public static volatile Long tokenizer;
    public static Long session;
    public static ArrayList<Long> sessions;
    public static String[] sessionIndex;
    public static LoadBalance loadBalance;
    public static LBPause LB_Pause;
    public ExecutorService executor;
    // Thread control flag for interrupting inference during fault recovery
    public volatile boolean isRunning = true;
    // Reference to the thread pool for inference tasks, used for fault recovery
    ThreadPoolExecutor pool;

    public Params param;
    public Config cfg;
    public String[] InputString;
    public volatile Map<Integer, ArrayList<Integer>> InputIds;
    public volatile Map<Integer, byte[]> InputData;
    public volatile Map<Integer, byte[]> OutputData;
    public volatile Map<Integer, Map<String, ArrayList<byte[]>>> ResidualDataFromDevice;
    public volatile Map<Integer, Map<String, ArrayList<byte[]>>> ResidualDataToDevice;
    public volatile Map<Integer, byte[]> logits;
    public int sampleId; // Current batch

    public boolean valid;
    public Map<String, Socket> sockets;
    public ZContext context;
    public Socket rootSocket;
    public Socket commuSocket;
    public Client beClient;
    public Server beServer;
    public LinkedBlockingQueue<ArrayList<Map<Integer, Socket>>> allSockets;
    public double[] timeUsage; // Preparation time
    public Set<Integer> receiveDeviceIndex;
    public Set<Integer> sendDeviceIndex;
    public TreeMap<String, ArrayList<JSONObject>> sendIndex;
    public TreeMap<String, ArrayList<JSONObject>> receiveIndex;
    public TreeMap<Integer, ArrayList<String>> sendD2D;
    public TreeMap<Integer, ArrayList<String>> receiveD2D;
    public TreeMap<String, Integer> module_on_devices;

    private Context conText;
    private String modelName;
    private int role;

    // 1. Add a static thread-safe fault time table in the Communication class:
    private static final java.util.concurrent.ConcurrentHashMap<String, Long> faultStartTimes = new java.util.concurrent.ConcurrentHashMap<>();

    // 1. Add to Communication class members:
    private boolean lastIsFaultMode = false;

    private void restartBackgroundServiceWithLastParams() {
        try {
            Log.d(TAG, "restartBackgroundServiceWithLastParams...");

            // Use previous parameters
            // Stop then start
            // 1. Stop all com-related threads and processes
            Log.d(TAG, "Stopping all com threads and processes...");

            // Stop heartbeat thread
            if (heartbeatThread != null && heartbeatThread.isAlive()) {
                isHeartbeatRunning = false;
                heartbeatThread.interrupt();
                heartbeatThread = null;
                Log.d(TAG, "Heartbeat thread stopped");
            }

            // Stop executor thread pool (includes com.runPrepareThread, etc.)
            if (executor != null && !executor.isShutdown()) {
                executor.shutdownNow();
                try {
                    if (!executor.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)) {
                        Log.w(TAG, "Executor did not terminate in time");
                    }
                } catch (InterruptedException e) {
                    Log.w(TAG, "Executor termination interrupted");
                }
                Log.d(TAG, "Executor thread pool stopped");
            }

            // Stop pool thread pool (includes multiSteps tasks, etc.)
            if (pool != null && !pool.isShutdown()) {
                pool.shutdownNow();
                try {
                    if (!pool.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)) {
                        Log.w(TAG, "Pool did not terminate in time");
                    }
                } catch (InterruptedException e) {
                    Log.w(TAG, "Pool termination interrupted");
                }
                Log.d(TAG, "Pool thread pool stopped");
            }

            // Stop ZMQ-related connections
            if (rootSocket != null) {
                rootSocket.close();
                rootSocket = null;
                Log.d(TAG, "Root socket closed");
            }

            // Reset com status
            if (param != null) {
                param.status = "Stopped";
            }

            // 2. Stop BackgroundService
            Context appContext = this.conText != null ? this.conText.getApplicationContext() : null;
            if (appContext == null) {
                Log.e(TAG, "No valid application context for restarting service");
                return;
            }

            Intent stopIntent = new Intent(appContext, com.example.distribute_ui.service.BackgroundService.class);
            appContext.stopService(stopIntent);
            Log.d(TAG, "BackgroundService stopped");

            // Wait for the service to completely stop
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                Log.w(TAG, "Sleep interrupted while waiting for service to stop");
            }

            // 3. Restart service with saved parameters
            Log.d(TAG, "Restarting BackgroundService with saved parameters...");

            // Use saved static parameters from BackgroundService
            Intent startIntent = new Intent(appContext, com.example.distribute_ui.service.BackgroundService.class);

            // Retrieve parameters from static variables
            if (com.example.distribute_ui.service.BackgroundService.lastStartIntent != null) {
                // Use the saved Intent directly
                startIntent = com.example.distribute_ui.service.BackgroundService.lastStartIntent;
                Log.d(TAG, "Using saved Intent for restart");
            } else {
                // Fallback: Retrieve from SharedPreferences
                android.content.SharedPreferences prefs = appContext.getSharedPreferences("app_prefs", Context.MODE_PRIVATE);
                int role = prefs.getInt("role", 0);
                String model = prefs.getString("model", "");
                String ip = prefs.getString("ip", "");

                startIntent.putExtra("role", role);
                startIntent.putExtra("model", model);
                startIntent.putExtra("ip", ip);
                Log.d(TAG, "Using SharedPreferences params for restart: role=" + role + ", model=" + model + ", ip=" + ip);
            }

            appContext.startService(startIntent);
            Log.d(TAG, "BackgroundService restarted successfully");

        } catch (Exception e) {
            Log.e(TAG, "Failed to restart BackgroundService: " + e.getMessage());
        }
    }

    public Communication(Config cfg, Context conText, String modelName, int role) {
        Communication.sessions = new ArrayList<>();
        Communication.LB_Pause = new LBPause();
        this.conText = conText;
        this.cfg = cfg;
        this.modelName = modelName;
        this.role = role;

        param = new Params();
        param.skip_special_token = false;

        InputIds = new HashMap<>();
        InputData = new ConcurrentHashMap<>();
        OutputData = new ConcurrentHashMap<>();
        ResidualDataFromDevice = new ConcurrentHashMap<>();
        ResidualDataToDevice = new ConcurrentHashMap<>();
        logits = new ConcurrentHashMap<>();

        sampleId = 0;
        sockets = new HashMap<>();
        context = new ZContext();
        beClient = new Client();
        beServer = new Server();

        allSockets = new LinkedBlockingQueue<>();
        sendIndex = new TreeMap<>();
        receiveIndex = new TreeMap<>();

        sendDeviceIndex = new TreeSet<>();
        receiveDeviceIndex = new TreeSet<>();
        timeUsage = new double[2];

        module_on_devices = new TreeMap<>();
    }

    public String sendIPToServer(String role, String modelRequest) throws JSONException {
        rootSocket = beClient.establish_connection(context, SocketType.DEALER, cfg.rootPort, cfg.root); // Establish connection with server
        Log.d(TAG, "socket establish connection");
        // Write own IP and role into JSON file; if head node, include model name
        String currentIP = Config.local;
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("ip", currentIP);
        jsonObject.put("role", role);
        if ("header".equals(role)) {
            jsonObject.put("model", modelRequest);
        }
        // Send own IP and other info to server
        rootSocket.sendMore("RegisterIP");
        rootSocket.send(jsonObject.toString());
        Log.d(TAG, "IP message sent");
        String msg_ = new String(rootSocket.recv(0));
        Log.d(TAG, "msg: " + msg_);

        // Start heartbeat detection thread
        startHeartbeatDetection();

        return msg_;
    }

    private volatile boolean isHeartbeatRunning = true;
    private Thread heartbeatThread = null;

    /**
     * Start heartbeat detection thread to periodically send heartbeat messages to the server
     * 
     * Heartbeat detection is a key mechanism for fault monitoring, achieved by periodically sending heartbeat packets:
     * 1. Prove to the server that the device is still online
     * 2. Receive system status information from the server
     * 3. Detect and respond to SYSTEM_FAILURE signals, triggering the fault recovery process
     * 
     * Heartbeat frequency: Sent every 10 seconds, below the server timeout threshold
     */
    public void startHeartbeatDetection() {
        if (heartbeatThread != null && heartbeatThread.isAlive()) {
            return; // Avoid duplicate start
        }

        heartbeatThread = new Thread(() -> {
            try {
                // Use the same socket connection, distinguished by different actions
                // "RegisterIP" for registration, "HEARTBEAT" for heartbeat detection
                Log.d(TAG, "Heartbeat thread started, using existing socket connection");

                // 1. Define outside the heartbeat detection thread loop
                boolean lastIsFaultMode = false;

                while (isHeartbeatRunning) {
                    try {
                        // Check if the app is in the background
                        Events.GetBackgroundStatusEvent statusEvent = new Events.GetBackgroundStatusEvent();
                        EventBus.getDefault().post(statusEvent);
                        boolean isInBackground = statusEvent.isInBackground();
                        // Add: Get screen-off status
                        boolean isScreenOff = false;
                        try {
                            java.lang.reflect.Constructor<?> cons = Class.forName("com.example.distribute_ui.service.BackgroundService$ScreenOffEvent").getDeclaredConstructor(boolean.class);
                            cons.setAccessible(true);
                            Object screenEvent = cons.newInstance(false); // Placeholder instance
                            // Assume global static variable BackgroundService.isScreenOff exists
                            isScreenOff = com.example.distribute_ui.service.BackgroundService.isScreenOff;
                        } catch (Exception e) {
                            // Default to false if unable to retrieve
                        }
                        if (isInBackground && !isScreenOff) {
                            // Fault mode: Background + screen on
                            rootSocket.sendMore("HEARTBEAT_InBackground_ScreenOn");
                            rootSocket.send("");
                            Log.d(TAG, "Heartbeat sent with action FAULT_HEARTBEAT (background+screenOn)");
                        } else {
                            // Normal mode
                            rootSocket.sendMore("HEARTBEAT");
                            rootSocket.send("");
                            Log.d(TAG, "Heartbeat sent with action HEARTBEAT");
                        }

                        // Receive heartbeat response
                        String response = new String(rootSocket.recv(0));

                        // Check for additional system status information
                        if (rootSocket.hasReceiveMore()) {
                            String systemStatus = new String(rootSocket.recv(0));

                            // If a system failure is detected, trigger fault recovery process
                            if ("SYSTEM_FAILURE".equals(systemStatus) && 
                                !"Recovery".equals(param.status) && 
                                !"Recovering".equals(param.status)) {

                                Log.w(TAG, "System failure detected! Preparing to enter fault recovery process");

                                // Set status to Recovery; Client.communicationOpenClose will detect this change and handle it
                                synchronized (param) {
                                    param.status = "Recovery";
                                }

                                Log.w(TAG, "Status set to Recovery, communication thread will handle recovery");
                            } else if ("SYSTEM_InBackground_ScreenOn".equals(systemStatus) &&
                                    !"Recovery".equals(param.status) &&
                                    !"Recovering".equals(param.status)) {

                                Log.d(TAG, "Heartbeat received with action FAULT_HEARTBEAT (background+screenOn)");

                                // Set status to WaitRestarting; Client.communicationOpenClose will detect this change and handle it
                                synchronized (param) {
                                    param.status = "WaitRestarting";
                                }

                                Log.w(TAG, "Status set to Restarting, proceeding with restart");
                            }
                        }

                        // Heartbeat interval, recommended to be less than half the server timeout threshold
                        Thread.sleep(1000); // Send heartbeat every 1 second

                        // Check if system status transitions from recovery to running
                        boolean isFaultMode = isInBackground && !isScreenOff;
                        if (Communication.this.lastIsFaultMode && !isFaultMode) {
                            Log.d(TAG, "Switch from FAULT to NORMAL, restarting BackgroundService...");
                            synchronized (param) {
                                param.status = "Restarting";
                            }
                            Communication.this.restartBackgroundServiceWithLastParams();
                        }
                        Communication.this.lastIsFaultMode = isFaultMode;
                    } catch (Exception e) {
                        Log.e(TAG, "Heartbeat loop error: " + e.getMessage());
                        // Heartbeat send failed, retry after a short wait
                        try {
                            Thread.sleep(500);
                        } catch (InterruptedException ie) {
                            // Ignore interrupt exception
                        }
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "Heartbeat thread fatal error: " + e.getMessage());
            }
        });

        heartbeatThread.setDaemon(true); // Set as daemon thread, does not prevent JVM exit
        heartbeatThread.start();
        Log.d(TAG, "Heartbeat detection thread started");
    }

    /**
     * Stop heartbeat detection thread
     * 
     * Called when the device is shutting down or task is completed to ensure proper termination of the heartbeat thread
     */
    public void stopHeartbeatDetection() {
        isHeartbeatRunning = false;
        if (heartbeatThread != null) {
            heartbeatThread.interrupt();
            heartbeatThread = null;
        }
        Log.d(TAG, "Heartbeat detection stopped");
    }

    public void handleSystemFailure() {
        // Avoid duplicate handling if already in failure or recovery state
        Log.d(TAG, "Entering system failure handling procedure");
        if ("Failure".equals(param.status) || "Recovering".equals(param.status)) {
            Log.d(TAG, "System already in failure/recovery state, ignoring duplicate recovery trigger");
            return;
        }
        // Fault occurrence time
        long faultTime = System.currentTimeMillis();
        Log.w(TAG, "System failure handling initiated");
        param.status = "Recovering";
        // Iterate through all active sample_ids and put them into faultStartTimes
        for (Integer sid : activeAggregators.keySet()) {
            String affectedQueryId = String.valueOf(sid);
            faultStartTimes.put(affectedQueryId, faultTime);
        }
        try {
            Communication.loadBalance.setReSampleId(sampleId);

            // Save current socket queue size for verification after recovery
            int socketPairsCount = allSockets.size();
            Log.d(TAG, "Current socket pair count: " + socketPairsCount);
            cleanExistingConnections();

            // Update device mapping and sessions using LoadBalance methods
            loadBalance.ModifySession();
            loadBalance.reLoadBalance();

            // Re-create socket connections, creating all needed sockets directly
            Log.w(TAG, "Re-creating socket connections");
            updateSockets(param.corePoolSize);

            // Check if socket count is sufficient; if not, create the missing number
            if (allSockets.size() < socketPairsCount) {
                int missingPairs = socketPairsCount - allSockets.size();
                Log.w(TAG, "Insufficient socket connections, need to add " + missingPairs + " pairs");
                // Create the remaining required socket pairs
                updateSockets(missingPairs);
            }

            Log.d(TAG, "Device connections re-established, socket pairs count: " + allSockets.size());

            // Reset reload flag
            Communication.loadBalance.setReSampleId(-1);
            Communication.LB_Pause.setConditionFalse();

            // Recovery complete
            param.status = "WaitingStart";
            Log.w(TAG, "Recovery completed, ready to restart, device status: " + param.status);

            // Note: We no longer call resumeInference() since we do not interrupt inference threads
            // Current inference threads will use the new socket connections in the next oneStep iteration

        } catch (Exception e) {
            Log.e(TAG, "Error during system failure recovery: " + e.getMessage());
            param.status = "Failure"; // Recovery failed, mark as failure state
        }
    }

    /**
     * Clean up existing socket connections
     * 
     * During fault recovery, close and re-create all communication sockets
     * This method safely closes all existing connections to ensure proper resource release
     */
    void cleanExistingConnections() {
        try {
            // Clean up existing connections
            while (!allSockets.isEmpty()) {
                ArrayList<Map<Integer, Socket>> socketPair = allSockets.take();
                for (Map<Integer, Socket> socketMap : socketPair) {
                    for (Socket socket : socketMap.values()) {
                        socket.setLinger(0); // Ensure quick port release
                        socket.close();
                    }
                }
            }

            Log.d(TAG, "Existing connections cleaned up");
        } catch (Exception e) {
            Log.e(TAG, "Error cleaning existing connections: " + e.getMessage());
        }
    }

    // Run the thread responsible for the Prepare function
    public void runPrepareThread(String param) {
        executor = Executors.newFixedThreadPool(2); // Create a thread pool with a maximum of 2 concurrent threads
        executor.submit(() -> {  // Submit task to the thread pool
            try {
                this.prepare(param); // Run the prepare method of the Communication class
            } catch (Exception e) { // If prepare() throws an exception, throw a runtime exception
                Log.e(TAG, "Error: " + e.getMessage());
                throw new RuntimeException(e);
            }
        });
    }

    public void runRunningThread(int corePoolSize, int maximumPoolSize, int keepAliveTime, ArrayList<String> input_data) {
        executor.submit(() -> {
            try {
                Log.w(TAG, "runPrepareThread communication starts running");
                this.running(corePoolSize, maximumPoolSize, keepAliveTime, input_data);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    public void shutDownPrepare() {
        executor.shutdown();
    }

    public void prepare(String param) throws Exception {
        long startTime = System.nanoTime(); // Record start time
        // Communicate with Root Server
        Log.d(TAG, "root IP: " + cfg.root + ", root port: " + cfg.rootPort);

        if (param.equals("active")) {
            Config.port = 10000;
            Log.e(TAG, "Port: " + Config.port);

            commuSocket = beClient.establish_connection(context, SocketType.DEALER, 23457, cfg.root); // Establish connection with server
            beClient.communicationOpenCloseActive(cfg, this, commuSocket, this.modelName, this.cfg.root, this.role);
        } else if (param.equals("working")) {
            commuSocket = beClient.establish_connection(context, SocketType.DEALER, 34567, cfg.root); // Establish connection with server
            beClient.communicationOpenClose(cfg, this, commuSocket, this.modelName, this.cfg.root, this.role);
        }

        long prepareTime = System.nanoTime(); // Record completion time
        System.out.println("Prepare Time in seconds: " + (prepareTime - startTime) / 1000000000.0);
        timeUsage[0] = (prepareTime - startTime) / 1000000000.0; // Record preparation time (in seconds)
    }

    public void cleanUpBuffer(int id) {
        InputData.remove(id);
        OutputData.remove(id);
    }

    // Retrieve residual index data related to the specified module index from sendIndex and return it.
    // The returned data is a 2D array int[][], where each subarray represents a residual index list
    public int[][] getResIndices(int module_idx) throws JSONException {
        if (!sendIndex.containsKey(sessionIndex[module_idx]) || sendIndex.get(sessionIndex[module_idx]).size() <= 1)
            return new int[0][];

        JSONObject resIndex = sendIndex.get(sessionIndex[module_idx]).get(1);
        resIndex.keys();
        int[][] ResIndex = new int[resIndex.length()][];
        Iterator<String> keys = resIndex.keys();
        // Need to sort before computing
        List<String> tmp = new ArrayList<>();
        while (keys.hasNext())
            tmp.add(keys.next());
        Collections.sort(tmp);

        for (int i = 0; i < tmp.size(); i++) {
            ResIndex[i] = Utils.JsonArray2IntArray(resIndex.getJSONArray(tmp.get(i)));
        }
        return ResIndex;
    }

    public ArrayList<byte[]> mergeResFromAndToDevice(int id, String module_idx) {
        ArrayList<byte[]> data1 = new ArrayList<>();
        ArrayList<byte[]> data2 = new ArrayList<>();
        if (ResidualDataFromDevice.containsKey(id) && ResidualDataFromDevice.get(id).containsKey(module_idx))
            data1 = ResidualDataFromDevice.get(id).get(module_idx); // Comes from previous module on a different device
        if (ResidualDataToDevice.containsKey(id) && ResidualDataToDevice.get(id).containsKey(module_idx))
            data2 = ResidualDataToDevice.get(id).get(module_idx); // Comes from previous module on the same device
        data1.addAll(data2); // If sorted, from previous device first, then from local device next
        System.out.println("Merge Receive Res Size: " + data1.size());
        return data1;
    }

    public void convertOutput(int id, int module_idx, Object[] result) {
        OutputData.put(id, (byte[]) result[0]); // Store inference result of a model part in the output map

        JSONObject resIndex = null;
        if (sendIndex.get(sessionIndex[module_idx]).size() > 1) // Get residual data index
            resIndex = sendIndex.get(sessionIndex[module_idx]).get(1);

        // Check if residual data exists
        if (result.length > 1 && resIndex != null) {
            Iterator<String> keys = resIndex.keys(); // Get all keys
            int i = 0;
            while (keys.hasNext()) { // Iterate through each key, store residual data in ResidualDataToDevice
                String k = keys.next();
                if (!ResidualDataToDevice.get(id).containsKey(k)) // Create if key does not exist
                    ResidualDataToDevice.get(id).put(k, new ArrayList<>());
                byte[][] val = (byte[][]) result[1];
                ResidualDataToDevice.get(id).get(k).add(val[i]); // Careful about the order added
                i++;
            }
            if (ResidualDataToDevice.get(id).size() > 0) // Print info when ResidualDataToDevice has data
                for (Map.Entry<String, ArrayList<byte[]>> e : ResidualDataToDevice.get(id).entrySet())
                    if (e.getValue().size() > 1)
                        System.out.println("To Module " + e.getKey() + " receive byte: " + e.getValue().size());
        }
    }

    public void inferenceProcedure(int id) throws JSONException {
        // id is SampleId. First, check if input data for the id exists
        if (((InputData.containsKey(id) && InputData.get(id) != null)) || (this.InputIds.get(id)) != null) {
            System.out.println("Start inference, session size: " + sessions.size());
            if (sessions.size() != 0) { // Ensure block count > 0
                byte[] res;
                Object[] result = null;
                // Add an entry (batch -> map [string -> list of byte array pointers]) to store inter-device data
                ResidualDataToDevice.put(id, new TreeMap<>());
                if (cfg.isHeader()) { // For head node
                    System.out.println("Inference on Master");
                    for (int i = 0; i < sessions.size(); i++) {
                        // Get send indices for the session from sendIndex and convert to integer array for sending to other devices
                        int[] to_send_seq_indices = Utils.JsonArray2IntArray(sendIndex.get(sessionIndex[i]).get(0).getJSONArray(String.valueOf(Integer.parseInt(sessionIndex[i]) + 1)));
                        System.out.println("to_send_seq_indices: " + Arrays.toString(to_send_seq_indices));
                        if (i == 0) {
                            ArrayList<Integer> inputIds = InputIds.get(id);
                            int[] currentToken = new int[]{inputIds.get(inputIds.size() - 1)};
                            result = ((Object[]) runInferenceMasterResidual(sessions.get(i), currentToken, to_send_seq_indices, getResIndices(i)));
                            System.out.println("current session: " + i + ", execute runInferenceMasterResidual");
                        } else {
                            System.out.println("current session: " + i + ", execute runInferenceWorkerResidual");
                            result = ((Object[]) runInferenceWorkerResidual(sessions.get(i), OutputData.get(id), mergeResFromAndToDevice(id, sessionIndex[i]), to_send_seq_indices, getResIndices(i)));
                        }
                        convertOutput(id, i, result); // Store result in OutputData and residual Data
                    }
                } else if (cfg.isTailer()) { // For tail node
                    System.out.println("Inference on Tail");
                    for (int i = 0; i < sessions.size(); i++) {
                        if (i == sessions.size() - 1) {
                            if (i == 0) {
                                System.out.println("current session: " + i + ", execute runInferenceWorkerResidualLastGeneration");
                                res = runInferenceWorkerResidualLastGeneration(sessions.get(i),
                                        InputData.get(id),
                                        mergeResFromAndToDevice(id, sessionIndex[i]),
                                        cfg.k,
                                        cfg.initial_temp);
                            } else {
                                System.out.println("current session: " + i + ", execute runInferenceWorkerResidualLastGeneration");
                                res = runInferenceWorkerResidualLastGeneration(sessions.get(i),
                                        OutputData.get(id),
                                        mergeResFromAndToDevice(id, sessionIndex[i]),
                                        cfg.k,
                                        cfg.initial_temp);
                            }
                            OutputData.put(id, res);
                            break;
                        } else if (i == 0) {
                            System.out.println("current session: " + i + ", execute runInferenceWorkerResidual");
                            result = ((Object[]) runInferenceWorkerResidual(sessions.get(i), InputData.get(id), mergeResFromAndToDevice(id, sessionIndex[i]), Utils.JsonArray2IntArray(sendIndex.get(sessionIndex[i]).get(0).getJSONArray(sessionIndex[i + 1])), getResIndices(i)));
                        } else {
                            System.out.println("current session: " + i + ", execute runInferenceWorkerResidual");
                            result = ((Object[]) runInferenceWorkerResidual(sessions.get(i), OutputData.get(id), mergeResFromAndToDevice(id, sessionIndex[i]), Utils.JsonArray2IntArray(sendIndex.get(sessionIndex[i]).get(0).getJSONArray(sessionIndex[i + 1])), getResIndices(i)));
                        }
                        convertOutput(id, i, result);
                    }
                } else {
                    System.out.println("Inference on Worker");
                    for (int i = 0; i < sessions.size(); i++) {
                        int[] to_send_seq_indices = Utils.JsonArray2IntArray(sendIndex.get(sessionIndex[i]).get(0).getJSONArray(String.valueOf(Integer.parseInt(sessionIndex[i]) + 1)));
                        System.out.println("to_send_seq_indices: " + Arrays.toString(to_send_seq_indices));
                        if (i == 0) {
                            System.out.println("current session: " + i + ", execute runInferenceWorkerResidual");
                            result = ((Object[]) runInferenceWorkerResidual(sessions.get(i), InputData.get(id), mergeResFromAndToDevice(id, sessionIndex[i]), to_send_seq_indices, getResIndices(i)));
                        } else {
                            System.out.println("current session: " + i + ", execute runInferenceWorkerResidual");
                            result = ((Object[]) runInferenceWorkerResidual(sessions.get(i), OutputData.get(id), mergeResFromAndToDevice(id, sessionIndex[i]), to_send_seq_indices, getResIndices(i)));
                        }
                        convertOutput(id, i, result);
                    }
                }
                System.out.println("Inference completed");
            }
        } else { // Data missing
            System.out.println("Data missing");
        }
    }

    // corePoolSize=2, maximumPoolSize=2, keepAliveTime=500
    public void running(int corePoolSize, int maximumPoolSize, int keepAliveTime, ArrayList<String> input_data) throws Exception {
        // Reset running flag
        isRunning = true;
        Log.w(TAG, "Starting running");
        while (!param.status.equals("Running")) {
            Thread.sleep(1000);
        }

        if (!cfg.isHeader()) // Clear input_data for non-head nodes
            input_data = null;

        if (param.corePoolSize > 0) { // param.corePoolSize == 1
            corePoolSize = param.corePoolSize;
            maximumPoolSize = param.corePoolSize;
        }
        System.out.println("corePoolSize: " + corePoolSize + ", maximumPoolSize: " + maximumPoolSize);

        Semaphore latch = new Semaphore(param.corePoolSize); // Set semaphore to corePoolSize

        Lock socketLock = new ReentrantLock();

        // Thread-safe queue for tasks waiting to be executed in the thread pool
        LinkedBlockingQueue<Runnable> waitingQueue = new LinkedBlockingQueue<Runnable>();

        // Thread pool for managing thread and task execution
        pool = new ThreadPoolExecutor(
                corePoolSize, // Minimum number of threads to keep in the pool
                maximumPoolSize, // Maximum number of threads the pool can create
                keepAliveTime, // Maximum idle time
                TimeUnit.MILLISECONDS,
                waitingQueue, // Waiting queue
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.AbortPolicy());

        // Create send and receive socket connections for the device, with special handling for head and tail nodes
        Log.w(TAG, "Creating socket connections in running thread");
        updateSockets(corePoolSize);

        System.out.println("Load Balance On Running");

        long startTime = System.nanoTime();

        while (isRunning) { // Check isRunning flag to support fault recovery interruption
            if (sampleId >= param.numSample) { // Exit after processing all batches
                break;
            } else {
                if (cfg.isHeader) { // For head node
                    while (sampleId >= input_data.size()) { // Wait if current batch has no input yet
                        // Check for interruption request
                        if (!isRunning) {
                            Log.d(TAG, "Received termination request while waiting for input, interrupting inference");
                            break;
                        }
                        Log.d(TAG, "Waiting for input");
                        Thread.sleep(1000);
                    }
                    // Check termination flag again
                    if (!isRunning) break;
                    // Pass current string and tokenizer pointer to encode string to integer array
                    int[] data = encodeString(input_data.get(sampleId), tokenizer);
                    System.out.println("encode array: " + Arrays.toString(data));
                    this.InputIds.put(sampleId, Utils.convertIntegerArrayToArrayList(data)); // Record encoding result in InputIds map
                    // Add: Sync InputString
                    if (InputString == null || InputString.length != input_data.size()) {
                        InputString = new String[input_data.size()];
                    }
                    InputString[sampleId] = input_data.get(sampleId);
                }
            }

            // Check termination flag again
            if (!isRunning) break;

            if (pool.getActiveCount() + waitingQueue.size() < corePoolSize) { // Check if thread pool can add new tasks
                if ((!LB_Pause.condition && loadBalance.reSampleId == -1) || sampleId < loadBalance.reSampleId) { // Conditions for submitting new tasks
                    latch.acquire(); // Acquire semaphore
                    System.out.println("Submitting new task, actually executing multiSteps.run()");
                    pool.execute(new multiSteps(sampleId, latch)); // Submit new task, actually executes multiSteps.run()
                    sampleId += 1; // Increment batch count
                } else if (LB_Pause.condition) {
                    // Logic to pause submitting new tasks
                    System.out.println("resampleId " + loadBalance.reSampleId);
                    System.out.println("wait the Process to Finish");
                    System.out.println("Active Thread Count: " + (pool.getActiveCount() + waitingQueue.size()));
                    if ((pool.getActiveCount() + waitingQueue.size()) == 0 && (loadBalance.reSampleId != -1 && sampleId >= loadBalance.reSampleId)) {
                        System.out.println("===================== Load Balance =====================");
                    }
                }
            }

            // Check for interruption request
            if (!isRunning) {
                Log.d(TAG, "Received thread termination request, interrupting inference loop");
                break;
            }

            // Avoid excessive CPU usage
            Thread.sleep(10);
        }

        // If loop exits due to termination signal, clean up resources
        if (!isRunning) {
            Log.d(TAG, "Inference interrupted, cleaning up resources");
            pool.shutdownNow(); // Forcefully terminate all tasks
        } else {
            // Normal completion
            Utils.await(latch, param.corePoolSize); // Check if all semaphores are released after completing all batches
            pool.shutdown();
        }

        long runningTime = System.nanoTime();
        System.out.println("Running Time in seconds: " + (runningTime - startTime) / 1000000000.0);
        timeUsage[1] = (runningTime - startTime) / 1000000000.0;

        // Post-processing
        param.status = "Finish";

        shutDownPrepare();

        // Print results
        System.out.println("Prepare time is: " + timeUsage[0] + " seconds");
        System.out.println("Running time is: " + timeUsage[1] + " seconds");

        if (cfg.isHeader()) {
            assert Objects.requireNonNull(input_data).size() >= logits.size();
            assert Objects.requireNonNull(input_data).size() >= param.numSample;
            for (int i = 0; i < param.numSample; i++) {
                if ((param.max_length == 0) && (param.task_type.equals("classification"))) {
                    System.out.println("The result of sample " + i + ": " + this.param.classes[binaryClassify(logits.get(i))]);
                    Log.d(TAG, "The result of sample " + i + ": " + this.param.classes[binaryClassify(logits.get(i))]);
                } else {
                    System.out.println(InputIds.get(i));
                    String decoding_String = decodeID(Utils.convertArrayListToIntArray(Objects.requireNonNull(InputIds.get(i))), tokenizer);
                    System.out.println("Generated sequence: " + decoding_String);
                    Log.d(TAG, "Generated sequence: " + decoding_String);
                }
            }
        }
    }

    // Class implementing Runnable interface, must override run() method, automatically called when submitted to thread pool
    class multiSteps implements Runnable {
        private Map<Integer, Socket> serverSocket;
        private Map<Integer, Socket> clientSocket;
        private final int sample_id;
        private final Semaphore latch;
        // Add status tracking variable
        private String prevStatus = ""; // Track previous status

        // Add: Aggregate log object
        private QueryLogAggregator logAggregator;
        // Add: Collect fault and energy events for this round
        private final List<Events.FaultEvent> faultEvents = new ArrayList<>();
        private final List<Events.EnergyEvent> energyEvents = new ArrayList<>();

        // In multiSteps class, add:
        private final Map<String, Long> faultStartTimes = new HashMap<>();

        // Constructor, initializes sample ID and semaphore, retrieves server and client socket maps from allSockets queue
        public multiSteps(int sample_id, Semaphore latch) {
            this.sample_id = sample_id;
            ArrayList<Map<Integer, Socket>> sockets = null;
            try {
                sockets = allSockets.take(); // Retrieve ArrayList<Map<Integer, Socket>> from queue, containing client and server socket configurations
            } catch (InterruptedException e) {
                System.out.println("Waiting for an element from the sockets queue...");
                e.printStackTrace();
            }
            this.clientSocket = sockets.get(0); // Get socket map for successor receiver
            this.serverSocket = sockets.get(1); // Get socket map for predecessor sender

            this.latch = latch;
            // Initialize aggregate log object
            logAggregator = new QueryLogAggregator();
            logAggregator.deviceId = String.valueOf(cfg.deviceId);
            logAggregator.role = cfg.isHeader() ? "header" : (cfg.isTailer() ? "tailer" : "worker");
            logAggregator.queryId = String.valueOf(sample_id);
            logAggregator.userQuery = cfg.isHeader() && InputString != null && InputString.length > sample_id ? InputString[sample_id] : "";
            // Register to global Map
            activeAggregators.put(sample_id, logAggregator);
            // Register event listeners
            EventBus.getDefault().register(this);
        }

        // Monitor fault and energy events for this round
        @Subscribe(threadMode = ThreadMode.BACKGROUND)
        public void onFaultEvent(Events.FaultEvent event) {
            synchronized (faultEvents) {
                faultEvents.add(event);
            }
        }

        @Subscribe(threadMode = ThreadMode.BACKGROUND)
        public void onEnergyEvent(Events.EnergyEvent event) {
            synchronized (energyEvents) {
                energyEvents.add(event);
            }
        }

        @Override
        public void run() {
            DataRepository.INSTANCE.updateSampleId(this.sample_id); // Update current batch in data repository
            System.out.println("SampleID: " + sample_id);
            if (param.max_length < 0) {
                System.out.println("ERROR: Set up max_length");
            } else if (param.max_length == 0) { // Classification task when max_length is 0
                System.out.println("SampleID: " + sample_id);
                System.out.println("param.max_length == 0");
                int receivedId = 0;
                try {
                    // Check recovery status before executing OneStep
                    if (!"Recovery".equals(param.status) && !"Recovering".equals(param.status) && !"Failure".equals(param.status)) {
                        receivedId = new OneStep(this.sample_id, serverSocket, clientSocket).run(logAggregator);
                    } else {
                        Log.d(TAG, "System in recovery or failure state, skipping inference");
                    }
                } catch (InterruptedException | JSONException e) {
                    throw new RuntimeException(e);
                }
                cleanUpBuffer(receivedId);
                // Inference complete, aggregate and report log
                logAggregator.tokens = cfg.isHeader() && InputIds.get(sample_id) != null ? InputIds.get(sample_id).size() : 0;
                if (logAggregator.tokens > 0 && logAggregator.tailerResultEnd > 0 && logAggregator.clientReceiveStart > 0) {
                    logAggregator.throughput = logAggregator.tokens * 1000.0 / (logAggregator.tailerResultEnd - logAggregator.clientReceiveStart);
                }
                // Package as SessionLogEvent and report
                Events.SessionLogEvent sessionLog = new Events.SessionLogEvent(
                        new Events.QueryLogEvent(
                                logAggregator.deviceId, logAggregator.role, logAggregator.queryId, logAggregator.userQuery, logAggregator.response,
                                logAggregator.clientReceiveStart, logAggregator.clientReceiveEnd,
                                logAggregator.inferenceStart, logAggregator.inferenceEnd,
                                logAggregator.serverSendStart, logAggregator.serverSendEnd,
                                logAggregator.tailerResultStart, logAggregator.tailerResultEnd,
                                logAggregator.tokens, logAggregator.throughput,
                                false, -1, -1,
                                logAggregator.clientReceiveTimes,
                                logAggregator.inferenceTimes,
                                logAggregator.serverSendTimes,
                                logAggregator.tailerResultTimes
                        ),
                        new ArrayList<>(faultEvents),
                        new ArrayList<>(energyEvents)
                );
                EventBus.getDefault().post(sessionLog);
                // Clean up Map to avoid memory leaks
                activeAggregators.remove(sample_id);
                // Unregister event listeners
                EventBus.getDefault().unregister(this);
            } else { // Generation task when max_length > 0
                int receivedId = sampleId - 1; // Get current batch
                int input_size = param.max_length; // Length of the string being processed
                System.out.println("Start inference current batch: " + this.sample_id + ", input_size is: " + input_size);

                Set<String> seenWindows = new HashSet<>();
                final int WINDOW_SIZE = 5;
                final int REPEAT_SIZE = 4;

                int m = 0;
                while (m < param.max_length) {
                    long startTime = System.nanoTime();
                    m += 1;
                    System.out.println("current token: " + m);
                    try {
                        // Update status check logic using class member variable prevStatus
                        // Check if system status transitions from recovery to running
                        Log.e(TAG, "Current system status: " + param.status + " Previous system status: " + prevStatus);
                        if (("ResumeStart".equals(param.status) && "Running".equals(prevStatus)) || (m == 1)) {
                            Log.d(TAG, "System transitioned from recovery to running, re-acquiring Socket");
                            long recoveryTime = System.currentTimeMillis();
                            String deviceId = "" + cfg.deviceId;
                            String roleStr = cfg.isHeader() ? "header" : (cfg.isTailer() ? "tailer" : "worker");
                            String faultType = "SYSTEM_FAILURE";
                            String affectedQueryId = String.valueOf(this.sample_id);
                            Long faultTime = Communication.faultStartTimes.get(affectedQueryId);
                            if (faultTime != null) {
                                Events.FaultEvent faultEvent = new Events.FaultEvent(deviceId, roleStr, faultType, faultTime, recoveryTime, affectedQueryId);
                                EventBus.getDefault().post(faultEvent);
                                Communication.faultStartTimes.remove(affectedQueryId);
                            } else {
                                // No corresponding fault time found, downgrade to standalone recovery event
                                Events.FaultEvent faultEvent = new Events.FaultEvent(deviceId, roleStr, faultType, -1, recoveryTime, affectedQueryId);
                                EventBus.getDefault().post(faultEvent);
                            }
                            // Add: Record to logAggregator
                            if (m == 0) {
                                Log.d(TAG, "m == 0");
                            }

                            try {
                                // Put current socket back to queue for other threads to use after recovery
                                allSockets.put(new ArrayList<Map<Integer, Socket>>() {{
                                    add(clientSocket);
                                    add(serverSocket);
                                }});

                                // Brief wait to ensure recovery process completes
                                Thread.sleep(500);

                                // Re-acquire socket from queue (new socket after recovery)
                                ArrayList<Map<Integer, Socket>> refreshedSockets = allSockets.take();
                                this.clientSocket = refreshedSockets.get(0);
                                this.serverSocket = refreshedSockets.get(1);

                                Log.d(TAG, "Socket updated, continuing inference");
                                param.status = "Running";
                            } catch (InterruptedException e) {
                                Log.e(TAG, "Interrupted while updating Socket: " + e.getMessage());
                            }
                        }

                        // If system is in recovery or failure state, skip current token processing and wait for recovery
                        if ("Recovery".equals(param.status) || "Recovering".equals(param.status) || "Failure".equals(param.status) || "WaitingStart".equals(param.status)) {
                            Log.d(TAG, "System in recovery or failure state, skipping current token processing");
                            Thread.sleep(100); // Brief sleep to avoid CPU spinning
                            m--; // Step back to reprocess current token after recovery
                            continue; // Skip current loop
                        }

                        // Update prevStatus for next loop status check
                        if (!prevStatus.equals(param.status)) {
                            prevStatus = param.status;
                        }

                        int flag = 1;
                        try {
                            // Pass logAggregator for aggregating timestamps
                            flag = new OneStep(this.sample_id, serverSocket, clientSocket).run(logAggregator);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                        // Restore logic for head node to update UI for each generated token
                        if (cfg.isHeader()) {
                            input_size = Math.min(input_size, InputIds.get(receivedId).size());
                            // Extract generated portion of the string
                            ArrayList<Integer> decodeList = new ArrayList(InputIds.get(receivedId).subList(input_size - 1, InputIds.get(receivedId).size()));
                            System.out.println("decode_ids: " + decodeList);
                            String decodedString = decodeID(Utils.convertArrayListToIntArray(
                                    Objects.requireNonNull(decodeList)), tokenizer);
                            System.out.println("decodedString: " + decodedString);
                            DataRepository.INSTANCE.updateDecodingString(decodedString);
                            System.out.println("token " + m + " Results Obtained");
                            // Add: Record latest response each time
                            logAggregator.response = decodedString;
                        }

                        if (flag == 0) {
                            Log.e(TAG, "flag = 0");
                        }

                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                    System.out.println("Token " + m + " Process Time: " + (System.nanoTime() - startTime) / 1000000000.0);
                }
                cleanUpBuffer(this.sample_id); // Clean up buffer

                logAggregator.tokens = cfg.isHeader() && InputIds.get(sample_id) != null ? InputIds.get(sample_id).size() : 0;
                if (logAggregator.tokens > 0 && logAggregator.tailerResultEnd > 0 && logAggregator.clientReceiveStart > 0) {
                    logAggregator.throughput = logAggregator.tokens * 1000.0 / (logAggregator.tailerResultEnd - logAggregator.clientReceiveStart);
                }
                // Package as SessionLogEvent and report
                Events.SessionLogEvent sessionLog = new Events.SessionLogEvent(
                        new Events.QueryLogEvent(
                                logAggregator.deviceId, logAggregator.role, logAggregator.queryId, logAggregator.userQuery, logAggregator.response,
                                logAggregator.clientReceiveStart, logAggregator.clientReceiveEnd,
                                logAggregator.inferenceStart, logAggregator.inferenceEnd,
                                logAggregator.serverSendStart, logAggregator.serverSendEnd,
                                logAggregator.tailerResultStart, logAggregator.tailerResultEnd,
                                logAggregator.tokens, logAggregator.throughput,
                                false, -1, -1,
                                logAggregator.clientReceiveTimes,
                                logAggregator.inferenceTimes,
                                logAggregator.serverSendTimes,
                                logAggregator.tailerResultTimes
                        ),
                        new ArrayList<>(faultEvents),
                        new ArrayList<>(energyEvents)
                );
                EventBus.getDefault().post(sessionLog);
                // Clean up Map to avoid memory leaks
                activeAggregators.remove(sample_id);
                // Unregister event listeners
                EventBus.getDefault().unregister(this);
            }

            try { // Put client and server sockets back into allSockets queue
                allSockets.put(new ArrayList<Map<Integer, Socket>>() {{
                    add(clientSocket);
                    add(serverSocket);
                }});
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            latch.release(); // Release semaphore
        }
    }

    /**
     * OneStep class manages single-step processing in distributed inference
     * Handles data reception, computation, and result sending between devices
     */
    public class OneStep {
        // Store socket mappings for communication with other devices
        private final Map<Integer, Socket> serverSocketMap; // Server socket map for receiving data
        private final Map<Integer, Socket> clientSocketMap; // Client socket map for sending data
        private final Socket serverSocket; // Socket for receiving data from predecessor node
        private final Socket clientSocket; // Socket for sending data to successor node
        private final int sample_id; // Current sample/batch ID
        private int current_token_index; // Current token index being processed

        /**
         * Constructor, initializes communication sockets and sample ID
         * 
         * @param sample_id Current sample ID being processed
         * @param serverSide Server socket map for receiving data
         * @param clientSide Client socket map for sending data
         */
        public OneStep(int sample_id, Map<Integer, Socket> serverSide, Map<Integer, Socket> clientSide) {
            this.sample_id = sample_id;
            this.serverSocketMap = serverSide;
            this.clientSocketMap = clientSide;
            this.serverSocket = serverSide.get(cfg.prevDeviceId()); // Socket for receiving from predecessor
            this.clientSocket = clientSide.get(cfg.nextDeviceId()); // Socket for sending to successor
        }

        /**
         * Method to process data as a client
         * Requests data from predecessor node and receives data sent by predecessor
         * 
         * @param receivedId Sample ID received
         * @return Processed sample ID
         * @throws InterruptedException If thread is interrupted
         */
        public int procssingAsClient(int receivedId) throws InterruptedException {
            if (!cfg.isHeader()) { // If not head node
                System.out.println("Start to be a Client");

                // Check system status; skip waiting for results if recovering
                if ("Recovery".equals(param.status) || "Recovering".equals(param.status) || "Failure".equals(param.status) || "WaitingStart".equals(param.status)) {
                    Log.d(TAG, "System in fault recovery, skipping client data reception");
                    return receivedId;
                }

                // Check for termination request
                if (!isRunning) {
                    Log.d(TAG, "Inference interrupted, skipping client data reception");
                    return receivedId;
                }

                try {
                    serverSocket.send("Request Data"); // Send data request to predecessor

                    // Receive and parse sample ID
                    byte[] idData = serverSocket.recv(0);
                    if (idData == null) {
                        Log.w(TAG, "Timeout waiting for predecessor response, possible fault");
                        return receivedId;
                    }
                    receivedId = Utils.convertByteArrayToInt(idData);

                    // Start a thread to asynchronously receive residual data (for model optimization and acceleration)
                    Thread workerThread = new Thread(new ReceiveResidualConnection(receivedId, serverSocketMap));
                    workerThread.start();

                    // Verify if received sample ID matches current sample ID; warn if mismatch
                    if (receivedId != this.sample_id) {
                        System.out.println("Client: Data out of order, sampleId: " + this.sample_id + ", receivedId: " + receivedId);
                    }

                    // Receive actual data from predecessor
                    byte[] msgFrom = serverSocket.recv(0);
                    if (msgFrom == null) {
                        Log.w(TAG, "Timeout waiting for predecessor data, possible fault");
                        return receivedId;
                    }

                    // Store received data
                    InputData.put(receivedId, msgFrom);
                    System.out.println("Received Data");

                    // Wait for residual data reception thread to complete
                    workerThread.join();
                    System.out.println("Received ResData");
                } catch (org.zeromq.ZMQException e) {
                    // Handle ZMQ exception, usually due to socket operation interruption during fault recovery
                    Log.w(TAG, "Socket operation interrupted, possibly due to fault recovery: " + e.getMessage());
                    return receivedId;
                }
            } else {
                // As head node, load data locally instead of receiving from other nodes
                if (logits.get(receivedId) == null) {
                    System.out.println("Load Data");
                }
            }
            return receivedId; // Return processed sample ID
        }

        /**
         * Method to process data as a server
         * Sends data response to successor node
         * 
         * @param receivedId Current sample ID being processed
         * @throws InterruptedException If thread is interrupted
         */
        public void processAsServer(int receivedId) throws InterruptedException {
            // Return data to head node or successor node
            if (clientSocket == null) {
                Log.e(TAG, "ProcessAsServer Error: clientSocket is null");
                return; // Cannot proceed
            }

            // Check system status; skip waiting for requests if recovering
            if ("Recovery".equals(param.status) || "Recovering".equals(param.status) || "Failure".equals(param.status) || "WaitingStart".equals(param.status)) {
                Log.d(TAG, "System in fault recovery, skipping server data sending");
                return;
            }

            // Check for termination request
            if (!isRunning) {
                Log.d(TAG, "Inference interrupted, skipping server data sending");
                return;
            }

            System.out.println("Start to be a Server");

            try {
                // Receive request ID and content from successor node
                byte[] comefrom_id = clientSocket.recv(0);
                if (comefrom_id == null) {
                    Log.w(TAG, "Timeout waiting for successor request ID, possible fault");
                    return;
                }

                byte[] msgTo = clientSocket.recv(0);
                if (msgTo == null) {
                    Log.w(TAG, "Timeout waiting for successor request content, possible fault");
                    return;
                }

                // If data request received
                if (new String(msgTo).contains("Request Data")) {
                    // Check if data is available to send
                    if (OutputData.containsKey(receivedId)) {
                        byte[] id = "from".getBytes();

                        // Start thread to send residual data
                        Thread workerThread = new Thread(new SendResidualConnection(receivedId, clientSocketMap));
                        workerThread.start();

                        // Send sample ID
                        id = Utils.convertIntToByteArray(receivedId);
                        boolean sendSuccess = true;

                        try {
                            clientSocket.sendMore(comefrom_id); // Send requester ID first
                            clientSocket.sendMore(id); // Then send sample ID

                            // Send output data; for tail node and generation task, send specific decode ID
                            if (cfg.isTailer() && (param.task_type.equals("generation"))) {
                                byte[] decode_id = OutputData.get(receivedId);
                                clientSocket.send(decode_id, 0);
                            } else {
                                clientSocket.send(OutputData.get(receivedId), 0);
                            }
                        } catch (org.zeromq.ZMQException e) {
                            Log.w(TAG, "Exception during data sending: " + e.getMessage());
                            sendSuccess = false;
                        }

                        if (sendSuccess) {
                            System.out.println("Sent Data");
                        } else {
                            Log.w(TAG, "Data sending failed, possibly due to disconnected receiver");
                        }

                        // Wait for residual data sending thread to complete
                        workerThread.join();
                    } else {
                        // Warn if data does not exist
                        System.out.println(receivedId + " is not in the OutputData");
                    }
                }
            } catch (org.zeromq.ZMQException e) {
                // Handle ZMQ exception, usually due to socket operation interruption during fault recovery
                Log.w(TAG, "Socket operation interrupted, possibly due to fault recovery: " + e.getMessage());
            }
        }

        /**
         * Method to obtain results from tail node
         * Used only by head node to obtain final results of distributed inference
         * 
         * @param receivedId Current sample ID being processed
         * @return Processing status flag: 0 for completed, 1 for continue processing
         */
        public int obtainResultsFromTailer(int receivedId) {
            // Head node-specific function to obtain results from tail node
            int flag = 1; // Default continue processing flag

            if (cfg.isHeader()) {
                try {
                    System.out.println("Start to obtain result from tailer");

                    // Check system status; skip waiting for results if recovering
                    if ("Recovery".equals(param.status) || "Recovering".equals(param.status) || "Failure".equals(param.status) || "WaitingStart".equals(param.status)) {
                        Log.d(TAG, "System in fault recovery, skipping wait for tail node results");
                        return flag;
                    }

                    // Check for termination request
                    if (!isRunning) {
                        Log.d(TAG, "Inference interrupted, skipping wait for tail node results");
                        return flag;
                    }

                    // Request results from tail node
                    serverSocket.send("Request Data");

                    // Set temporary receive timeout to prevent permanent blocking during faults
                    int originalTimeout = serverSocket.getReceiveTimeOut();
                    serverSocket.setReceiveTimeOut(5000); // 5-second timeout

                    // Receive sample ID
                    byte[] idData = serverSocket.recv(0);
                    Log.w(TAG, "Received idData");
                    if (idData == null) {
                        Log.w(TAG, "Timeout waiting for tail node response, possible fault");
                        return flag;
                    }
                    receivedId = Utils.convertByteArrayToInt(idData);

                    // Verify if sample ID matches
                    if (receivedId != this.sample_id) {
                        System.out.println("Server: Data out of order, sampleId: " + this.sample_id + ", receivedId: " + receivedId);
                    }

                    // Receive result data
                    byte[] res = serverSocket.recv(0);
                    if (res == null) {
                        Log.w(TAG, "Timeout waiting for tail node data, possible fault");
                        return flag;
                    }

                    // Restore original timeout setting
                    serverSocket.setReceiveTimeOut(originalTimeout);

                    // Process data based on task type
                    if (param.task_type.equals("generation")) {
                        // Generation task: Parse decode ID and add to input sequence
                        int decode_id = deserializeInt(res);
                        System.out.println("Obtain decode_id: " + decode_id);

                        // If decode ID is 2 (usually end marker), set completion flag
                        if (decode_id == 2) {
                            flag = 0;
                            Log.e(TAG, "If decode ID is 2 (usually end marker), set completion flag = 0");
                        }

                        // Add decode ID to input sequence
                        InputIds.get(receivedId).add(decode_id);
                    } else {
                        // Non-generation task: Store logits data directly
                        logits.put(receivedId, res);
                    }
                } catch (org.zeromq.ZMQException e) {
                    // Handle ZMQ exception, usually due to socket operation interruption during fault recovery
                    Log.w(TAG, "Socket operation interrupted, possibly due to fault recovery: " + e.getMessage());

                    // Check if system is recovering
                    if ("Recovery".equals(param.status) || "Recovering".equals(param.status) || "Failure".equals(param.status) || "WaitingStart".equals(param.status)) {
                        Log.d(TAG, "System executing fault recovery, socket interruption is expected");
                    } else {
                        Log.e(TAG, "Unexpected socket communication interruption", e);
                    }
                } catch (Exception e) {
                    // Handle other exceptions
                    Log.e(TAG, "Exception occurred while obtaining tail node results: " + e.getMessage(), e);
                }
            }
            return flag; // Return processing status flag
        }

        /**
         * Execute complete single-step inference process
         * Includes client processing, inference computation, server processing, and result retrieval
         * 
         * @return Processing status flag: 0 for completed, 1 for continue processing
         * @throws RuntimeException Runtime exception
         * @throws InterruptedException If thread is interrupted
         * @throws JSONException JSON processing exception
         */
        public int run(QueryLogAggregator aggregator) throws RuntimeException, InterruptedException, JSONException {
            int receivedId = this.sample_id; // Get current batch ID
            int flag = 1; // Default continue processing flag

            // Detailed timestamps for four phases
            long clientReceiveStart = 0, clientReceiveEnd = 0;
            long inferenceStart = 0, inferenceEnd = 0;
            long serverSendStart = 0, serverSendEnd = 0;
            long tailerResultStart = 0, tailerResultEnd = 0;

            try {
                // Check system status and whether inference is interrupted
                if ("Recovery".equals(param.status) || "Recovering".equals(param.status) || "Failure".equals(param.status) || "WaitingStart".equals(param.status)) {
                    Log.d(TAG, "System in fault recovery, skipping inference step");
                    return flag;
                }
                if (!isRunning) {
                    Log.d(TAG, "Inference interrupted, skipping inference step");
                    return flag;
                }

                // Step 1: Receive data as client
                clientReceiveStart = System.currentTimeMillis();
                try {
                    receivedId = procssingAsClient(receivedId);
                } catch (org.zeromq.ZMQException e) {
                    Log.w(TAG, "Socket operation interrupted during client data reception: " + e.getMessage());
                    checkSystemStatus();
                    return flag;
                }
                clientReceiveEnd = System.currentTimeMillis();
                Log.d(TAG, "[QueryLog] Client receive phase: " + (clientReceiveEnd - clientReceiveStart) + " ms, sampleId: " + receivedId);

                if (checkSystemStatus()) return flag;

                // Step 2: Perform inference computation
                inferenceStart = System.currentTimeMillis();
                try {
                    inferenceProcedure(receivedId); // Call inference processing method
                } catch (Exception e) {
                    Log.e(TAG, "Exception during inference computation: " + e.getMessage(), e);
                    checkSystemStatus();
                    return flag;
                }
                inferenceEnd = System.currentTimeMillis();
                Log.d(TAG, "[QueryLog] Inference phase: " + (inferenceEnd - inferenceStart) + " ms, sampleId: " + receivedId);

                if (checkSystemStatus()) return flag;

                // Step 3: Send data as server
                serverSendStart = System.currentTimeMillis();
                try {
                    processAsServer(receivedId);
                } catch (org.zeromq.ZMQException e) {
                    Log.w(TAG, "Socket operation interrupted during server data sending: " + e.getMessage());
                    checkSystemStatus();
                    return flag;
                }
                serverSendEnd = System.currentTimeMillis();
                Log.d(TAG, "[QueryLog] Server send phase: " + (serverSendEnd - serverSendStart) + " ms, sampleId: " + receivedId);

                if (checkSystemStatus()) return flag;

                // Step 4: Obtain tail node results
                tailerResultStart = System.currentTimeMillis();
                try {
                    flag = obtainResultsFromTailer(receivedId); // Obtain processing status flag
                } catch (Exception e) {
                    Log.w(TAG, "Exception while obtaining tail node results: " + e.getMessage());
                    checkSystemStatus();
                    return flag;
                }
                tailerResultEnd = System.currentTimeMillis();
                Log.d(TAG, "[QueryLog] Tailer result phase: " + (tailerResultEnd - tailerResultStart) + " ms, sampleId: " + receivedId);

            } catch (Exception e) {
                Log.e(TAG, "Unexpected exception in inference process: " + e.getMessage(), e);
            }

            // Append this token's timestamps to the aggregator object
            if (aggregator != null) {
                aggregator.updatePhaseTimes(
                        clientReceiveStart, clientReceiveEnd,
                        inferenceStart, inferenceEnd,
                        serverSendStart, serverSendEnd,
                        tailerResultStart, tailerResultEnd
                );
            }

            return flag;
        }

        /**
         * Check system status
         * 
         * @return true if system is in fault recovery state or inference stopped, false for normal operation
         */
        private boolean checkSystemStatus() {
            if ("Recovery".equals(param.status) || "Recovering".equals(param.status) || "Failure".equals(param.status) || "WaitingStart".equals(param.status)) {
                Log.d(TAG, "System currently in fault recovery state, inference step interrupted");
                return true;
            }

            if (!isRunning) {
                Log.d(TAG, "Inference interrupted, current step terminated");
                return true;
            }

            return false;
        }
    }

    /**
     * Update communication sockets
     * Create sockets for communication with predecessor and successor nodes for each processing core
     * 
     * @param corePoolSize Thread pool core size, determines number of socket groups created
     * @throws InterruptedException If thread is interrupted
     */
    public void updateSockets(int corePoolSize) throws InterruptedException {
        Log.d(TAG, "Starting updateSockets, core pool size: " + corePoolSize);
        Log.d(TAG, "Send device set: " + sendDeviceIndex + ", Receive device set: " + receiveDeviceIndex);
        if ("Recovery".equals(param.status) || "Recovering".equals(param.status) || "Failure".equals(param.status) || "WaitingStart".equals(param.status)) {
            Config.port = 10000;
            Log.e(TAG, "updateSockets Port: " + Config.port);
        } else {
            Log.e(TAG, "updateSockets Port: " + Config.port);
        }
        int j = cfg.ipGraph.length; // Get length of IP graph (total number of devices)
        System.out.println("Graph length: " + j);
        Log.d(TAG, "IP graph content: " + Arrays.toString(cfg.ipGraph) + ", Current device ID: " + cfg.deviceId);

        // Create one socket group for each core
        for (int i = 0; i < corePoolSize; i++) {
            Log.d(TAG, "Creating socket group for core " + i);
            ArrayList<Map<Integer, Socket>> socketContainer = new ArrayList<>(); // Create socket container

            // Create send socket map
            Log.d(TAG, "Starting to create send socket map, device ID: " + cfg.deviceId);
            Map<Integer, Socket> SendSocket = new HashMap<>();

            for (Integer idx : sendDeviceIndex) {
                int portNum = Config.port + j * i + (idx - cfg.deviceId);
                int maxRetries = 10;
                for (int retry = 0; retry < maxRetries; retry++) {
                    try {
                        Log.d(TAG, "Attempting to create send socket to device: " + idx + ", port: " + portNum + ", retry: " + retry);
                        Socket temp = beServer.establish_connection(context, SocketType.ROUTER, portNum);
                        temp.setIdentity(("ROUTER Send From " + cfg.deviceId + " to " + idx + "." + portNum).getBytes());
                        SendSocket.put(idx, temp);
                        Log.d(TAG, "Successfully created send socket to device: " + idx);
                        break;
                    } catch (ZMQException e) {
                        if (e.getErrorCode() == ZMQ.Error.EADDRINUSE.getCode() && retry < maxRetries - 1) {
                            Log.w(TAG, "Port " + portNum + " in use, retrying...");
                            Thread.sleep(200); // Wait 200ms
                        } else {
                            throw new RuntimeException("Failed to create send socket to device: " + idx, e);
                        }
                    }
                }
            }

            // If current node is tail node, create additional socket for communication with head node
            if (cfg.isTailer()) {
                try {
                    int portNum = Config.port + j * i + 1;
                    Log.d(TAG, "Tail node creating additional socket to head node, port: " + portNum);
                    Socket temp = beServer.establish_connection(context, SocketType.ROUTER, portNum);

                    temp.setIdentity(("ROUTER Send From " + cfg.deviceId + " to " + cfg.nextDeviceId() + "." + portNum).getBytes());

                    SendSocket.put(cfg.nextDeviceId(), temp);
                    Log.d(TAG, "Tail node successfully created additional socket to head node: " + cfg.nextDeviceId());
                } catch (Exception e) {
                    Log.e(TAG, "Tail node failed to create additional socket to head node: " + e.getMessage(), e);
                }
            }

            // Add send socket map to container
            socketContainer.add(SendSocket);
            Log.d(TAG, "Send socket map added to container, size: " + SendSocket.size());

            // Create receive socket map
            Log.d(TAG, "Starting to create receive socket map");
            Map<Integer, Socket> receiveSocket = new HashMap<>();
            for (Integer idx : receiveDeviceIndex) { // Iterate through device indices for receiving data
                try {
                    int portNum = Config.port + j * i + (cfg.deviceId - idx);
                    String targetIP = cfg.ipGraph[idx];
                    Log.d(TAG, "Attempting to create receive socket from device: " + idx + ", IP: " + targetIP + ", port: " + portNum);

                    // Create dealer-type socket (many-to-one communication)
                    Socket temp = beClient.establish_connection(context, SocketType.DEALER, portNum, targetIP);

                    // Set socket identity
                    temp.setIdentity(("DEALER Receive From: " + cfg.deviceId + " to " + idx + "." + portNum).getBytes());

                    // Add socket to map
                    receiveSocket.put(idx, temp);
                    Log.d(TAG, "Successfully created receive socket from device: " + idx);
                } catch (Exception e) {
                    Log.e(TAG, "Failed to create receive socket from device: " + idx + ": " + e.getMessage() + ", IP: " + cfg.ipGraph[idx] + ", port: " + (Config.port + j * i + (cfg.deviceId - idx)), e);
                }
            }

            // If current node is head node, create additional socket for communication with tail node
            if (cfg.isHeader()) {
                try {
                    int portNum = Config.port + j * i + 1;
                    String targetIP = cfg.prevNodes.get(0);
                    Log.d(TAG, "Head node creating additional socket from tail node, IP: " + targetIP + ", port: " + portNum);
                    Socket temp = beClient.establish_connection(context, SocketType.DEALER, portNum, targetIP);

                    temp.setIdentity(("DEALER Receive From: " + cfg.deviceId + " to " + cfg.nextDeviceId() + "." + portNum).getBytes());

                    receiveSocket.put(cfg.prevDeviceId(), temp);
                    Log.d(TAG, "Head node successfully created additional socket from tail node: " + cfg.prevDeviceId());
                } catch (Exception e) {
                    Log.e(TAG, "Head node failed to create additional socket from tail node: " + e.getMessage() + ", IP: " + cfg.prevNodes.get(0) + ", port: " + (Config.port + j * i + 1), e);
                }
            }

            // Add receive socket map to container
            socketContainer.add(receiveSocket);
            Log.d(TAG, "Receive socket map added to container, size: " + receiveSocket.size());

            try {
                // Add entire socket container to global socket list
                allSockets.put(socketContainer);
                Log.d(TAG, "Socket container added to global queue");
            } catch (Exception e) {
                Log.e(TAG, "Failed to add socket container to global queue: " + e.getMessage(), e);
            }
        }

        Log.d(TAG, "updateSockets function completed");
        System.out.println("Sockets are built successfully"); // Output socket creation success message
    }

    /**
     * Get device-to-module mapping for sending residual data
     * Used to optimize communication for inter-model residual connections
     */
    public void getSendResDevice2Device() {
        sendD2D = new TreeMap<>(); // Create ordered map to store device-module relationships for sending residual data

        // Iterate through all send indices
        for (ArrayList<JSONObject> sendIndexList : sendIndex.values()) {
            // If residual index exists (usually the second index)
            if (sendIndexList.size() > 1) {
                JSONObject sendResIndex = sendIndexList.get(1); // Get residual index
                Iterator<String> keys = sendResIndex.keys(); // Get all module keys

                // Iterate through all modules
                while (keys.hasNext()) {
                    String k = keys.next(); // Get module name
                    int device = module_on_devices.get(k); // Get device ID where module resides

                    // If module is not on current device, send via network
                    if (device != cfg.deviceId) {
                        // Create new list if mapping does not exist for this device
                        if (!sendD2D.containsKey(device))
                            sendD2D.put(device, new ArrayList<>());

                        // Add module name to the device's list
                        sendD2D.get(device).add(k);
                    }
                }
            }
        }

        // Sort module lists for each device to ensure consistent operation order
        for (List<String> i : sendD2D.values())
            Collections.sort(i);
    }

  /**
 * Retrieves the device-to-module mapping for receiving residual data
 * Used to optimize communication for residual connections between models
 */
public void getReceiveResDevice2Device() {
    receiveD2D = new TreeMap<>();  // Create an ordered map to store device-module relationships for receiving residual data

    // Iterate through all receive indices
    for (Map.Entry<String, ArrayList<JSONObject>> receiveIndexList : receiveIndex.entrySet()) {
        // If there is a residual index (usually the second index)
        if (receiveIndexList.getValue().size() > 1) {
            JSONObject receiveResIndex = receiveIndexList.getValue().get(1);  // Get the residual index
            Iterator<String> keys = receiveResIndex.keys();  // Get all module keys

            // Iterate through all modules
            while (keys.hasNext()) {
                String k = keys.next();  // Get module name
                int device = module_on_devices.get(k);  // Get the device ID where the module resides

                // If the module is not on the current device, it needs to be received over the network
                if (device != cfg.deviceId) {
                    // If the device is not in the map, create a new list
                    if (!receiveD2D.containsKey(device))
                        receiveD2D.put(device, new ArrayList<>());

                    // Add the current module name to the corresponding device's list
                    receiveD2D.get(device).add(receiveIndexList.getKey());
                }
            }
        }
    }

    // Sort the module lists for each device to ensure consistent operation order
    for (List<String> i : receiveD2D.values())
        Collections.sort(i);
}

public void RecordResult(Object[] result) {
    // Assume each result[i] is of type byte[]
    // Calculate the total length of all arrays
    int totalLength = 0;
    for (Object obj : result) {
        byte[] bytes = (byte[]) obj;
        totalLength += bytes.length;
    }

    // Create an array large enough to hold all bytes
    byte[] res = new byte[totalLength];
    int currentIndex = 0;
    for (Object obj : result) {
        byte[] bytes = (byte[]) obj;
        System.arraycopy(bytes, 0, res, currentIndex, bytes.length);
        currentIndex += bytes.length;
    }

    // Construct a unique filename based on whether the file exists
    int fileIndex = 0;
    String fileName = "result_" + fileIndex + ".bin";
    File file = new File(conText.getFilesDir(), fileName);
    while (file.exists()) {
        fileIndex++;
        fileName = "result_" + fileIndex + ".bin";
        file = new File(conText.getFilesDir(), fileName);
    }

    // Write the merged byte array to the file
    try (FileOutputStream fos = new FileOutputStream(file)) {
        fos.write(res);
        fos.flush();
        Log.d(TAG, "File " + fileName + " saved at: " + file.getAbsolutePath());
    } catch (IOException e) {
        e.printStackTrace();
    }
}

class SendResidualConnection implements Runnable {
    int receiveId;
    Map<Integer, Socket> clientSide;

    public SendResidualConnection(int receiveId, Map<Integer, Socket> clientSide) {
        this.receiveId = receiveId;
        this.clientSide = clientSide;
    }

    @Override
    public void run() {
        for (Map.Entry<Integer, ArrayList<String>> entry : sendD2D.entrySet()) {
            int target_device_id = entry.getKey();
            System.out.println("Send to device " + target_device_id);
            Socket sendSocket = this.clientSide.get(target_device_id);
            System.out.println(new String(sendSocket.getIdentity()));
            byte[] comefrom_id = sendSocket.recv(0);
            int target_id = Utils.convertByteArrayToInt(sendSocket.recv(0));
            assert target_id == target_device_id;
            byte[] msgTo = sendSocket.recv(0);
            System.out.println(new String(msgTo));
            if (new String(msgTo).contains("Request Res Data")) {
                sendSocket.sendMore(comefrom_id);
                sendSocket.sendMore(Utils.convertIntToByteArray(cfg.deviceId));
                System.out.println("Target Device ID: " + target_id);
                List<String> sendByte = entry.getValue();
                for (String k : sendByte) {
                    ArrayList<byte[]> data = ResidualDataToDevice.get(receiveId).get(k);
                    for (byte[] i : data)
                        sendSocket.sendMore(i);
                    sendSocket.sendMore(";");
                }
                sendSocket.send("Over");
            }
            System.out.println("Send the Residual Data to Device " + entry.getKey());
        }
    }
}

class ReceiveResidualConnection implements Runnable {
    int receiveId;
    Map<Integer, Socket> serverSide;

    public ReceiveResidualConnection(int receiveId, Map<Integer, Socket> serverSide) {
        this.receiveId = receiveId;
        this.serverSide = serverSide;
    }

    @Override
    public void run() {
        receiveIndex.keySet();
        for (Map.Entry<Integer, ArrayList<String>> entry : receiveD2D.entrySet()) {
            Socket receiveSocket = serverSide.get(entry.getKey());
            System.out.println(new String(receiveSocket.getIdentity()));
            receiveSocket.sendMore(Utils.convertIntToByteArray(cfg.deviceId));
            receiveSocket.send("Request Res Data");
            int send_device_id = Utils.convertByteArrayToInt(receiveSocket.recv(0));
            System.out.println("Actual Receive the Residual Data from Device " + send_device_id);

            int i = 0;
            List<String> keyOnDevices = entry.getValue();
            Map<String, ArrayList<byte[]>> tmpReceiver = ResidualDataFromDevice.get(receiveId); // get() method returns a reference to the object, not a copy
            if (tmpReceiver == null) {
                tmpReceiver = new TreeMap<>();
                ResidualDataFromDevice.put(receiveId, tmpReceiver);
            }
            tmpReceiver.put(keyOnDevices.get(i), new ArrayList<>());

            while (true) {
                byte[] data = receiveSocket.recv(0);
                if (new String(data).equals("Over")) {
                    break;
                } else if (new String(data).equals(";")) {
                    i += 1;
                    if (keyOnDevices.size() > i && !tmpReceiver.containsKey(keyOnDevices.get(i)))
                        tmpReceiver.put(keyOnDevices.get(i), new ArrayList<>());
                } else {
                    tmpReceiver.get(keyOnDevices.get(i)).add(data);
                }
            }

            System.out.println("Receive the Residual Data from Device " + entry.getKey());
            System.out.println(this.receiveId + " With the idx and size ");
        }
    }
}

public Socket getSocketsInQueue(LinkedBlockingQueue<Socket> queue, String identity) {
    Iterator<Socket> iterator = queue.iterator();
    while (iterator.hasNext()) {
        Socket item = iterator.next();
        if (Arrays.toString(item.getIdentity()).equals(identity)) {
            iterator.remove();  // Remove the current item
            return item;
        }
    }
    return null;
}

public native int tensorSizeDebug(byte[] logits);
public native byte[] performInferenceMaster(long session, int[] input_ids);
public native byte[] performInferenceWorker(long session, byte[] data);
public native int binaryClassify(byte[] data);
public native int[] encodeString(String input_string, long tokenizer);
public native int greedyDecoding(byte[] data);
public native String decodeID(int[] data, long tokenizer);

public native Double modelFlopsPerSecond(int modelFlops, long session, int[] input_ids_j);
public native Object runInferenceMasterResidual(long session, int[] input_ids_j, int[] to_send_seq_indices, int[][] to_send_res_indices);
public native Object runInferenceWorkerResidual(long session, byte[] sequential_input, ArrayList<byte[]> residual_input, int[] to_send_seq_indices, int[][] to_send_res_indices);
public native byte[] runInferenceWorkerResidualLast(long session, byte[] sequential_input, ArrayList<byte[]> residual_input);

public native byte[] runInferenceWorkerResidualLastGeneration(long session, byte[] sequential_input, ArrayList<byte[]> residual_input, int k, float init_temp);

public native byte[] runInferenceWorkerResidualLastClassification(long session, byte[] sequential_input, ArrayList<byte[]> residual_input);

public native int deserializeInt(byte[] decode_id);

public native int TokenToID(String token, long tokenizer);

public native boolean EosCheck(byte[] output, long tokenizer); // TODO: adding EOS string check for generation early stopping - Junchen 02/28/2024

/**
 * Safely terminates the inference process
 * Called during fault recovery to stop existing inference threads
 *
 * @return whether the inference was successfully stopped
 */
public boolean stopInference() {
    try {
        Log.d(TAG, "Request to stop inference process");

        // Set stop flag
        isRunning = false;

        // If there is an active inference thread pool, attempt to terminate it
        if (pool != null && !pool.isTerminated()) {
            Log.d(TAG, "Starting to shut down inference thread pool...");

            // First attempt a normal shutdown, allowing tasks to complete
            pool.shutdown();

            // Set socket timeout to avoid indefinite blocking
            if (allSockets != null && !allSockets.isEmpty()) {
                ArrayList<ArrayList<Map<Integer, Socket>>> socketsCopy = new ArrayList<>();
                allSockets.drainTo(socketsCopy);

                Log.d(TAG, "Setting all socket timeouts to 100 milliseconds");
                for (ArrayList<Map<Integer, Socket>> socketPair : socketsCopy) {
                    for (Map<Integer, Socket> socketMap : socketPair) {
                        for (Socket socket : socketMap.values()) {
                            try {
                                // Set a short receive timeout
                                socket.setReceiveTimeOut(100);

                                // Attempt to send an "INTERRUPT" message to wake up waiting threads
                                socket.send("INTERRUPT", ZMQ.DONTWAIT);
                            } catch (Exception e) {
                                // Ignore send errors and continue processing
                                Log.w(TAG, "Failed to set socket parameters: " + e.getMessage());
                            }
                        }
                    }
                }

                // Put sockets back into the queue
                for (ArrayList<Map<Integer, Socket>> socketPair : socketsCopy) {
                    allSockets.put(socketPair);
                }
            }

            // Wait for a period to allow inference threads to respond to the stop flag
            try {
                Log.d(TAG, "Waiting for inference threads to terminate normally...");
                boolean terminated = pool.awaitTermination(5000, TimeUnit.MILLISECONDS);

                // If timeout occurs and still not terminated, force shutdown
                if (!terminated) {
                    Log.w(TAG, "Inference thread pool did not terminate on its own, forcing shutdown");
                    List<Runnable> pendingTasks = pool.shutdownNow();
                    Log.d(TAG, "Number of unexecuted tasks: " + pendingTasks.size());

                    // Wait again to ensure threads are interrupted
                    try {
                        terminated = pool.awaitTermination(2000, TimeUnit.MILLISECONDS);
                        if (!terminated) {
                            Log.e(TAG, "Thread pool still not fully shut down after forced interruption");
                        }
                    } catch (InterruptedException ie) {
                        Log.w(TAG, "Interrupted while waiting for thread pool shutdown");
                        Thread.currentThread().interrupt(); // Reset interrupt status
                    }
                }
            } catch (InterruptedException e) {
                Log.w(TAG, "Interrupted while waiting for thread pool shutdown");
                // If the current thread is interrupted, restore interrupt status and continue
                Thread.currentThread().interrupt();
                // Force shutdown of the thread pool
                pool.shutdownNow();
            }

            Log.d(TAG, "Inference threads terminated");
            return true;
        } else {
            Log.d(TAG, "No active inference threads to stop");
            return false;
        }
    } catch (Exception e) {
        Log.e(TAG, "Failed to stop inference: " + e.getMessage());
        e.printStackTrace();
        return false;
    }
}

// Helper class for aggregating logs of one interaction
public static class QueryLogAggregator {
    public String deviceId;
    public String role;
    public String queryId;
    public String userQuery;
    public String response;
    public long clientReceiveStart = -1;
    public long clientReceiveEnd = -1;
    public long inferenceStart = -1;
    public long inferenceEnd = -1;
    public long serverSendStart = -1;
    public long serverSendEnd = -1;
    public long tailerResultStart = -1;
    public long tailerResultEnd = -1;
    public int tokens = 0;
    public double throughput = 0.0;
    // New: Detailed phase timestamps for each token
    public List<long[]> clientReceiveTimes = new ArrayList<>(); // [start, end]
    public List<long[]> inferenceTimes = new ArrayList<>();
    public List<long[]> serverSendTimes = new ArrayList<>();
    public List<long[]> tailerResultTimes = new ArrayList<>();

    // Record first/last timestamps for each phase
    public void updatePhaseTimes(long crs, long cre, long is, long ie, long sss, long sse, long trs, long tre) {
        if (clientReceiveStart == -1 || crs < clientReceiveStart) clientReceiveStart = crs;
        if (clientReceiveEnd == -1 || cre < clientReceiveEnd) clientReceiveEnd = cre;
        if (inferenceStart == -1 || is < inferenceStart) inferenceStart = is;
        if (inferenceEnd == -1 || ie > inferenceEnd) inferenceEnd = ie;
        if (serverSendStart == -1 || sss < serverSendStart) serverSendStart = sss;
        if (serverSendEnd == -1 || sse > serverSendEnd) serverSendEnd = sse;
        if (tailerResultStart == -1 || trs < tailerResultStart) tailerResultStart = trs;
        if (tailerResultEnd == -1 || tre > tailerResultEnd) tailerResultEnd = tre;
        // New: Detailed phase timestamps for each token
        clientReceiveTimes.add(new long[]{crs, cre});
        inferenceTimes.add(new long[]{is, ie});
        serverSendTimes.add(new long[]{sss, sse});
        tailerResultTimes.add(new long[]{trs, tre});
    }
}

// Global static map to record active log aggregators
private static final Map<Integer, QueryLogAggregator> activeAggregators = new ConcurrentHashMap<>();
}