/**
 * BackgroundService is the core backend service of the distributed inference system
 * Main functions:
 * 1. Device initialization and communication with server
 * 2. Model loading and preparation
 * 3. Receiving user input and performing inference
 * 4. Handling fault recovery
 * 
 * This service handles two modes:
 * - Working mode (working): Normal participation in inference calculation
 * - Active mode (active): Standby state, ready to replace faulty device
 * 
 * Device roles:
 * - Header node (header): Receives user input, processes the beginning part of the model
 * - Worker node (worker): Processes the middle layer of the model
 */
package com.example.distribute_ui.service;
import android.app.ActivityManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.IBinder;
import android.util.Log;
import androidx.annotation.Nullable;

import com.example.SecureConnection.Communication;
import com.example.SecureConnection.Config;
import com.example.SecureConnection.Dataset;
import com.example.SecureConnection.LoadBalance;
import com.example.distribute_ui.DataRepository;
import com.example.distribute_ui.Events;

import org.greenrobot.eventbus.EventBus;
import org.greenrobot.eventbus.Subscribe;
import org.greenrobot.eventbus.ThreadMode;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.Properties;
import java.text.SimpleDateFormat;
import java.util.Date;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;

public class BackgroundService extends Service {    // Inherits from Service, indicating a service
    public static double[] results;                 // Stores inference results
    public static final String TAG = "StarDust_backend";
    private String role = "worker";                 // Device role, default is "worker"
    private  String serverStatus = "active";           // Whether monitor service is needed
    private final boolean running_classification = false;   // Whether it's a classification task
    private boolean shouldStartInference = false;   // Whether inference should start
    public static boolean runningStatus = false;          // Whether it's running state
    public static boolean isScreenOff = false;          // Whether received message
    public static boolean isServiceRunning = false; // Whether service is running
    private boolean isAppInBackground = false; // Whether APP is in background
    private Thread backgroundCheckThread = null; // Background check thread
    private volatile boolean isBackgroundCheckRunning = true; // Background check thread running flag
    private Thread energyMonitorThread = null;
    private volatile boolean isEnergyMonitorRunning = true;

    private String messageContent = "";             // Stores user input message content

    // 1. New event class (suggested to be officially placed in Events.java)
    public static class ScreenOffEvent {
        private final boolean isScreenOff;
        public ScreenOffEvent(boolean isScreenOff) {
            this.isScreenOff = isScreenOff;
        }
        public boolean isScreenOff() { return isScreenOff; }
    }

    // 2. Add broadcast receiver related members in BackgroundService
    private android.content.BroadcastReceiver screenReceiver = null;

    // === Global member variables ===
    private String serverIP = "";
    private String deviceIP = "";
    private ZMQ.Socket logSocket = null;
    private ZContext zmqContext = null;

    // === Static member variables, save startup parameters for Communication.java access ===
    public static Intent lastStartIntent = null;
    public static int lastStartFlags = 0;
    public static int lastStartId = 0;
    public static String lastRole = "";
    public static String lastModelName = "";
    public static String lastServerIP = "";

    /**
     * Listen to RunningStatusEvent event
     * When Communication class initialization is complete, it will send this event
     * This method executes in the background thread to update the service running status
     * 
     * @param event Contains the event object with running status
     */
    @Subscribe(threadMode = ThreadMode.BACKGROUND)
    public void onRunningStatus(Events.RunningStatusEvent event){
        runningStatus = event.isRunning;
        System.out.println("Running Status is: " + runningStatus);
    }

    /**
     * Listen to messageSentEvent event
     * Triggered when user sends a message in the chat interface
     * Records message content for subsequent inference processing
     * 
     * @param event Contains the event object with message status and content
     */
    @Subscribe(threadMode = ThreadMode.BACKGROUND)
    public void onMessageSentEvent(Events.messageSentEvent event) {
        isScreenOff = event.messageSent;
        messageContent = event.messageContent;
        System.out.println("messageSent Status is: " + isScreenOff);
        System.out.println("message Content is: " + messageContent);
    }

    /**
     * Listen to enterChatEvent event
     * Triggered when user enters the chat interface
     * Used to mark whether inference process should start
     * 
     * @param event Contains the event object with entering chat status
     */
    @Subscribe(threadMode = ThreadMode.BACKGROUND)
    public void onEnterChatEvent(Events.enterChatEvent event) {
        shouldStartInference = event.enterChat;
        System.out.println("ShouldStartInference is: " + shouldStartInference);
    }

    /**
     * Get server IP address, priority: Intent extra > Config class > config.properties
     * @param intent Launch service Intent, can be null
     * @return Server IP address string
     */
    private String getServerIPAddress(Intent intent) {
        String serverIP = null;
        // 1. Priority from Intent extra
        if (intent != null && intent.hasExtra("ip")) {
            serverIP = intent.getStringExtra("ip");
            if (serverIP != null && !serverIP.isEmpty()) {
                return serverIP;
            }
        }
        // 2. Then use Config.root
        try {
            Class<?> configClass = Class.forName("com.example.SecureConnection.Config");
            serverIP = (String) configClass.getField("root").get(null);
            if (serverIP != null && !serverIP.isEmpty()) {
                return serverIP;
            }
        } catch (Exception e) {
            // Ignore, enter next fallback
        }
        // 3. Last fallback config.properties
        Properties properties = new Properties();
        try {
            InputStream inputStream = getAssets().open("config.properties");
            properties.load(inputStream);
            serverIP = properties.getProperty("server_ip");
            inputStream.close();
        } catch (IOException ioException) {
            ioException.printStackTrace();
        }
        return serverIP != null ? serverIP : "";
    }

    /**
     * Check whether the model directory is empty
     * Used to determine whether model files have been downloaded
     * 
     * @param modelPath Model file path
     * @return If directory is empty, return true; otherwise, return false
     */
    private boolean isModelDirectoryEmpty(String modelPath) {
        File modelDir = new File(modelPath + "/device");
        if (modelDir.isDirectory()) {
            String[] files = modelDir.list();
            return files == null || files.length == 0;
        }
        // Return true if it's not a directory, indicating "empty" in this context.
        return true;
    }
    // When service starts (through startService(Intent) or bind)
    /**
     * Update model directory status to data repository
     * When model is ready, notify UI to update
     * 
     * @param isDirEmpty Directory is empty
     */
    private void updateIsDirEmpty(boolean isDirEmpty) {
        // Update the repository with the new value
        DataRepository.INSTANCE.setIsDirEmpty(isDirEmpty);
    }

    /**
     * Service start callback method
     * Responsible for initializing inference environment and starting inference process
     * 
     * Overall process:
     * 1. Get device role, model, and server information
     * 2. Create configuration object and communication object
     * 3. Register and get work status (working/active) from server
     * 4. Execute different initialization processes based on status
     * 5. Wait for model to be ready
     * 6. For header node, wait for user input to start inference
     * 7. Execute actual inference task
     * 
     * @param intent Contains the Intent with startup parameters
     * @param flags Startup flag
     * @param startId Startup ID
     * @return Service start mode
     */
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {  // flags and startId (used to identify service, needed when terminating service) are automatically passed
        Log.d(TAG, "background service started");
        Log.d(TAG, "Startup parameters - intent: " + intent + ", flags: " + flags + ", startId: " + startId);
        //        Check what parameters are available
//        Then restart in another function

        // Save startup parameters to static variable values 
        lastStartIntent = intent;
        lastStartFlags = flags;
        lastStartId = startId;
        isServiceRunning = true;
        
        int id;
        if (intent != null && intent.hasExtra("role")) {
            id = intent.getIntExtra("role", 0);
        } else {
            id = 0;
        }
        if (id == 1) {  // If id is 1, change role to header node
            role = "header";
        }
        Log.d(TAG, "role is " + role);

        // Get model name
        String modelName = "";
        if (intent != null && intent.hasExtra("model")) {   // Extract the value of the extra information "model" in Intent
            modelName = intent.getStringExtra("model");     // Get model name
            System.out.println("model name is: "+ modelName);
        }

        // Get server IP, priority: Intent > Config > config.properties
        serverIP = null;
        if (intent != null && intent.hasExtra("ip")) {
            serverIP = intent.getStringExtra("ip");
        }
        if (serverIP == null || serverIP.isEmpty()) {
            try {
                Class<?> configClass = Class.forName("com.example.SecureConnection.Config");
                serverIP = (String) configClass.getField("root").get(null);
            } catch (Exception e) {
                // Ignore, enter next fallback
            }
        }
        if (serverIP == null || serverIP.isEmpty()) {
            Properties properties = new Properties();
            try {
                InputStream inputStream = getAssets().open("config.properties");
                properties.load(inputStream);
                serverIP = properties.getProperty("server_ip");
                inputStream.close();
            } catch (IOException ioException) {
                ioException.printStackTrace();
            }
        }
        if (serverIP == null) serverIP = "";
        System.out.println("root ip: "+ serverIP);

        deviceIP = Config.local;
        System.out.println("deviceIP ip: "+ deviceIP);
        // Save all parameters to static variable
        lastRole = role;
        lastModelName = modelName;
        lastServerIP = serverIP;
        
        Log.d(TAG, "Startup parameters saved - role: " + lastRole + ", model: " + lastModelName + ", serverIP: " + lastServerIP);

        // === New: Save parameters to SharedPreferences ===
        android.content.SharedPreferences prefs = getApplicationContext().getSharedPreferences("app_prefs", Context.MODE_PRIVATE);
        android.content.SharedPreferences.Editor editor = prefs.edit();
        editor.putInt("role", id);
        editor.putString("model", modelName);
        editor.putString("ip", serverIP);
        editor.putString("device_ip", deviceIP);
        editor.apply();
        // Create a single-threaded thread pool, all tasks in the pool execute in order, at most one task is executing at a time
        // Through task submission to thread pool execution, current thread can continue to execute other operations without being blocked, thread pool will automatically manage the lifecycle of worker threads
        ExecutorService executor = Executors.newSingleThreadExecutor();
        String finalModelName = modelName;  // Model name
        executor.submit(() -> {             // Submit a task to executor (lambda form)

            // k is top-k sampling parameter
            // initial_temp is temperature parameter
            // Instantiate a configuration class, server address is server_ip:23456, top-k sampling, in addition to itself ip:port
            Config cfg = new Config(serverIP, 23456, 7, 0.7f);

            Communication com = new Communication(cfg, this, finalModelName, id); // Instantiate a Communication based on configuration cfg
            Communication.loadBalance = new LoadBalance(com, cfg);  // Instantiate a LoadBalance based on com and cfg
            com.param.modelPath = getFilesDir() + "";   // Return application private file storage directory in string form
//            com.param.modelPath =  "/sdcard";
            Log.d(TAG, "Storage path is:" + com.param.modelPath);

            // 1. send IP to server to request model
            // Establish connection with server, send itself ip (for header node, also add model name), determine need_monitor as true/false based on information received from server
            if (role.equals("header")) {
                serverStatus = com.sendIPToServer(role, finalModelName); // Header node needs to provide model name
            } else {
                serverStatus = com.sendIPToServer(role, ""); // Worker node does not need to provide model name
            }
            Log.d(TAG, "serverStatus = " + serverStatus);

            // 2. Initiate device monitor for server-side optimization
            // If need_monitor is true, send broadcast with action "START_MONITOR",
            // Receiver in MainActivity will start MonitorService and append role information when receiving this broadcast
//            if (need_monitor) {
//                Intent broadcastIntent = new Intent();
//                broadcastIntent.setAction("START_MONITOR"); // Set broadcast "action"
//                LocalBroadcastManager.getInstance(this).sendBroadcast(broadcastIntent);
//                sendBroadcast(broadcastIntent);
//                Log.d(TAG, "broadcast sent by backgroundService");
//            }
            if (serverStatus.equals("working")){
                Log.d(TAG, "serverStatus :working ");
//               Initialization phase
//                1. Transmission control signal 34567
//                    1.1 Ready->Open->Prepare->Initialized->Start->Running
//
                // 3.1 start downloading required model and tokenizer files from server
                // Execute Client.communicationOpenClose code corresponding to param.status.equals("Ready"), including preparing model files and tokenizer from initialization work
                com.runPrepareThread(serverStatus);

            }

            if (serverStatus.equals("active")){
                Log.d(TAG, "serverStatus :active ");
                // 3.1 start downloading required model and tokenizer files from server
                // Execute Client.communicationOpenClose code corresponding to param.status.equals("Ready"), including preparing model files and tokenizer from initialization work
                com.runPrepareThread(serverStatus);
//               Running phase
//                1. Transmission control signal 34567
//                    1.1 Ready->Open->Prepare->Initialized up to here but do not start inference
//                    Phone 1 enters fault recovery function
//                 Find communication IP diagram (config["graph"],
//                            config["session_index"],）
//                            receiveIPGraph(cfg, receiver); -> Config.buildCommunicationGraph()
//                            receiveSessionIndex(receiver);
//                       And registration IP diagram location, Communication.updateSockets
//
//                   Re-communicate IP diagram, start phone 3
//                   Re-register IP diagram, start phone 1
//                   Recover inference: phone 1 communicates to phone 3 based on IP diagram



            }





            // 3.2 Check whether the model file exists
            // When param.status == "Running" you will receive event RunningStatusEvent->runningStatus=true
            // Then check whether model file is ready

            while (!runningStatus) {
                try {
                    Thread.sleep(1000); // Sleep for a short duration to avoid busy waiting
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt(); // Restore the interrupted status
                    break; // Exit the loop if the thread is interrupted
                }
            }
            
            // Check whether model file is ready
            boolean isDirEmpty = isModelDirectoryEmpty(com.param.modelPath);
            Log.d(TAG, "check the direction is empty: " + isDirEmpty);
            if (runningStatus && !isDirEmpty){
                System.out.println("Prepare is Finished.");
                // If header node, update DataRepository isDirEmptyLiveData value->ModelScreen ConfirmButton clickable->Send event enterChatEvent
                // -> shouldStartInference=true
                if (cfg.isHeader()){
                    updateIsDirEmpty(isDirEmpty);
                }
            }

            // For header node, wait for user to confirm start inference
            // When user clicks start inference button, enterChatEvent event will be sent, shouldStartInference set to true
            if (cfg.isHeader()) {
                while (!shouldStartInference) {
                    try {
                        Thread.sleep(1000); // Sleep for a short duration to avoid busy waiting
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt(); // Restore the interrupted status
                        break; // Exit the loop if the thread is interrupted
                    }
                }
            }

            // Header node inference process
            if (shouldStartInference && cfg.isHeader()){
                // Set classification label
                com.param.classes = new String[]{"Negative", "Positive"};
                // 4.2 Dataset would be used if we need conduct evaluation experiment
                Dataset dataset = null;

                // Wait until numSample > 0
                while (com.param.numSample <= 0)
                    Thread.sleep(1000);

                System.out.println("batch size is: " + com.param.numSample);
                // 4.3 Create input string array to store user input query. By default, the array size
                // is set to 1 for testing single-turn chat conversation.

                // 4.4 Based on whether user give input to run the inference
                ArrayList<String> test_input = new ArrayList<>();

                // 4.4.1 Receive userinput from chatscreen and save it to test_input array
                // Wait until user presses send button->Send event messageSentEvent->messageStatus=true
                while (!isScreenOff) {
                    try {
                        Thread.sleep(1000); // Sleep for a short duration to avoid busy waiting
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt(); // Restore the interrupted status
                        break; // Exit the loop if the thread is interrupted
                    }
                }

                // Create thread to handle user input
                if (cfg.isHeader()) {
                    new Thread(() -> {
                        int j = 0;  // Record current batch number
                        String userinput = "";
                        // Automatic input related variables
                        // int autoInputCount = 1;
                        // boolean autoInputEnabled = true; // Can add switch
                        while (j < com.param.numSample) {           // Execute numSample(BatchSize) times
                            // Check for new user input
                            if (messageContent.equals(userinput)){
                                try {
                                    Thread.sleep(1000);
                                } catch (InterruptedException e) {
                                    throw new RuntimeException(e);
                                }
                            } else {
                                // Receive new message, process and add to input list
                                System.out.println("current numSample:" + j + ", New prompt:" + messageContent);
                                Log.d(TAG, "[AutoInput] New message detected from user input: " + messageContent);
                                userinput = messageContent;
                                test_input.add(userinput);      // Add prompt to list
                                Log.d(TAG, "[AutoInput] test_input updated, current size: " + test_input.size());
                                j++;                            // Increment current batch count
                            }

                            // // Automatic input logic
                            // if (autoInputEnabled && j < com.param.numSample) {
                            //     // Check whether inference thread is waiting for input
                            //     Log.d(TAG, "In autoInputEnabled");
                            //     if (com.sampleId >= test_input.size()) {
                            //         Log.d(TAG, "[AutoInput] Detection inference thread waiting for input, sampleId=" + com.sampleId + ", test_input.size=" + test_input.size());
                            //         // Generate automatic input content
                            //         String autoMsg = "Simulated user input" + autoInputCount;
                            //         autoInputCount++;
                            //         // Send event, simulated user input
                            //         messageContent = autoMsg;
                            //         EventBus.getDefault().post(new Events.messageSentEvent(true, autoMsg));
                            //         Log.d(TAG, "[AutoInput] Automatic input sent: " + autoMsg);
                            //         // Wait for inference thread to process current input
                            //         while (com.sampleId < test_input.size()) {
                            //             Log.d(TAG, "[AutoInput] Waiting for inference thread to process automatic input, sampleId=" + com.sampleId + ", test_input.size=" + test_input.size());
                            //             try {
                            //                 Thread.sleep(500);
                            //             } catch (InterruptedException e) {
                            //                 throw new RuntimeException(e);
                            //             }
                            //         }
                            //         // Wait 10 seconds before automatically inputting next
                            //         Log.d(TAG, "[AutoInput] Inference completed, waiting 10 seconds before preparing next automatic input");
                            //         try {
                            //             Thread.sleep(10000);
                            //         } catch (InterruptedException e) {
                            //             throw new RuntimeException(e);
                            //         }
                            //     }
                            // }
                        }
                    }).start();
                }
                
                // Set thread pool parameters and start inference
                int corePoolSize = 2;      // Core thread count
                int maximumPoolSize = 2;   // Maximum thread count
                int keepAliveTime = 500;   // Thread idle timeout
                try {
                    Log.w(TAG, "onStartCommand inside communication starts to running");
                    // Start actual inference task, pass thread pool parameters and input data, this is not a thread, it's directly started, but wrapped in ExecutorService executor
                    com.running(corePoolSize, maximumPoolSize, keepAliveTime, test_input);
                } catch (IOException | InterruptedException e) {
                    throw new RuntimeException(e);
                }
                double startTime = System.nanoTime();
                results = com.timeUsage;   // Save time statistics results

                Log.d(TAG, "Results Computation Time: " + (System.nanoTime() - startTime) / 1000000000.0);
                return null;
            }

            // Non-header node inference process
            // Worker node does not need user input, directly execute inference task
            else if (!shouldStartInference && !cfg.isHeader()){
                com.param.classes = new String[]{"Negative", "Positive"};
                Dataset dataset = null;
                // Wait for batch processing size setting to complete
                while (com.param.numSample <= 0)
                    Thread.sleep(1000);
                
                // Worker node does not need actual input data, but needs to provide an empty list
                ArrayList<String> test_input = new ArrayList<>();
                int corePoolSize = 2;
                int maximumPoolSize = 2;
                int keepAliveTime = 500;

                try {

                    // Start inference task
                    Log.w(TAG, "onStartCommand inside communication starts to running");
                    com.running(corePoolSize, maximumPoolSize, keepAliveTime, test_input);
                } catch (IOException | InterruptedException e) {
                    throw new RuntimeException(e);
                }
                results = com.timeUsage;
                return null;
            }
            return null;
        });

        // Initialize ZeroMQ log socket
        if (zmqContext == null) {
            zmqContext = new ZContext();
        }
        if (logSocket == null) {
            logSocket = zmqContext.createSocket(SocketType.DEALER);
            String connectStr = "tcp://" + serverIP + ":9889";
            logSocket.connect(connectStr);
            Log.d(TAG, "Log socket connected to " + connectStr);
        }

        return START_STICKY; // If system kills service, will attempt to restart and restore Intent
    }

    /**
     * Service bind callback
     * This service does not support binding, return null
     */
    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
    
    /**
     * Service create callback
     * Register EventBus event listener
     */
    @Override
    public void onCreate() {
        super.onCreate();
        isServiceRunning = true;
        EventBus.getDefault().register(this);  // Register event bus listener
        startBackgroundCheck(); // Start background check
        startEnergyMonitor(); // Start power consumption collection
        // Register screen off/screen on broadcast
        registerScreenReceiver();
        Log.d(TAG, "onCreate");
    }

    /**
     * Service destroy callback
     * Cancel EventBus event listener
     */
    @Override
    public void onDestroy() {
        super.onDestroy();
        isServiceRunning = false;
//        stopBackgroundCheck(); // Stop background check
//        stopEnergyMonitor(); // Stop power consumption collection
//        EventBus.getDefault().unregister(this);  // Cancel event bus listener
//        // Unregister screen off/screen on broadcast
//        unregisterScreenReceiver();
    }

    /**
     * Start background check thread
     * Periodically check whether APP is running in the background
     */
    private void startBackgroundCheck() {
        backgroundCheckThread = new Thread(() -> {
            while (isBackgroundCheckRunning) {
                try {
                    // Check whether APP is in the background
                    boolean isBackground = !isAppInForeground();
                    
                    // If status changes, send event
                    if (isBackground != isAppInBackground) {
                        isAppInBackground = isBackground;
                        EventBus.getDefault().post(new Events.AppBackgroundEvent(isBackground));
                        Log.d(TAG, "App background status changed: " + (isBackground ? "in background" : "in foreground"));
                    }
                    
                    // Check every second
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Log.e(TAG, "Background check thread interrupted: " + e.getMessage());
                    break;
                } catch (Exception e) {
                    Log.e(TAG, "Error in background check: " + e.getMessage());
                }
            }
        });
        backgroundCheckThread.setDaemon(true);
        backgroundCheckThread.start();
        Log.d(TAG, "Background check thread started");
    }

    /**
     * Stop background check thread
     */
    private void stopBackgroundCheck() {
        isBackgroundCheckRunning = false;
        if (backgroundCheckThread != null) {
            backgroundCheckThread.interrupt();
            backgroundCheckThread = null;
        }
        Log.d(TAG, "Background check thread stopped");
    }

    /**
     * Check whether APP is running in the foreground
     * @return true if app is in foreground, false otherwise
     */
    private boolean isAppInForeground() {
        ActivityManager activityManager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
        if (activityManager == null) return false;

        List<ActivityManager.RunningAppProcessInfo> appProcesses = activityManager.getRunningAppProcesses();
        if (appProcesses == null) return false;

        String packageName = getPackageName();
        for (ActivityManager.RunningAppProcessInfo appProcess : appProcesses) {
            if (appProcess.importance == ActivityManager.RunningAppProcessInfo.IMPORTANCE_FOREGROUND
                    && appProcess.processName.equals(packageName)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Listen to APP background status change event
     */
    @Subscribe(threadMode = ThreadMode.BACKGROUND)
    public void onAppBackgroundEvent(Events.AppBackgroundEvent event) {
        if (event.isInBackground()) {
            Log.d(TAG, "App entered background, taking appropriate actions");
            // Add operations needed when APP enters background
            // For example: Pause certain operations, save state, etc.
        } else {
            Log.d(TAG, "App entered foreground, resuming normal operations");
            // Add operations needed when APP enters foreground
            // For example: Restore paused operations, update UI, etc.
        }
    }

    /**
     * Handle get background status event
     */
    @Subscribe(threadMode = ThreadMode.BACKGROUND)
    public void onGetBackgroundStatus(Events.GetBackgroundStatusEvent event) {
        event.setInBackground(isAppInBackground);
    }

    /**
     * Start power consumption collection thread
     */
    private void startEnergyMonitor() {
        energyMonitorThread = new Thread(() -> {
            while (isEnergyMonitorRunning) {
                try {
                    // Collect power consumption data
                    int battery = getBatteryLevel();
                    double cpuUsage = getCpuUsage();
                    double temperature = getDeviceTemperature();
                    long timestamp = System.currentTimeMillis();
                    String deviceId = android.os.Build.SERIAL;
                    String roleStr = role;
                    // Send power consumption event
                    EventBus.getDefault().post(new com.example.distribute_ui.Events.EnergyEvent(deviceId, roleStr, timestamp, battery, cpuUsage, temperature));
                    Thread.sleep(10000); // Collect every 10 seconds
                } catch (InterruptedException e) {
                    break;
                }
            }
        });
        energyMonitorThread.setDaemon(true);
        energyMonitorThread.start();
    }

    private void stopEnergyMonitor() {
        isEnergyMonitorRunning = false;
        if (energyMonitorThread != null) {
            energyMonitorThread.interrupt();
            energyMonitorThread = null;
        }
    }

    // Get battery percentage
    private int getBatteryLevel() {
        android.os.BatteryManager bm = (android.os.BatteryManager) getSystemService(BATTERY_SERVICE);
        if (bm != null) {
            int level = bm.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY);
            return level; // 0-100
        }
        return -1; // Get failed
    }
    // Get CPU usage (system-wide)
    private double getCpuUsage() {
        // Android 10+ cannot access /proc/stat, directly return 0.0
        if (android.os.Build.VERSION.SDK_INT >= 29) {
            Log.w(TAG, "getCpuUsage: Current system does not support CPU usage collection, returning 0.0");
            return 0.0;
        }
        try {
            java.io.RandomAccessFile reader = new java.io.RandomAccessFile("/proc/stat", "r");
            String load = reader.readLine();
            String[] toks = load.split(" +"); // Multiple spaces split
            long idle1 = Long.parseLong(toks[4]);
            long cpu1 = 0;
            for (int i = 1; i < 8; i++) {
                cpu1 += Long.parseLong(toks[i]);
            }
            Thread.sleep(360);
            reader.seek(0);
            load = reader.readLine();
            reader.close();
            toks = load.split(" +");
            long idle2 = Long.parseLong(toks[4]);
            long cpu2 = 0;
            for (int i = 1; i < 8; i++) {
                cpu2 += Long.parseLong(toks[i]);
            }
            return (double) (cpu2 - cpu1 - (idle2 - idle1)) / (cpu2 - cpu1);
        } catch (Exception e) {
            Log.w(TAG, "getCpuUsage: Collection failed, returning 0.0");
            return 0.0;
        }
    }
    // Get device temperature (battery temperature)
    private double getDeviceTemperature() {
        android.content.Intent intent = registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        if (intent != null) {
            int temp = intent.getIntExtra(android.os.BatteryManager.EXTRA_TEMPERATURE, -1);
            if (temp != -1) {
                return temp / 10.0; // Unit is 0.1°C
            }
        }
        return -1;
    }

    // Listen to SessionLogEvent
    @Subscribe(threadMode = ThreadMode.BACKGROUND)
    public void onSessionLogEvent(com.example.distribute_ui.Events.SessionLogEvent event) {
        sendLogToServer(event);
    }

    // Log sent to server (currently only print log, can be extended to send)
    private void sendLogToServer(Object logEvent) {
        if (logEvent instanceof com.example.distribute_ui.Events.SessionLogEvent) {
            com.example.distribute_ui.Events.SessionLogEvent sessionLog = (com.example.distribute_ui.Events.SessionLogEvent) logEvent;
            StringBuilder sb = new StringBuilder();
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
            sb.append("\n==== SessionLogEvent ====");
            // Print QueryLogEvent
            com.example.distribute_ui.Events.QueryLogEvent q = sessionLog.queryLog;
            sb.append("\n[QueryLog] id:").append(q.queryId)
              .append(", userQuery:").append(q.userQuery)
              .append(", response:").append(q.response)
              .append("\n tokens:").append(q.tokens)
              .append(", throughput:").append(q.throughput);
            // Print detailed stage timestamp for each token
            int tokenCount = Math.max(
                Math.max(
                    q.clientReceiveTimes != null ? q.clientReceiveTimes.size() : 0,
                    q.inferenceTimes != null ? q.inferenceTimes.size() : 0
                ),
                Math.max(
                    q.serverSendTimes != null ? q.serverSendTimes.size() : 0,
                    q.tailerResultTimes != null ? q.tailerResultTimes.size() : 0
                )
            );
            sb.append("\n[TokenStageTimes] count:").append(tokenCount);
            for (int i = 0; i < tokenCount; i++) {
                StringBuilder sbToken = new StringBuilder();
                sbToken.append("token[").append(i).append("] ");
                // clientReceive
                if (q.clientReceiveTimes != null && i < q.clientReceiveTimes.size()) {
                    long[] t = q.clientReceiveTimes.get(i);
                    sbToken.append("clientReceive:")
                        .append(sdf.format(new Date(t[0]))).append("-")
                        .append(sdf.format(new Date(t[1]))).append(", ");
                }
                // inference
                if (q.inferenceTimes != null && i < q.inferenceTimes.size()) {
                    long[] t = q.inferenceTimes.get(i);
                    sbToken.append("inference:")
                        .append(sdf.format(new Date(t[0]))).append("-")
                        .append(sdf.format(new Date(t[1]))).append(", ");
                }
                // serverSend
                if (q.serverSendTimes != null && i < q.serverSendTimes.size()) {
                    long[] t = q.serverSendTimes.get(i);
                    sbToken.append("serverSend:")
                        .append(sdf.format(new Date(t[0]))).append("-")
                        .append(sdf.format(new Date(t[1]))).append(", ");
                }
                // tailerResult
                if (q.tailerResultTimes != null && i < q.tailerResultTimes.size()) {
                    long[] t = q.tailerResultTimes.get(i);
                    sbToken.append("tailerResult:")
                        .append(sdf.format(new Date(t[0]))).append("-")
                        .append(sdf.format(new Date(t[1])));
                }
                sb.append(sbToken.toString());
            }
            // Print FaultEvent
            sb.append("\n[FaultEvents] count:").append(sessionLog.faultEvents.size());
            for (com.example.distribute_ui.Events.FaultEvent f : sessionLog.faultEvents) {
                sb.append("\n  type:").append(f.faultType)
                  .append(", time:").append(f.faultTime > 0 ? sdf.format(new Date(f.faultTime)) : "-")
                  .append(", recovery:").append(f.recoveryTime > 0 ? sdf.format(new Date(f.recoveryTime)) : "-")
                  .append(", affectedQueryId:").append(f.affectedQueryId);
            }
            // Print EnergyEvent
            sb.append("\n[EnergyEvents] count:").append(sessionLog.energyEvents.size());
            for (com.example.distribute_ui.Events.EnergyEvent e : sessionLog.energyEvents) {
                sb.append("\n  time:").append(e.timestamp > 0 ? sdf.format(new Date(e.timestamp)) : "-")
                  .append(", battery:").append(e.battery)
                  .append(", cpu:").append(e.cpuUsage)
                  .append(", temp:").append(e.temperature);
            }
            sb.append("\n========================");
            Log.d(TAG, sb.toString());

            // New: Send log to Python server (ZeroMQ method)
            try {
                JSONObject json = new JSONObject();
                // Device IP
                json.put("deviceIP", deviceIP != null ? deviceIP :Config.local);

                // QueryLogEvent
                JSONObject queryLogJson = new JSONObject();
                queryLogJson.put("queryId", q.queryId);
                queryLogJson.put("userQuery", q.userQuery);
                queryLogJson.put("response", q.response);
                queryLogJson.put("tokens", q.tokens);
                queryLogJson.put("throughput", q.throughput);
                // token stage timestamp (all converted to string and merged into one line)
                JSONArray tokenStageTimes = new JSONArray();
                tokenCount = Math.max(
                    Math.max(
                        q.clientReceiveTimes != null ? q.clientReceiveTimes.size() : 0,
                        q.inferenceTimes != null ? q.inferenceTimes.size() : 0
                    ),
                    Math.max(
                        q.serverSendTimes != null ? q.serverSendTimes.size() : 0,
                        q.tailerResultTimes != null ? q.tailerResultTimes.size() : 0
                    )
                );
                for (int i = 0; i < tokenCount; i++) {
                    StringBuilder sbToken = new StringBuilder();
                    sbToken.append("token[").append(i).append("] ");
                    // clientReceive
                    if (q.clientReceiveTimes != null && i < q.clientReceiveTimes.size()) {
                        long[] t = q.clientReceiveTimes.get(i);
                        sbToken.append("clientReceive:")
                            .append(sdf.format(new Date(t[0]))).append("-")
                            .append(sdf.format(new Date(t[1]))).append(", ");
                    }
                    // inference
                    if (q.inferenceTimes != null && i < q.inferenceTimes.size()) {
                        long[] t = q.inferenceTimes.get(i);
                        sbToken.append("inference:")
                            .append(sdf.format(new Date(t[0]))).append("-")
                            .append(sdf.format(new Date(t[1]))).append(", ");
                    }
                    // serverSend
                    if (q.serverSendTimes != null && i < q.serverSendTimes.size()) {
                        long[] t = q.serverSendTimes.get(i);
                        sbToken.append("serverSend:")
                            .append(sdf.format(new Date(t[0]))).append("-")
                            .append(sdf.format(new Date(t[1]))).append(", ");
                    }
                    // tailerResult
                    if (q.tailerResultTimes != null && i < q.tailerResultTimes.size()) {
                        long[] t = q.tailerResultTimes.get(i);
                        sbToken.append("tailerResult:")
                            .append(sdf.format(new Date(t[0]))).append("-")
                            .append(sdf.format(new Date(t[1])));
                    }
                    tokenStageTimes.put(sbToken.toString());
                }
                queryLogJson.put("tokenStageTimes", tokenStageTimes);
                json.put("queryLog", queryLogJson);
                // FaultEvents
                JSONArray faultEventsJson = new JSONArray();
                for (com.example.distribute_ui.Events.FaultEvent f : sessionLog.faultEvents) {
                    JSONObject fJson = new JSONObject();
                    fJson.put("faultType", f.faultType);
                    fJson.put("faultTime", f.faultTime);
                    fJson.put("recoveryTime", f.recoveryTime);
                    fJson.put("affectedQueryId", f.affectedQueryId);
                    faultEventsJson.put(fJson);
                }
                json.put("faultEvents", faultEventsJson);
                // EnergyEvents
                JSONArray energyEventsJson = new JSONArray();
                for (com.example.distribute_ui.Events.EnergyEvent e : sessionLog.energyEvents) {
                    JSONObject eJson = new JSONObject();
                    eJson.put("timestamp", e.timestamp);
                    eJson.put("battery", e.battery);
                    eJson.put("cpuUsage", e.cpuUsage);
                    eJson.put("temperature", e.temperature);
                    energyEventsJson.put(eJson);
                }
                json.put("energyEvents", energyEventsJson);

                // New: Send time span (first token and last token's clientReceive[0] time)
                String timeSpan = "";
                if (q.clientReceiveTimes != null && q.clientReceiveTimes.size() > 0) {
                    long first = q.clientReceiveTimes.get(0)[0];
                    long last = q.clientReceiveTimes.get(q.clientReceiveTimes.size() - 1)[0];
                    timeSpan = sdf.format(new Date(first)) + "~" + sdf.format(new Date(last));
                }
                json.put("timeSpan", timeSpan);

                // Send log to Python server (ZeroMQ method)
                if (logSocket == null) {
                    Log.e(TAG, "Log socket not initialized");
                    return;
                }
                String jsonStr = json.toString();
                logSocket.send(jsonStr);
                Log.d(TAG, "Log sent to server via ZeroMQ");
            } catch (Exception e) {
                Log.e(TAG, "Failed to send log to server: " + e.getMessage());
            }
        } else {
            Log.d(TAG, "Send log to server: " + logEvent.toString());
        }
    }



    // 3. Register/unregister broadcast method
    private void registerScreenReceiver() {
        if (screenReceiver == null) {
            screenReceiver = new android.content.BroadcastReceiver() {
                @Override
                public void onReceive(Context context, Intent intent) {
                    String action = intent.getAction();
                    if (Intent.ACTION_SCREEN_OFF.equals(action)) {
                        isScreenOff = true;
                        EventBus.getDefault().post(new ScreenOffEvent(true));
                        Log.d(TAG, "Screen turned off");
                    } else if (Intent.ACTION_SCREEN_ON.equals(action)) {
                        isScreenOff = false;
                        EventBus.getDefault().post(new ScreenOffEvent(false));
                        Log.d(TAG, "Screen turned on");
                    }
                }
            };
            IntentFilter filter = new IntentFilter();
            filter.addAction(Intent.ACTION_SCREEN_OFF);
            filter.addAction(Intent.ACTION_SCREEN_ON);
            registerReceiver(screenReceiver, filter);
            Log.d(TAG, "Screen receiver registered");
        }
    }
    private void unregisterScreenReceiver() {
        if (screenReceiver != null) {
            unregisterReceiver(screenReceiver);
            screenReceiver = null;
            Log.d(TAG, "Screen receiver unregistered");
        }
    }

    // 4. Subscribe screen off event
    @org.greenrobot.eventbus.Subscribe(threadMode = org.greenrobot.eventbus.ThreadMode.BACKGROUND)
    public void onScreenOffEvent(ScreenOffEvent event) {
        if (event.isScreenOff()) {
            Log.d(TAG, "Screen is off, take appropriate actions");
            // Here you can add business logic when screen is off
        } else {
            Log.d(TAG, "Screen is on, resume normal operations");
            // Here you can add business logic when screen is on
        }
    }
}
