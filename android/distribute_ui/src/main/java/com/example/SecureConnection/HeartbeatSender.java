package com.example.SecureConnection;

import org.zeromq.ZContext;
import org.zeromq.ZMQ;
import org.json.JSONObject;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import com.example.distribute_ui.service.MonitorServiceKt;
import com.example.SecureConnection.Config;
public class HeartbeatSender {

    private static final String SERVER_HOST = "tcp://127.0.0.1:34567";  // Server IP address and port
    private static final long HEARTBEAT_INTERVAL = 5000;  // Send heartbeat every 5 seconds

    private static ZMQ.Socket monitorSocket;
    // Create Config instance
    static Config config = new Config("127.0.0.1", 8080);

    // Call getCurrentDeviceIP() method
    static String deviceIP = config.getCurrentDeviceIP();



    public static void main(String[] args) {
        // Start heartbeat sending function
        heartrun();  // Call heartrun method to start sending heartbeats
    }

    public static void heartrun() {
        // Create ZeroMQ context
        ZContext context = new ZContext();

        // Create DEALER socket
        monitorSocket = context.createSocket(ZMQ.DEALER);
        monitorSocket.connect(SERVER_HOST);  // Connect to the specified server

        // Create a scheduled task executor
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

        // Schedule heartbeat sending
        scheduler.scheduleWithFixedDelay(new Runnable() {
            @Override
            public void run() {
                sendHeartbeat();
            }
        }, 0, HEARTBEAT_INTERVAL, TimeUnit.MILLISECONDS);  // Send every 5 seconds
    }

    public static void sendHeartbeat() {
        try {
            // Create a JSON object to store heartbeat packet information
            JSONObject jsonObject = new JSONObject();
            String currentIP =  deviceIP;  // Get local IP address (replace as needed)
            jsonObject.put("ip", currentIP);
            jsonObject.put("role", "client");  // Set role

            // Convert JSON to string and send
            String heartbeatMessage = jsonObject.toString();
            monitorSocket.send(heartbeatMessage.getBytes(StandardCharsets.UTF_8));

            System.out.println("Sent heartbeat: " + heartbeatMessage);
        } catch (org.json.JSONException e) {
            // Catch and handle JSON exception
            System.err.println("Error creating JSON object: " + e.getMessage());
        } catch (Exception e) {
            // Catch other exceptions
            System.err.println("Error sending heartbeat: " + e.getMessage());
        }
    }
}