package com.example.distribute_ui;

import java.util.List;

// Contains multiple static classes representing different types of events (event objects), used for data transfer and communication within the application
// This event-driven design is used to achieve decoupling, allowing different components to pass data through events without directly depending on each other
public class Events {
    // Event for registering the communication running status for service communication
    // Used to record whether communication with the server is running
    public static class RunningStatusEvent{
        public final boolean isRunning; // Indicates whether the server is running
        public RunningStatusEvent(boolean isRunning){
            this.isRunning = isRunning;
        }   // Constructor of the class
    }

    // Event for UI-service communication to let the background service know the inference chat can initiate
    // Used to send a message to the background service, requesting the inference service
    public static class messageSentEvent{
        public final boolean messageSent;   // Records whether the message has been sent
        public final String messageContent; // Stores the content of the sent message
        public messageSentEvent(boolean messageSent, String messageContent){
            this.messageSent = messageSent;
            this.messageContent = messageContent;
        }
    }

    // Used to indicate whether to enter the chat interface
    public static class enterChatEvent{
        public final boolean enterChat;     // Whether to enter the chat interface
        public enterChatEvent(boolean enterChat){
            this.enterChat = enterChat;
        }
    }

    // Used to pass sampleId
    public static class sampleIdEvent{
        public final int sampleId;          // sampleId
        public sampleIdEvent(int sampleId){
            this.sampleId = sampleId;
        }
    }

    public static class AppBackgroundEvent {
        private final boolean isInBackground;

        public AppBackgroundEvent(boolean isInBackground) {
            this.isInBackground = isInBackground;
        }

        public boolean isInBackground() {
            return isInBackground;
        }
    }

    public static class GetBackgroundStatusEvent {
        private boolean isInBackground;

        public GetBackgroundStatusEvent() {
            this.isInBackground = false;
        }

        public void setInBackground(boolean inBackground) {
            isInBackground = inBackground;
        }

        public boolean isInBackground() {
            return isInBackground;
        }
    }

    // User interaction log event
    public static class QueryLogEvent {
        public String deviceId;
        public String role;
        public String queryId;
        public String userQuery;
        public String response;
        // Four-stage detailed timestamps
        public long clientReceiveStart;
        public long clientReceiveEnd;
        public long inferenceStart;
        public long inferenceEnd;
        public long serverSendStart;
        public long serverSendEnd;
        public long tailerResultStart;
        public long tailerResultEnd;
        public int tokens;
        public double throughput;
        // Fault related
        public boolean hasFault;
        public long faultStartTime;
        public long faultRecoveryTime;
        // New: detailed stage timestamps for each token
        public java.util.List<long[]> clientReceiveTimes;
        public java.util.List<long[]> inferenceTimes;
        public java.util.List<long[]> serverSendTimes;
        public java.util.List<long[]> tailerResultTimes;
        public QueryLogEvent(String deviceId, String role, String queryId, String userQuery, String response,
                             long clientReceiveStart, long clientReceiveEnd,
                             long inferenceStart, long inferenceEnd,
                             long serverSendStart, long serverSendEnd,
                             long tailerResultStart, long tailerResultEnd,
                             int tokens, double throughput,
                             boolean hasFault, long faultStartTime, long faultRecoveryTime,
                             java.util.List<long[]> clientReceiveTimes,
                             java.util.List<long[]> inferenceTimes,
                             java.util.List<long[]> serverSendTimes,
                             java.util.List<long[]> tailerResultTimes) {
            this.deviceId = deviceId;
            this.role = role;
            this.queryId = queryId;
            this.userQuery = userQuery;
            this.response = response;
            this.clientReceiveStart = clientReceiveStart;
            this.clientReceiveEnd = clientReceiveEnd;
            this.inferenceStart = inferenceStart;
            this.inferenceEnd = inferenceEnd;
            this.serverSendStart = serverSendStart;
            this.serverSendEnd = serverSendEnd;
            this.tailerResultStart = tailerResultStart;
            this.tailerResultEnd = tailerResultEnd;
            this.tokens = tokens;
            this.throughput = throughput;
            this.hasFault = hasFault;
            this.faultStartTime = faultStartTime;
            this.faultRecoveryTime = faultRecoveryTime;
            this.clientReceiveTimes = clientReceiveTimes;
            this.inferenceTimes = inferenceTimes;
            this.serverSendTimes = serverSendTimes;
            this.tailerResultTimes = tailerResultTimes;
        }
    }

    // Fault log event
    public static class FaultEvent {
        public String deviceId;
        public String role;
        public String faultType;
        public long faultTime;
        public long recoveryTime;
        public String affectedQueryId;
        public FaultEvent(String deviceId, String role, String faultType, long faultTime, long recoveryTime, String affectedQueryId) {
            this.deviceId = deviceId;
            this.role = role;
            this.faultType = faultType;
            this.faultTime = faultTime;
            this.recoveryTime = recoveryTime;
            this.affectedQueryId = affectedQueryId;
        }
    }

    // Energy consumption log event
    public static class EnergyEvent {
        public String deviceId;
        public String role;
        public long timestamp;
        public int battery;
        public double cpuUsage;
        public double temperature;
        public EnergyEvent(String deviceId, String role, long timestamp, int battery, double cpuUsage, double temperature) {
            this.deviceId = deviceId;
            this.role = role;
            this.timestamp = timestamp;
            this.battery = battery;
            this.cpuUsage = cpuUsage;
            this.temperature = temperature;
        }
    }

    // Composite log event: all logs of one round of conversation are packaged
    public static class SessionLogEvent {
        public QueryLogEvent queryLog;
        public List<FaultEvent> faultEvents;
        public List<EnergyEvent> energyEvents;
        public SessionLogEvent(QueryLogEvent queryLog, List<FaultEvent> faultEvents, List<EnergyEvent> energyEvents) {
            this.queryLog = queryLog;
            this.faultEvents = faultEvents;
            this.energyEvents = energyEvents;
        }
    }
}
