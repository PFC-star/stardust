package com.example.SecureConnection;

import static com.example.distribute_ui.service.BackgroundService.TAG;

import android.util.Log;

import org.zeromq.SocketType;
import org.zeromq.ZMQ.Socket;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;
import org.zeromq.ZMQException;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.CountDownLatch;

public class  Server {
    /***
     On behalf of the server, communicate with the client devices.
     ***/

    public Map<String, Socket> nextNodes;

    public Server() {}

    public Socket establish_connection(ZContext context, SocketType type, int port) {
        try (java.net.ServerSocket testSocket = new java.net.ServerSocket(port)) {
            testSocket.close();
        } catch (IOException e) {
            Log.e(TAG, "Port " + port + " is already in use: " + e.getMessage());
            throw new ZMQException("Port " + port + " is already in use", ZMQ.Error.EADDRINUSE.getCode());
        }
        Socket socket = context.createSocket(type);


        // Check if the port is occupied
        try {
            socket.bind("tcp://*:" + port);
            Log.d(TAG, "Successfully bound to port " + port);
        } catch (ZMQException e) {
            Log.e(TAG, "Port " + port + " binding failed: " + e.getMessage() + ", error code: " + e.getErrorCode());
            if (e.getErrorCode() == ZMQ.Error.EADDRINUSE.getCode()) {
                Log.e(TAG, "Port " + port + " is already in use, unable to bind");
            }
            // Rethrow the exception to maintain original behavior
            throw e;
        } catch (Exception e) {
            Log.e(TAG, "Unknown exception occurred while binding to port " + port + ": " + e.getMessage());
            throw e;
        }
        

        socket.setIdentity(Config.local.getBytes());

        return socket;
    }


//    class CommunicationAsServer implements Runnable {
//        private String targetIP;
//        private Socket sender;
//        private final CountDownLatch countDown;
//
//        CommunicationAsServer(String targetIp, Socket sender, CountDownLatch countDown) {
//            this.targetIP = targetIP;
//            this.sender = sender;
//            this.countDown = countDown;
//        }
//
//        @Override
//        public void run() {
//            try {
////                String msg = new String(sender.recv(0));
//                ZMsg receivedMsg = ZMsg.recvMsg(sender);
//                // Get the identity frame (client ID)
//                ZFrame identity = receivedMsg.unwrap();
//                String msg = receivedMsg.getLast().toString();
//
//                if (msg.contains("Request data")) {
//                    System.out.println(param.choice);
//                    ZMsg replyMsg = new ZMsg();
//                    replyMsg.wrap(identity.duplicate());
//                    countDown.await();
//                    replyMsg.add(OutputData.get(param.choice));
//                    replyMsg.send(sender);
//                }
//            } catch (InterruptedException e) {
//                throw new RuntimeException(e);
//            }
//
//        }
//    }
}