/**
 * 建议添加到Android客户端的故障恢复握手处理代码
 * 应该添加在handleSystemFailure()方法的开头部分
 */

public void handleSystemFailure() {
    // 如果已经在故障处理或恢复中，避免重复处理
    Log.d(TAG, "Entering system failure handling procedure");
    if ("Failure".equals(param.status) || "Recovering".equals(param.status)) {
        Log.d(TAG, "System already in failure/recovery state, ignoring duplicate recovery trigger");
        return;
    }
    
    commuSocket.setReceiveTimeOut(30000);  // 30秒超时
    Log.w(TAG, "System failure handling initiated");
    param.status = "Recovering"; // 更改为恢复中状态，区别于完全故障
    Log.d(TAG, "param status: " +  param.status);
    
    // ======== 新增握手协议代码 ========
    try {
        Log.d(TAG, "Waiting for handshake request from server...");
        
        // 等待握手请求 - 最多等待60秒
        long startTime = System.currentTimeMillis();
        long timeout = 60000; // 60秒超时
        boolean handshakeReceived = false;
        
        while (System.currentTimeMillis() - startTime < timeout) {
            try {
                // 尝试接收握手请求
                byte[] messageBytes = commuSocket.recv(1000); // 使用短超时循环检查
                
                if (messageBytes != null) {
                    String message = new String(messageBytes);
                    Log.d(TAG, "Received message: " + message);
                    
                    if ("HANDSHAKE_REQUEST".equals(message)) {
                        Log.d(TAG, "Received handshake request from server");
                        handshakeReceived = true;
                        
                        // 回复准备就绪消息
                        Log.d(TAG, "Sending HANDSHAKE_READY response");
                        commuSocket.send("HANDSHAKE_READY".getBytes());
                        
                        Log.d(TAG, "Handshake completed, waiting for recovery signal");
                        break; // 退出等待循环
                    } else {
                        Log.w(TAG, "Unexpected message during handshake: " + message);
                        // 可能是其他消息，尝试下一次接收
                    }
                }
            } catch (Exception e) {
                // 超时或其他错误，继续等待
                Log.d(TAG, "Waiting for handshake request... (" + (System.currentTimeMillis() - startTime) / 1000 + "s)");
            }
            
            // 短暂休眠，避免CPU占用过高
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                Log.e(TAG, "Sleep interrupted", e);
            }
        }
        
        if (!handshakeReceived) {
            Log.e(TAG, "Handshake request not received within timeout period");
            // 可以选择继续等待故障恢复信号，或者放弃并返回失败
            // 这里我们选择继续，因为服务器可能使用旧协议
            Log.d(TAG, "Continuing with regular recovery process despite handshake failure");
        }
        
    } catch (Exception e) {
        Log.e(TAG, "Error during handshake: " + e.getMessage());
        e.printStackTrace();
        // 继续处理，尝试接收故障恢复信号
    }
    // ======== 握手协议代码结束 ========
    
    try {
        // 等待接收服务器发送的故障恢复信号
        Log.d(TAG, "Waiting for failure recovery signal from server");
        
        // ... 原有的故障恢复接收逻辑 ...
    } catch (Exception e) {
        // ... 原有的异常处理 ...
    }
} 