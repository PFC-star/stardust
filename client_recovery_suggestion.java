/**
 * 建议修改Android客户端代码，添加故障恢复确认
 * 这些修改应该添加到相应的故障处理函数中
 */

// 当成功处理故障恢复信号后，发送确认消息
private void sendRecoveryAcknowledgment() {
    try {
        Log.d(TAG, "发送故障恢复确认");
        
        // 确保通信套接字正常
        if (commuSocket != null) {
            // 发送确认消息
            commuSocket.send("FAILURE_RECOVERY_ACK".getBytes());
            Log.d(TAG, "故障恢复确认已发送");
            
            // 等待服务器回复确认收到
            try {
                byte[] response = commuSocket.recv(5000);  // 5秒超时
                if (response != null) {
                    String responseStr = new String(response);
                    Log.d(TAG, "收到服务器确认回复: " + responseStr);
                } else {
                    Log.w(TAG, "未收到服务器确认回复，但继续处理");
                }
            } catch (Exception e) {
                Log.w(TAG, "等待服务器确认回复时出错: " + e.getMessage());
                // 继续处理，不阻止恢复流程
            }
        } else {
            Log.e(TAG, "通信套接字为null，无法发送确认");
        }
    } catch (Exception e) {
        Log.e(TAG, "发送故障恢复确认时出错: " + e.getMessage());
        e.printStackTrace();
    }
}

// 当接收到无替代设备故障通知时，发送确认
private void sendNoReplacementAcknowledgment() {
    try {
        Log.d(TAG, "发送无替代设备故障确认");
        
        if (commuSocket != null) {
            // 发送确认消息
            commuSocket.send("SYSTEM_FAILURE_NO_REPLACEMENT_ACK".getBytes());
            Log.d(TAG, "无替代设备故障确认已发送");
            
            // 等待服务器回复确认收到
            try {
                byte[] response = commuSocket.recv(5000);  // 5秒超时
                if (response != null) {
                    String responseStr = new String(response);
                    Log.d(TAG, "收到服务器确认回复: " + responseStr);
                } else {
                    Log.w(TAG, "未收到服务器确认回复");
                }
            } catch (Exception e) {
                Log.w(TAG, "等待服务器确认回复时出错: " + e.getMessage());
            }
        } else {
            Log.e(TAG, "通信套接字为null，无法发送确认");
        }
    } catch (Exception e) {
        Log.e(TAG, "发送无替代设备故障确认时出错: " + e.getMessage());
        e.printStackTrace();
    }
}

/**
 * 将这些函数调用添加到以下位置：
 * 
 * 1. 在处理完FAILURE_RECOVERY信号并恢复运行状态后:
 *    param.status = "Running";
 *    Log.d(TAG, "故障恢复完成，恢复运行状态");
 *    sendRecoveryAcknowledgment();  // 添加此行
 * 
 * 2. 在处理完SYSTEM_FAILURE_NO_REPLACEMENT信号后:
 *    param.status = "Suspended";
 *    Log.d(TAG, "系统已暂停，等待人工干预");
 *    sendNoReplacementAcknowledgment();  // 添加此行
 */ 