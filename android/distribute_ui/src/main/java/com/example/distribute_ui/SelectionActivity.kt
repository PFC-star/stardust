package com.example.distribute_ui

import android.Manifest
import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.example.distribute_ui.data.PreferenceHelper
import com.example.distribute_ui.data.serverIp
import com.example.distribute_ui.service.BackgroundService
import com.example.distribute_ui.service.MonitorService
import com.example.distribute_ui.ui.InferenceViewModel
import com.example.distribute_ui.ui.components.SplashScreen
import com.example.distribute_ui.ui.theme.Distributed_inference_demoTheme
import com.google.accompanist.systemuicontroller.rememberSystemUiController
import org.greenrobot.eventbus.EventBus


const val TAG = "StarDust"

// Define the permission request code
private const val MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE = 1 // or any other unique integer

// Main entry point of the application, inherits from ComponentActivity and implements LatencyMeasurementCallbacks interface
class SelectionActivity : ComponentActivity(), LatencyMeasurementCallbacks {
    // Intent class: the basic class for inter-component communication. Used to pass messages between components, start activities, start services, and send broadcasts
    private var monitorIntent: Intent? = null       // Intent object responsible for the monitor service
    private var backgroundIntent: Intent? = null    // Intent object responsible for the background service
    // InferenceViewModel instance obtained by delegation, used to manage and save UI data; its lifecycle is bound to the Activity by delegation
    private val viewModel : InferenceViewModel by viewModels()

    private var service: MonitorService? = null // Service instance of MonitorService
    private var serviceBound = false            // Flag indicating whether the service has been successfully bound (whether the service exists)

    private var id = 0 // Flag for whether the device is a header node or a participant node: 1 -> header, 0 -> worker
    private var modelName = ""  // Record the model name

    // An instance object of ServiceConnection
    private val serviceConnection = object : ServiceConnection {
        // When the service is connected successfully, get the service instance through iBinder and set serviceBound to true
        override fun onServiceConnected(className: ComponentName, iBinder: IBinder) {
            Log.d(TAG, "monitor service connection is successful")

//            val binder = service as MonitorActions.MyBinder
//            service = binder.getService()

            val binder = iBinder as MonitorService.LocalBinder
            service = binder.getService()
            serviceBound = true

            // Fetch data from service and update the ViewModel, upload memory and CPU frequency
//            val memory = service?.getAvailableMemory()
//            val freq = service?.getFrequency()
//            viewModel.prepareUploadData(memory ?: 0, freq ?: 0.0)
        }

        // When the service is disconnected, set serviceBound to false
        override fun onServiceDisconnected(arg0: ComponentName) {
            serviceBound = false
        }
    }

    // An instance object of BroadcastReceiver, used to receive broadcasts
    private val receiver: BroadcastReceiver = object : BroadcastReceiver() {
        // After receiving the broadcast, start monitorIntent and pass the current device id
        override fun onReceive(context: Context?, intent: Intent?) {
            Log.d(TAG, "selectionActivity receives the broadcast")
            monitorIntent!!.putExtra("role", id)   // Add extra data named "role" with value id to the Intent
            startService(monitorIntent)                  // Start the background service MonitorService via monitorIntent
        }
    }

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Request permission to read and write external storage
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Log.d(BackgroundService.TAG, "write external storage denied")
            // Permission is not granted
            // Should we show an explanation?
            if (ActivityCompat.shouldShowRequestPermissionRationale(
                    this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
                )
            ) {
                // Show an explanation to the user *asynchronously* -- don't block
                // this thread waiting for the user's response! After the user
                // sees the explanation, try again to request the permission.
            } else {
                // No explanation needed; request the permission
                ActivityCompat.requestPermissions(
                    this, arrayOf<String>(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                    MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE
                )

                // MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE is an
                // app-defined int constant. The callback method gets the
                // result of the request.
            }
        } else {
            // Permission has already been granted
            Log.d(BackgroundService.TAG, "write external storage permit is ok")
        }

        backgroundIntent = Intent(this, BackgroundService::class.java)  // Intent instance pointing to BackgroundService
        monitorIntent = Intent(this ,MonitorService::class.java)        // Intent instance pointing to MonitorService
//        startService(monitorIntent)

        // The filter is used to select which broadcasts to receive. It specifies that only broadcasts with the action 'START_MONITOR' will be received, i.e., any component that sends a broadcast with this action will trigger this receiver
        val filter = IntentFilter("START_MONITOR")
        // First get the class for sending and receiving broadcasts at the application level, then bind the broadcast receiver and filter. Whenever a broadcast that meets the filter conditions is sent, the receiver will receive it and execute the corresponding logic
        LocalBroadcastManager.getInstance(this).registerReceiver(receiver, filter)


        WindowCompat.setDecorFitsSystemWindows(window, false)
        serverIp = PreferenceHelper.loadServerIp(this)
        setContent {
            var showSplash by remember { mutableStateOf(true) }
            val systemUiController = rememberSystemUiController()
            systemUiController.setSystemBarsColor(
                color = Color.Transparent,
                darkIcons = false
            )

            if(showSplash){
                SplashScreen(onAnimationEnd = { showSplash = false})
            }
            else {
                HomeScreen(
                    onMonitorStarted = {    // Start MonitorService, pass id
                        monitorIntent!!.putExtra("role", id)
                        startService(monitorIntent)
                    },
                    onBackendStarted = {    // If there is no running BackgroundService, start it and pass id and model name
                        if (!BackgroundService.isServiceRunning) {
                            backgroundIntent!!.putExtra("role", id)
                            backgroundIntent!!.putExtra("model", modelName)
                            backgroundIntent!!.putExtra("ip", serverIp)
                            startService(backgroundIntent)
                        }
                    },
                    onModelSelected = { // Set model name
                        setModel(it)
                    },
                    onRolePassed = {    // Set role
                        setRole(it)
                    },
                    viewModel = viewModel
                )
            }
        }
    }

    private fun setRole(id: Int) {      // Set the value of property id
        this.id = id
        Log.d(TAG, "id is $id")
    }

    private fun setModel(modelName: String) {   // Set the value of property modelName
        this.modelName = modelName
        Log.d(TAG, "model name is $modelName")
    }

    // When the Activity is destroyed, unregister the broadcast receiver, unbind the Monitor service instance, and stop the background service
    override fun onDestroy() {
        super.onDestroy()
//        unregisterReceiver(receiver);
        LocalBroadcastManager.getInstance(this).unregisterReceiver(receiver)
        if (serviceBound) {
            unbindService(serviceConnection)
            serviceBound = false
        }
//        stopService(monitorIntent)
        stopService(backgroundIntent)
    }

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onLatencyMeasured(latency: Double) {   // Update the ViewModel's latency data with latency
        viewModel.updateLatency(latency)
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        val isScreenOn = pm.isInteractive // Compatible with API 20+
        EventBus.getDefault().post(com.example.distribute_ui.service.BackgroundService.ScreenOffEvent(!isScreenOn))
        android.util.Log.d("ScreenDetect", "onWindowFocusChanged: hasFocus=$hasFocus, isScreenOn=$isScreenOn")
    }

    companion object {
        init {
            System.loadLibrary("distributed_inference_demo")    // Load native library
        }
    }
    // Declare external functions (JNI methods)
    external fun createSession(inference_model_path:String): Long   // Create session, return
    external fun modelFlopsPerSecond(modelFlops: Int, session: Long, data: ByteArray?): Double  // Calculate model FLOPS per second
}

interface LatencyMeasurementCallbacks {
    fun onLatencyMeasured(latency: Double)
}

