package com.example.distribute_ui.ui

import android.content.Intent
import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.requiredHeight
import androidx.compose.foundation.layout.requiredWidth
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.dimensionResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.distribute_ui.R
import com.example.distribute_ui.TAG
import com.example.distribute_ui.service.InferenceService
import com.example.distribute_ui.ui.components.ButtonBar
import com.example.distribute_ui.ui.theme.Distributed_inference_demoTheme

@Composable
fun NodeSelectionScreen(
    viewModel: InferenceViewModel,
    onHeaderClicked: () -> Unit,
    onWorkerClicked: () -> Unit,
    onCancelClicked: () -> Unit,
    onNextClicked: (id: Int) -> Unit,
    onBackendStarted: () -> Unit,
    onMonitorStarted: () -> Unit,
    modifier: Modifier = Modifier
) {
    val mediumPadding = dimensionResource(R.dimen.padding_medium)
    val selectedValue = remember { mutableStateOf(false) }
    // 0 for worker, 1 for header
    val selectionNode = remember { mutableStateOf(0) }
    val nextClickedState = remember { mutableStateOf(false) }
    val buttonEnable = remember {
        mutableStateOf(false)
    }

    Column (modifier = Modifier.fillMaxSize()){
        Box (modifier = Modifier
            .fillMaxSize()
            .background(Color.White),
            contentAlignment = Alignment.Center){

            Card(
                modifier = Modifier.requiredWidth(360.dp).requiredHeight(180.dp).padding(16.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 5.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFFe5e496))
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier.fillMaxSize()
                ) {
                    Spacer(modifier = Modifier.height(20.dp))
                    Text(
                        text = "Become header and start chat",
                        style = MaterialTheme.typography.bodyLarge
                    )
                    Spacer(modifier = Modifier.height(20.dp))
                    Card(
                        shape = RoundedCornerShape(50.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFFD4C159)),
                        modifier = Modifier
                            .requiredWidth(120.dp).requiredHeight(60.dp)
                            .padding(top = 10.dp)
                    ){
                        Column (modifier = Modifier.fillMaxSize().clickable {
                            selectedValue.value = true
                            selectionNode.value = 1
                            onHeaderClicked()

                            nextClickedState.value = true
                            onNextClicked(selectionNode.value)
                        },
                            verticalArrangement = Arrangement.Center,
                            horizontalAlignment = Alignment.CenterHorizontally){
                            Text("Next",
                                fontSize = 14.sp,
                                color = Color.White,
                            )
                        }
                    }
                }
            }
        }
    }
}