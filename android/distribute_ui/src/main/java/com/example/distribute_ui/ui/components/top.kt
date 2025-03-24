package com.example.distribute_ui.ui.components

import android.widget.Space
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.requiredHeight
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material3.DrawerState
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.distribute_ui.R
import com.example.distribute_ui.data.Dim
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TopBar(scope: CoroutineScope, drawerState: DrawerState, navController: NavController){
    val gradientBrush = Brush.horizontalGradient(
        colors = listOf(Color(0xFFDAB800), Color(0xFFD4C159)) // Gradient from left to right
    )
    val moFont = FontFamily(
        Font(R.font.mono)
    )
    Box(modifier = Modifier.fillMaxWidth().requiredHeight((0.12 * Dim.height).dp).background(gradientBrush)){
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = (0.055 * Dim.height).dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {

            Icon(Icons.Filled.Menu, "", tint = Color.White,
                modifier = Modifier.size(40.dp).padding(start = 10.dp)
                    .clickable {
                        scope.launch {
                            drawerState.open()
                        }
                    }
            )

            Row  (
                verticalAlignment = Alignment.CenterVertically
            ){
                Image(painter = painterResource(R.drawable.star2), "",
                    modifier = Modifier.size(30.dp))
                Text(
                    text = "StarDust",
                    color = Color.White,
                    fontSize = 25.sp,
                    fontFamily = moFont,
                    letterSpacing = 4.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }

            Icon(Icons.Outlined.Edit, "", tint = Color.White,
                modifier = Modifier.size(36.dp).padding(end = 10.dp)
                    .clickable {
                        navController.navigate("Chat") {
                            // Pop up to the "Chat" route, inclusive = true means including "Chat" itself
                            popUpTo("Chat") {
                                inclusive = true
                            }
                            // Ensure only one instance of "Chat" exists at the top of the stack, and don't restore previous state
                            launchSingleTop = true
                            restoreState = false
                        }
                    }
            )
        }
    }
}