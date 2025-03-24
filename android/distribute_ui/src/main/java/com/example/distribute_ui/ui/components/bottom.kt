package com.example.distribute_ui.ui.components

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.outlined.Extension
import androidx.compose.material.icons.outlined.Forum
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.tooling.preview.Preview
import androidx.navigation.NavController
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController

sealed class Bottom(
    val route:String,
    val icon:ImageVector,
    val title:String){

    object Chat:Bottom("Start", Icons.Outlined.Forum, "Chat")
    object Finetune:Bottom("Finetune", Icons.Outlined.Extension, "Finetune")
    object Settings:Bottom("Settings", Icons.Filled.Settings, "Settings")
}

@Composable
fun BottomBar(navController: NavController){
    val bottomlist = listOf(Bottom.Chat, Bottom.Finetune, Bottom.Settings)

    NavigationBar (containerColor = Color.White){
        val navBackStackEntry by navController.currentBackStackEntryAsState()
        val currentRoute = navBackStackEntry?.destination?.route
        bottomlist.forEach {
            NavigationBarItem(
                selected = if (it != Bottom.Chat) (currentRoute == it.route) else {
                    currentRoute == "Start" || currentRoute == "Chat/{index}" || currentRoute == "Select"
                },
                label = { Text(it.title) },
                alwaysShowLabel = true,
                icon = { Icon(it.icon, "") },
                onClick = {
                    navController.navigate(it.route) {   // Navigate to the specified target route and configure navigation behavior
                        // .let checks if the variable is non-null, if so executes the code inside {} and passes the variable as parameter
                        // Get the starting screen of navController
                        navController.graph.startDestinationRoute?.let { startRoute ->   // This is navController.graph.startDestinationRoute
                            popUpTo(startRoute) {    // Before navigating, pop up screens in the navigation stack until the specified screen
                                saveState = true    // When popping the target page, save its state for restoration
                            }
                        }
                        launchSingleTop = true  // Ensure only one instance of the screen at the top of the stack, avoiding duplicate creation
                        restoreState = true     // When navigating to a screen, restore the state saved by popUpTo
                    }
                },
                colors = NavigationBarItemDefaults.colors(indicatorColor = Color(0x0FD4C159))
            )
        }
    }
}