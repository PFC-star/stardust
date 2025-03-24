package com.example.distribute_ui.data
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

object Dim{
    var width: Int = 0
    var height: Int = 0
}

var serverIp = "192.168.50.214"

val exampleModelName = listOf(
    "bloom560m",
    "bloom560m-int8",
    "bloom1b1",
    "bloom1b1-int8",
    "bloom1b7",
    "bloom1b7-int8",
    "bloom3b",
    "bloom3b-int8",
    "bloom7b",
    "bloom7b-int8"
)

val modelMap: HashMap<String, String> = hashMapOf(
    "Bloom" to "bloom560m"
)
