package com.example.distribute_ui
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData

object DataRepository {
    private val _isDirEmptyLiveData = MutableLiveData<Boolean>()    // Private LiveData variable, used to record whether the directory is empty
    val isDirEmptyLiveData: LiveData<Boolean> = _isDirEmptyLiveData // Externally accessible read-only LiveData

    private val _decodingStringLiveData = MutableLiveData<String>() // Records the decoded string
    val decodingStringLiveData: LiveData<String> = _decodingStringLiveData

    // sample refers to drawing a specific output from the model's output probability distribution
    private val _sampleId = MutableLiveData<Int>()  // Records the ID of the device performing sampling
    val sampleId: LiveData<Int> = _sampleId

    fun updateSampleId(sampleId: Int) {
        _sampleId.postValue(sampleId)   // Update the value of sampleId and notify all observers; postValue updates LiveData asynchronously on a thread other than the main thread
    }

    fun updateDecodingString(updatedString: String) {
//        val responsePosition: Int = updatedString.indexOf("Response:")
//        val decodedStringAfterResponse: String = updatedString.substring(responsePosition + 9)
        _decodingStringLiveData.postValue(updatedString)    // Update the value of decodingString and notify all observers
    }

    fun setIsDirEmpty(isEmpty: Boolean) {       // Set the value of isDirEmptyLiveData
        _isDirEmptyLiveData.postValue(isEmpty)
    }
}