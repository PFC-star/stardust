Mi = [0.16476441, 0.1897459 , 0.19924598 ,0.20809644, 0.23814727]

def load_time(M):
    return 5.57 * M ** 0.44


def inference_time(M):
    return 0.15 * M ** 0.73

def print_device_info(Mi):
    # 打印表头
    print(f"{'设备':<5}{'模型大小':<15}{'加载时间 (s)':<15}{'推理时间 (s)':<15}")

    # 遍历每个设备上的模型大小，计算并打印加载时间和推理时间
    for i in range(len(Mi)):
        model_size = Mi[i]
        load_t = load_time(model_size)
        inference_t = inference_time(model_size)

        # 打印设备编号、模型大小、加载时间、推理时间
        print(f"{i:<5}{model_size:<15.4f}{load_t:<15.4f}{inference_t:<15.4f}")
print_device_info(Mi)
