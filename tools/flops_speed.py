import numpy as np

{0: 3140000000, 1: 9160000000}
flops_spped_total= np.array([17128238928,
                            52008456659,
                            22955103793,
                            34432655689,
                            18917814128,
                            22710674009,
                            21637610608,
                            26405737197,
                            38184389470,
                            11982841446,
                            27515222061,
                            15837547706,
                            20149608771])






def inference_time(flop,device):

    inference_time_ = flop / flops_spped_total[device]
    # print("inference_time_:",inference_time_)
    return inference_time_

def inference_time_ni(M):
    return 0.15 * M ** 0.73

# flop = 63510000000
flop = 3140000000
device = 5
M = 0.5
print("flop:{}   {}s".format(flop,inference_time(flop,device)))
print("M:{}   {}s".format(M,inference_time_ni(M)))

