## This File remote kill the communication process, if it fails

from control import remote_shell, kill_process

if __name__ == "__main__":
    root_ip = "192.168.1.30"

    # Connection layers:
    # communication_layers = [
    #     ["192.168.1.2"],
    #     ["192.168.1.31", "192.168.1.32", "192.168.1.33"],
    #     ["192.168.1.35", "192.168.1.36"]
    # ]

    communication_layers = [
        ["192.168.1.40"],
        ["192.168.1.41"],
        ["192.168.1.42"]
    ]

    ips = communication_layers[0] + communication_layers[1] + communication_layers[2]
    results = {i: [] for i in ips}
    threads = []

    # shells_l0 = remote_shell(communication_layers[0], username='yurun', passwd='uitim1', port=22)
    shells_l0 = remote_shell(communication_layers[0])
    shells_l1 = remote_shell(communication_layers[1])
    shells_l2 = remote_shell(communication_layers[2])

    for sh in shells_l0 + shells_l1 + shells_l2:
        kill_process(sh)
