import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Define ping latency and bandwidth arrays (unit: seconds, MB)
ping_latency_total = [
    [0, 0.034866, 0.017796, 0.076512, 0.030985],
    [0.03257, 0, 0.027257, 0.079517, 0.018823],
    [0.108798, 0.065485, 0, 0.064679, 0.043849],
    [0.206233, 0.028255, 0.034938, 0, 0.02909],
    [0.021318, 0.034842, 0.019245, 0.042098, 0]
]

# Bandwidth array (unit: MB/s)
bandwidths_total = [
    [float("inf"), 1.666069031, 1.81388855, 2.56729126, 4.906654358],
    [0.312805176, float("inf"), 1.200675964, 0.575065613, 0.590324402],
    [0.994682312, 2.725601196, float("inf"), 0.332832336, 0.280007935],
    [0.734411507, 0.634801417, 0.433241694, float("inf"), 0.703819996],
    [1.302121697, 1.801237821, 0.502415113, 0.52142342, float("inf")]
]


# Define communication data size
data_size_kb = 20  # 20 KB = 160 kilobits

# Define loading time and inference time functions
def load_time(M, device):
    param_ = load_time_param[device]
    a = param_[0]
    b = param_[1]
    return max(a* M ** b,0.5)

def inference_time(m,device):
    flop = m/M_total* FLOPs_allocation[faulty_device]
    inference_time_ = flop / flops_spped_total[device]
    # print("inference_time_:",inference_time_)
    return inference_time_

# Calculate communication time between devices
def communication_time(i, j, data_size_kb=20):
    data_size_kilobits = data_size_kb/ 1024 # Convert to MB
    commu_time= ping_latency_total[i][j] + data_size_kilobits / bandwidths_total[i][j]
    return commu_time

# Simulated annealing function

def memory_constraint(Mi):
    ratio =np.array( [m / sum(Mi) for m in Mi])
    memory_occupy = np.array(memory) * ratio
    for index,device_index in enumerate(selected_device_index):
        if memory_occupy[index] > available_memory_total[device_index]:
            return False
    return True

def simulated_annealing(M_total, n, iterations=10000, initial_temp=100, cooling_rate=0.95):
    # Initialize model allocation randomly
    # n = len(device_lst)
    while True:
        Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)

        current_temp = initial_temp
        best_Mi = Mi

        # print("M：",best_Mi)
        best_time, stages = calc_total_time(best_Mi)
        if memory_constraint(best_Mi):
            break
    # Annealing process
    for iteration in range(iterations):
        # Generate new allocation, sorted for fairness
        new_Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)
        if memory_constraint(new_Mi):
            # Calculate total time
            new_time, new_stages = calc_total_time(new_Mi)

            # Accept the new allocation if it's better or by chance
            if new_time < best_time or random.uniform(0, 1) < np.exp((best_time - new_time) / current_temp):
                best_Mi = new_Mi
                best_time = new_time
                stages = new_stages

            # Cool down
            current_temp *= cooling_rate

    return best_Mi, best_time, stages
def simulated_annealing_random(M_total, n):
    # Initialize model allocation randomly
    # n = len(device_lst)
    for i in range(100):
        Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)


        best_Mi = Mi

        # print("M：",best_Mi)
        best_time, stages = calc_total_time(best_Mi)
        if memory_constraint(best_Mi):
            break




    return best_Mi, best_time, stages


# Calculating the total time for all devices
def calc_total_time(Mi):
    total_times = []
    stages = []

    # First device timing (starts at t=0)
    load_start = 0
    load_end = load_time(Mi[0],selected_device_index[0])
    infer_start = load_end
    infer_end = infer_start + inference_time(Mi[0],selected_device_index[0])
    comm_start = infer_end
    comm_end = comm_start + communication_time(selected_device_index[0], selected_device_index[1])  # First device communicates with the next

    total_times.append(comm_end)

    # Record stage timing
    stages.append({
        'load_start': load_start, 'load_end': load_end,
        'infer_start': infer_start, 'infer_end': infer_end,
        'comm_start': comm_start, 'comm_end': comm_end
    })

    # Compute timing for all other devices
    for i in range(1, n):
        load_start = 0
        load_end = load_time(Mi[i],selected_device_index[i])
        infer_start = max(stages[i - 1]['comm_end'], load_end) # Inference can start after previous round communication ends or current load finishes

        infer_end = infer_start + inference_time(Mi[i],selected_device_index[i])
        comm_start = infer_end
        if i==n-1:
            comm_end = comm_start + communication_time(selected_device_index[i], initial_device_index)  # Last device communicates with the first
        else:
            comm_end = comm_start + communication_time(selected_device_index[i], (selected_device_index[i] + 1) % n)  # Circular communication between devices

        total_times.append(comm_end)

        # Record stage timing
        stages.append({
            'load_start': load_start, 'load_end': load_end,
            'infer_start': infer_start, 'infer_end': infer_end,
            'comm_start': comm_start, 'comm_end': comm_end
        })

    # Return the maximum total time, representing the total recovery time
    return max(total_times), stages


def print_stage_info(stages):
    # Create data table
    table = []
    headers = ['Device', 'Load Start (s)', 'Load End (s)', 'Inference Start (s)', 'Inference End (s)', 'Comm Start (s)', 'Comm End (s)']

    # Fill data
    for i, stage in enumerate(stages):
        row = [
            i,
            stage['load_start'],
            stage['load_end'],
            stage['infer_start'],
            stage['infer_end'],
            stage['comm_start'],
            stage['comm_end']
        ]
        table.append(row)

    # Use tabulate to format output table
    print(tabulate(table, headers=headers, floatfmt=".2f", tablefmt="grid"))


def plot_pipeline(results_plot):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,4*n), sharex=True)  # Horizontal arrangement of two subplots
    for index, ax in enumerate(axes):
        stages, (min_index,last_load_time), recovery_time_single_device_lst = results_plot[index]


        for i, stage in enumerate(stages):
            # Load stage
            ax.broken_barh([(stage['load_start'], stage['load_end'] - stage['load_start'])], (i - 0.4, 0.8),
                           facecolors='tab:blue', label="load" if i == 0 else "")
            # Inference stage
            ax.broken_barh([(stage['infer_start'], stage['infer_end'] - stage['infer_start'])], (i - 0.4, 0.8),
                           facecolors='tab:green', label="infer" if i == 0 else "")
            # Communication stage
            ax.broken_barh([(stage['comm_start'], stage['comm_end'] - stage['comm_start'])], (i - 0.4, 0.8),
                           facecolors='tab:red', label="commu" if i == 0 else "")
            if selected_device_index.index(min_index)== i:
                # Load stage
                ax.broken_barh([(stage['comm_end'],last_load_time)], (i - 0.4, 0.8),
                               facecolors='tab:cyan', label="load remain" )
        # Single device recovery
        for index,single_device in enumerate(recovery_time_single_device_lst):
              ax.broken_barh([(0, single_device)], (i +(index+1) - 0.4, 0.8),
                       facecolors='tab:grey', label="single device" if i == 0 else "")
        ax.set_xlabel('time (s)')
        ax.set_ylabel('device')
        ax.set_yticks(range(2*n))
        ax.set_yticklabels([f'device {i}' for i in selected_device_index+selected_device_index])
        ax.grid(True)
        ax.legend()
    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

# Single phone recovery

def recovery_time_single_device(selected_device_index):
    for i in  selected_device_index:
        load_time_single_device = load_time(M_total,i)
        inference_time_single_device = inference_time(M_total,i)
        commu_time_single_device = communication_time(i,initial_device_index)
        recovery_time_single_device = load_time_single_device + inference_time_single_device + commu_time_single_device
        # print(f"Single device:{i} Recovery time: {recovery_time_single_device:.2f} seconds")
        print(recovery_time_single_device)

        recovery_time_single_device_lst.append(recovery_time_single_device)
def last_load(best_Mi,stages):
    recovery_time_temp_lst = []
    recovery_time_lst = []
    last_load_time_lst = []
    for index,Mi in enumerate(best_Mi):
         last_M = M_total - Mi
         last_load_time = load_time(last_M,selected_device_index[index])
         recovery_time = last_load_time + stages[index]['comm_end']
         recovery_time_lst.append(stages[index]['comm_end'])
         recovery_time_temp_lst.append(recovery_time)
         last_load_time_lst.append(last_load_time)
    min_recovery_time_temp = min(recovery_time_temp_lst)
    min_index = recovery_time_temp_lst.index(min_recovery_time_temp)
    recovery_time_lst.append(min_recovery_time_temp)
    min_recovery_time = max(recovery_time_lst)
    return min_recovery_time,selected_device_index[min_index], last_load_time_lst[min_index]

initial_module_arrangement=[ [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 0 ],
                             [1 ,0 ,0 ,0, 0, 0 ,1 ,1, 1, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],
                             [0 ,1 ,1 ,1 ,1 ,1, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 0 ],
                             [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,1 ,1, 1 ,1 ,1 ,1 ,1, 1 ,1 ,1 ,1 ,1 ,1, 1 ,1, 0 ],
                             [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0 ,0 ,0, 0 ,0, 1 ]]

param_dict_1 = {"selected_device_index":[0,3],
              "selected_single_device_index":0,
              "initial_device_index":2,
              "faulty_device":1}
param_dict_2 = {"selected_device_index":[3,4],
              "selected_single_device_index":0,
              "initial_device_index":2,
              "faulty_device":1}
param_dict_3 = {"selected_device_index":[0,4],
              "selected_single_device_index":0,
              "initial_device_index":2,
              "faulty_device":1}

param_dict_4 = {"selected_device_index":[0,3,4],
              "selected_single_device_index":0,
              "initial_device_index":1,
              "faulty_device":1}
param_dict_5 = {"selected_device_index":[0,3],
              "selected_single_device_index":0,
              "initial_device_index":1,
              "faulty_device":2}
param_dict_6 = {"selected_device_index":[3,4],
              "selected_single_device_index":0,
              "initial_device_index":1,
              "faulty_device":2}
param_dict_7 = {"selected_device_index":[0,4],
              "selected_single_device_index":0,
              "initial_device_index":1,
              "faulty_device":2}


param_dict_8 = {"selected_device_index":[0,3,4],
              "selected_single_device_index":0,
              "initial_device_index":1,
              "faulty_device":2}
param_list = [param_dict_1,param_dict_2,param_dict_3,param_dict_4,param_dict_5,param_dict_6,param_dict_7,param_dict_8]

param_allocation = [ sum(i)/len(i)* param for i in initial_module_arrangement]
FLOPs_allocation = [ sum(i)/len(i)* flop for i in initial_module_arrangement]


results_out = []

for param_dict in param_list:
    results = []
    results_plot = []
    recovery_time_single_device_lst = []
    selected_device_index = param_dict["selected_device_index"]
    selected_single_device_index = param_dict["selected_single_device_index"]
    initial_device_index = param_dict["initial_device_index"]
    faulty_device = param_dict["faulty_device"]
    print("selected_device_index:",selected_device_index)
    print("selected_single_device_index:",selected_single_device_index)
    print("initial_device_index:",initial_device_index)
    print("faulty_device:",faulty_device)

    # M_total = param_allocation[faulty_device]
    # M_total = sum(param_allocation)
    M_total = param_allocation[4]

    print("\ndevice_allocation:", param_allocation)
    print("faulty_device:", faulty_device)
    print("M_total:", M_total)

    # ping_latency =np.array(   [ping_latency_total[i].tolist() for i in selected_device_index])

    # # bandwidths =np.array( [bandwidths_total[i].tolist()  for i in selected_device_index])
    # tatal_memory = np.array( [tatal_memory_total[i]   for i in selected_device_index])
    # available_memory =  [available_memory_total[i]   for i in selected_device_index]
    # flops_spped =  [flops_spped_total[i]  for i in selected_device_index]
    device_lst = ["device" + str(i) for i in selected_device_index]
    print("device_lst:", device_lst)

    # Define total model size and device count
    # M_total = 1 # Full model size on device 3
    n = len(selected_device_index)  # n devices

    recovery_time_single_device(selected_device_index)

    # Run optimization and record time

    # Multi-phone optimal recovery (Ours)
    best_Mi, best_time, stages = simulated_annealing(M_total, n)

    min_recovery_time, min_index, last_load_time = last_load(best_Mi, stages)
    last_load_time_tuple = (min_index, last_load_time)
    print("last_load_time:",last_load_time)
    # print(f"Best submodel allocation: {best_Mi}")
    # print(f"Minimum imperceptible recovery time: {best_time:.2f} seconds")
    # print(f"Complete recovery time: {min_recovery_time:.2f} seconds")
    results_out.append([best_Mi, best_time, min_recovery_time, min_index])
    results_plot.append([stages, last_load_time_tuple, recovery_time_single_device_lst])
    # # Print time segment information for each device
    # print_stage_info(stages)

    # Multi-phone random recovery
    # Even split

    for i in range(30):
        best_Mi, best_time, stages = simulated_annealing_random(M_total, n)
        min_recovery_time, min_index, last_load_time = last_load(best_Mi, stages)
        last_load_time_tuple = (min_index, last_load_time)
        # Record results

        results_plot.append([stages, last_load_time_tuple, recovery_time_single_device_lst])
        results.append([best_Mi, best_time, min_recovery_time])

    # Draw pipeline diagram
    plot_pipeline(results_plot)

    # Convert to DataFrame
    df_results = pd.DataFrame(results, columns=["Submodel Allocation", "Imperceptible Recovery Time (seconds)", "Complete Recovery Time (seconds)"])
    # Calculate mean and add to table
    mean_times = df_results[["Imperceptible Recovery Time (seconds)", "Complete Recovery Time (seconds)"]].mean()
    mean_times["Submodel Allocation"] = "Mean"
    df_results = pd.concat([df_results, pd.DataFrame([mean_times], columns=df_results.columns)], ignore_index=True)
    # tools.display_dataframe_to_user(name="Simulated Annealing Results", dataframe=df_results)
    # print(df_results)
    # print(f"Best submodel allocation: {best_Mi}")
    # print(f"Imperceptible recovery time: {df_results['Imperceptible Recovery Time (seconds)'].iloc[-1]:.2f} seconds")
    # print(f"Complete recovery time: {df_results['Complete Recovery Time (seconds)'].iloc[-1]:.2f} seconds")
    recovery_time_single_device([0])
    break

    results_out.append(
        [[], df_results['Imperceptible Recovery Time (seconds)'].iloc[-1], df_results['Complete Recovery Time (seconds)'].iloc[-1], min_index])

    df_results_out = pd.DataFrame(results_out,
                                  columns=["Submodel Allocation", "Imperceptible Recovery Time (seconds)", "Complete Recovery Time (seconds)", "Secondary Load Device"])
print(df_results_out)


# Output to an Excel file
os.makedirs( "results",exist_ok=True)
excel_file_path = 'results/df_results_out_transposed.csv'
df_results_out.T.to_csv(excel_file_path, header=False)

# Simulate each phone's state, including performance, points, etc.
class Phone:
    def __init__(self, id, performance, available_time):
        self.id = id
        self.performance = performance  # Performance (calculation ability, the larger the faster)
        self.available_time = available_time  # Phone idle time, unit seconds
        self.is_free = True
        self.used_time = 0  # Used to record phone usage time

    def use(self, task):
        if self.is_free:
            self.is_free = False
            processing_time = task.complexity / self.performance  # Calculate processing time based on task complexity and phone performance
            self.used_time = processing_time
            return processing_time
        return None

    def free_up(self):
        self.is_free = True
        self.used_time = 0

# User class, representing each user's phone, points, etc.
class User:
    def __init__(self, id, phone, initial_points):
        self.id = id
        self.phone = phone
        self.points = initial_points  # User initial points

    def spend_points(self, points):
        """Consume points"""
        self.points -= points

    def earn_points(self, points):
        """Earn points"""
        self.points += points

# Task class, representing inference task, containing task complexity and required phone processing time
class Task:
    def __init__(self, token_count, model_complexity):
        self.token_count = token_count  # token count
        self.model_complexity = model_complexity  # Model complexity
        self.complexity = self.token_count * self.model_complexity  # Total task complexity = token count * model complexity

# Simulator class, managing task scheduling, point calculation, etc.
class Simulator:
    def __init__(self, users):
        self.users = users  # User list

    def execute_task(self, main_user, task):
        """
        Main user initiates inference task, collaborative users participate in inference.
        Main user consumes points, collaborative users earn points.
        """
        # Calculate total task time and task allocation for each collaborative user
        total_processing_time = 0
        total_points_consumed = 0
        total_points_earned = 0

        # Task split among multiple collaborative users
        for user in self.users:
            if user != main_user:
                phone = user.phone
                processing_time = phone.use(task)
                if processing_time is not None:
                    total_processing_time += processing_time
                    # Collaborative users earn points based on their processing time and task complexity
                    points_earned = task.complexity * (processing_time / total_processing_time)
                    user.earn_points(points_earned)
                    total_points_earned += points_earned

        # Main user consumes points
        points_spent = task.complexity * (total_processing_time / task.complexity)*3
        main_user.spend_points(points_spent)
        total_points_consumed += points_spent

        # Output task execution information
        print(f"Task executed by main user {main_user.id} with {len(self.users) - 1} collaborative users.")
        print(f"Total points consumed by main user: {total_points_consumed:.2f}")
        print(f"Total points earned by collaborative users: {total_points_earned:.2f}")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print("\nUser points after task execution:")
        for user in self.users:
            print(f"User {user.id}: {user.points:.2f} points")

# Main program
if __name__ == "__main__":
    # Create users and their phones
    users = [User(i, Phone(i, random.uniform(1, 5), random.randint(5, 10)), 100) for i in range(5)]
    main_user = users[0]  # Assume the first user is the main user

    # Create task
    task = Task(token_count=1000, model_complexity=0.5)

    # Create and run simulator
    simulator = Simulator(users)
    simulator.execute_task(main_user, task)