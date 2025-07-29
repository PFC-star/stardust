import json
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from datetime import datetime
import re
import pandas as pd # Import pandas for Excel export

# Function to parse timeSpan and compute mean time
def parse_timespan(timespan):
    """Parse timeSpan string (e.g., 'start~end') and return mean datetime."""
    try:
        start_str, end_str = timespan.split('~')
        start_time = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S.%f')
        end_time = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S.%f')
        return start_time + (end_time - start_time) / 2
    except ValueError as e:
        print(f"Invalid timeSpan format: {timespan}")
        raise e


# Function to parse timestamp
def parse_timestamp(timestamp):
    """Parse a single timestamp string to datetime."""
    try:
        return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError as e:
        print(f"Invalid timestamp format: {timestamp}")
        raise e


# Function to parse tokenStageTimes for clientReceive and tailerResult
def parse_token_stage_times(token_stage_str):
    """Parse tokenStageTimes string to extract clientReceive and tailerResult times."""
    try:
        # Extract clientReceive and tailerResult timestamps using regex
        client_pattern = r'clientReceive:(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3})-'
        tailer_pattern = r'tailerResult:.*?(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3})$'

        client_match = re.search(client_pattern, token_stage_str)
        tailer_match = re.search(tailer_pattern, token_stage_str)

        if not client_match or not tailer_match:
            print(f"Skipping invalid token stage format: {token_stage_str}")
            return None, None

        client_start = parse_timestamp(client_match.group(1))
        tailer_end = parse_timestamp(tailer_match.group(1))
        return client_start, tailer_end
    except ValueError as e:
        print(f"Error parsing token stage times: {token_stage_str}")
        return None, None


# Function to parse log data for throughput plot
def parse_throughput_data(filename):
    """Parse log file for timeSpan and throughput data."""
    times = []
    throughputs = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                log_entry = json.loads(line.strip())
                timespan = log_entry['timeSpan']
                throughput = log_entry['queryLog']['throughput']
                mean_time = parse_timespan(timespan)
                times.append(mean_time)
                throughputs.append(throughput)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
            except KeyError as e:
                print(f"Missing key {e} in line: {line.strip()}")
    processed_times = []
    processed_throughputs = []
    threshold_seconds = 60 * 60 * 2  # Threshold for time gap (e.g., 60 seconds), adjust as needed
    for i in range(len(times)):
        processed_times.append(times[i])
        processed_throughputs.append(throughputs[i])
        if i < len(times) - 1:
            time_gap = (times[i + 1] - times[i]).total_seconds()
            if time_gap > threshold_seconds:
                processed_times.append(None)  # Insert None for mean_times
                processed_throughputs.append(None)  # Insert None for time_diffs
    return processed_times, processed_throughputs

def parse_temperature_data(filename):
    """Parse log file for timeSpan and throughput data."""
    times = []
    temperature = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                log_entry = json.loads(line.strip())
                # Ensure 'energyEvents' exists and is not empty before trying to access index 0
                if 'energyEvents' in log_entry and len(log_entry['energyEvents']) > 0:
                    temperature_ = log_entry['energyEvents'][0]['temperature']
                    timespan = log_entry['timeSpan'] # Use timespan from queryLog for consistent X-axis with other plots
                    mean_time = parse_timespan(timespan)
                    times.append(mean_time)
                    temperature.append(temperature_)
                else:
                    print(f"Skipping line: 'energyEvents' not found or empty in {line.strip()}")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
            except KeyError as e:
                print(f"Missing key {e} in line: {line.strip()}")
    processed_times = []
    processed_temperature = []
    threshold_seconds = 60 * 60 * 2  # Threshold for time gap (e.g., 60 seconds), adjust as needed
    for i in range(len(times)):
        processed_times.append(times[i])
        processed_temperature.append(temperature[i])
        if i < len(times) - 1:
            time_gap = (times[i + 1] - times[i]).total_seconds()
            if time_gap > threshold_seconds:
                processed_times.append(None)  # Insert None for mean_times
                processed_temperature.append(None)  # Insert None for time_diffs
    return processed_times, processed_temperature


# Function to parse log data for fine-grained plot (every N tokens)
def parse_fine_grained_data(filename, n_tokens=5):
    """Parse log file for clientReceive and tailerResult from tokenStageTimes, grouping by n_tokens."""
    mean_times = []
    throughputs = [] # Renamed from time_diffs to better reflect throughput
    with open(filename, 'r') as file:
        for line in file:
            try:
                log_entry = json.loads(line.strip())
                token_stage_times = log_entry['queryLog']['tokenStageTimes']
                # Process tokens in groups of n_tokens
                for i in range(0, len(token_stage_times), n_tokens):
                    token_group = token_stage_times[i:i + n_tokens]
                    if len(token_group) < n_tokens:
                        print(f"Skipping incomplete token group (size {len(token_group)} < {n_tokens})")
                        continue
                    # Get first clientReceive and last tailerResult
                    first_token_str = token_group[0]
                    last_token_str = token_group[-1]

                    first_client_start, _ = parse_token_stage_times(first_token_str)
                    _, last_tailer_end = parse_token_stage_times(last_token_str)

                    if first_client_start and last_tailer_end:
                        mean_time = first_client_start + (last_tailer_end - first_client_start) / 2
                        duration = (last_tailer_end - first_client_start).total_seconds()
                        if duration > 0: # Avoid division by zero
                            throughput = n_tokens / duration
                            mean_times.append(mean_time)
                            throughputs.append(throughput)
                        else:
                            print(f"Skipping zero duration token group: {first_token_str} to {last_token_str}")

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
            except KeyError as e:
                print(f"Missing key {e} in line: {line.strip()}")
    return mean_times, throughputs

# New function to export data to Excel
def export_to_excel(data_dict, sheet_name, excel_writer):
    """Exports a dictionary of data to a specified sheet in an Excel file."""
    df = pd.DataFrame(data_dict)
    df.to_excel(excel_writer, sheet_name=sheet_name, index=False)
    print(f"Data exported to Excel sheet: '{sheet_name}'")


# Function to plot throughput vs time
def plot_throughput(times, throughputs, output_file='throughput_plot.png', excel_writer=None):
    """Plot throughput against mean timeSpan and export data to Excel."""
    plt.figure(figsize=(10, 6))
    plt.plot(times, throughputs, marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Throughput (dialogues/sec)') # More specific unit for dialogue-based throughput
    plt.title('Throughput (Every Dialogue)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()

    if excel_writer:
        data_to_export = {
            'Time': times,
            'Throughput': throughputs
        }
        export_to_excel(data_to_export, 'Throughput_Dialogue', excel_writer)


def plot_temperature(times, temperature, output_file='temperature_plot.png', excel_writer=None):
    """Plot temperature against mean timeSpan and export data to Excel."""
    plt.figure(figsize=(10, 6))
    plt.plot(times, temperature, marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)') # Specific unit for temperature
    plt.title('Device Temperature Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()

    if excel_writer:
        data_to_export = {
            'Time': times,
            'Temperature': temperature
        }
        export_to_excel(data_to_export, 'Temperature', excel_writer)


# Function to plot fine-grained time difference
def plot_fine_grained(mean_times, throughputs, n_tokens=5, output_file='throughput_fine_grained.png', excel_writer=None):
    """Plot fine-grained throughput and export data to Excel."""
    plt.figure(figsize=(10, 6))
    plt.plot(mean_times, throughputs, marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel(f'Throughput (Tokens per {n_tokens} seconds)') # Adjusted label
    plt.title(f'Fine-Grained Throughput (Every {n_tokens} Tokens)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()

    if excel_writer:
        data_to_export = {
            'Time': mean_times,
            'Throughput_Tokens': throughputs
        }
        export_to_excel(data_to_export, f'Throughput_{n_tokens}Tokens', excel_writer)


# Main function to run all plots and export data
def main():
    """Main function to execute all plotting tasks and export data to Excel."""
    log_file = 'device_logs12/192.168.226.136.json'
    n_tokens = 10  # Number of tokens per group, adjustable
    output_excel_file = 'experiment_data2.xlsx' # Define the output Excel file name

    # Create a single ExcelWriter object to write to multiple sheets
    # 'openpyxl' engine is required for .xlsx files
    with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
        # Plot throughput vs timeSpan
        times, throughputs = parse_throughput_data(log_file)
        if times and throughputs:
            plot_throughput(times, throughputs, 'throughput_plot.png', writer)
        else:
            print("No valid data for throughput plot.")

        # Plot temperature vs timeSpan
        times, temperature = parse_temperature_data(log_file)
        if times and temperature:
            plot_temperature(times, temperature, 'temperature_plot.png', writer)
        else:
            print("No valid data for temperature plot.")

        # Plot fine-grained time difference
        mean_times_fg, throughputs_fg = parse_fine_grained_data(log_file, n_tokens)
        if mean_times_fg and throughputs_fg:
            plot_fine_grained(mean_times_fg, throughputs_fg, n_tokens, 'throughput_fine_grained.png', writer)
        else:
            print("No valid data for fine-grained plot.")

    print(f"\nAll plots generated and data exported to '{output_excel_file}'.")


if __name__ == '__main__':
    main()