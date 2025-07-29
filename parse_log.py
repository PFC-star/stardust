import json
import matplotlib

matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from datetime import datetime
import re


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
    return times, throughputs


# Function to parse log data for fine-grained plot
def parse_fine_grained_data(filename):
    """Parse log file for clientReceive and tailerResult from tokenStageTimes."""
    mean_times = []
    time_diffs = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                log_entry = json.loads(line.strip())
                token_stage_times = log_entry['queryLog']['tokenStageTimes']
                for token_str in token_stage_times:
                    client_start, tailer_end = parse_token_stage_times(token_str)
                    if client_start and tailer_end:
                        mean_time = client_start + (tailer_end - client_start) / 2
                        time_diff = (tailer_end - client_start).total_seconds()
                        mean_times.append(mean_time)
                        time_diffs.append(time_diff)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
            except KeyError as e:
                print(f"Missing key {e} in line: {line.strip()}")
    return mean_times, time_diffs


# Function to plot throughput vs time
def plot_throughput(times, throughputs, output_file='throughput_plot.png'):
    """Plot throughput against mean timeSpan."""
    plt.figure(figsize=(10, 6))
    plt.plot(times, throughputs, marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Throughput')
    plt.title('Throughput vs Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()


# Function to plot fine-grained time difference
def plot_fine_grained(mean_times, time_diffs, output_file='throughput_fine_grained.png'):
    """Plot time difference against mean of clientReceive and tailerResult."""
    plt.figure(figsize=(10, 6))
    plt.scatter(mean_times, time_diffs, marker='o')
    plt.xlabel('Mean Time (clientReceive and tailerResult)')
    plt.ylabel('Time Difference (seconds)')
    plt.title('Time Difference vs Mean Time for Each Token')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()


# Main function to run both plots
def main():
    """Main function to execute both plotting tasks."""
    log_file = 'device_logs/192.168.226.136.json'

    # Plot throughput vs timeSpan
    times, throughputs = parse_throughput_data(log_file)
    if times and throughputs:
        plot_throughput(times, throughputs, 'throughput_plot.png')
    else:
        print("No valid data for throughput plot.")

    # Plot fine-grained time difference
    mean_times, time_diffs = parse_fine_grained_data(log_file)
    if mean_times and time_diffs:
        plot_fine_grained(mean_times, time_diffs, 'throughput_fine_grained.png')
    else:
        print("No valid data for fine-grained plot.")


if __name__ == '__main__':
    main()