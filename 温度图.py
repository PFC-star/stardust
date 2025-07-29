import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates  # Import for date formatting
from datetime import datetime, date
import numpy as np  # Used for NaN values

# Set Matplotlib backend. Keep this if you're on macOS and facing display issues.
# Otherwise, you might not need this line or can use 'Agg'.
matplotlib.use('MacOSX')


def process_data_for_gaps(times, data_series, threshold_seconds):
    """
    Inserts None into time and data series where the time difference
    between consecutive points exceeds the threshold_seconds. This is used
    to break lines in the plot.

    Args:
        times (list): List of datetime objects for the x-axis.
        data_series (list): List of data points corresponding to times.
        threshold_seconds (float): The time gap threshold in seconds.

    Returns:
        tuple: (processed_times, processed_data_series) with Nones inserted.
    """
    processed_times = []
    processed_data_series = []

    for i in range(len(times)):
        processed_times.append(times[i])
        processed_data_series.append(data_series[i])

        if i < len(times) - 1:
            # Check for None values before calculating time_gap
            if times[i] is None or times[i + 1] is None:
                # If a None is encountered, we're already handling a gap, just append None
                processed_times.append(None)
                processed_data_series.append(None)
                continue

            time_gap = (times[i + 1] - times[i]).total_seconds()
            if time_gap > threshold_seconds:
                # Insert None to break the line
                processed_times.append(None)
                processed_data_series.append(None)
    return processed_times, processed_data_series


def plot_temperatures_from_excel(excel_file_path, sheet_name=0, output_image_file='temperatures_plot.png',
                                 time_gap_threshold_minutes=30):
    """
    Reads temperature data from an Excel file, forces all dates to 2025/06/25,
    and plots three temperature series against time.
    Breaks lines where time gaps are too large.
    The X-axis ticks will only display Hour and Minute.

    Args:
        excel_file_path (str): The path to the Excel file.
        sheet_name (int or str, optional): The name or index of the sheet to read. Defaults to the first sheet (0).
        output_image_file (str, optional): The filename for the saved plot image.
                                       Defaults to 'temperatures_plot.png'.
        time_gap_threshold_minutes (float): Time gap threshold in minutes for breaking lines.
                                            Defaults to 30 minutes.
    """
    try:
        # 1. Read Excel file, no initial date parsing here.
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name,
                           engine='openpyxl')

        # Check if the DataFrame has at least 4 columns (Time, Temp1, Temp2, Temp3)
        if df.shape[1] < 4:
            print(f"Error: The Excel file '{excel_file_path}' (sheet '{sheet_name}') "
                  f"must have at least 4 columns (Time, Temperature 1, Temperature 2, Temperature 3). "
                  f"Found {df.shape[1]} columns.")
            return

        # 2. Explicitly assign column names
        df.columns = ['Time', 'Temperature_Device_A', 'Temperature_Device_B', 'Temperature_Device_C']

        # 3. Explicitly convert 'Time' column to datetime objects
        # Specify your time format: '%Y/%m/%d %H:%M:%S'.
        # errors='coerce' will turn unparseable strings into NaT (Not a Time).
        df['Time'] = pd.to_datetime(df['Time'], format='%Y/%m/%d %H:%M:%S', errors='coerce')

        # 4. Drop rows where 'Time' column failed to parse (is NaT)
        df.dropna(subset=['Time'], inplace=True)

        # 5. Force the date part of 'Time' column to 2025/06/25
        fixed_date = date(2025, 6, 25)
        df['Time'] = df['Time'].apply(lambda dt:
                                      datetime(fixed_date.year, fixed_date.month, fixed_date.day,
                                               dt.hour, dt.minute, dt.second, dt.microsecond)
                                      )

        # 6. Convert time gap threshold to seconds
        threshold_seconds = time_gap_threshold_minutes * 60

        # 7. Apply time gap processing to each temperature curve
        times_A, temps_A = process_data_for_gaps(df['Time'].tolist(), df['Temperature_Device_A'].tolist(),
                                                 threshold_seconds)
        times_B, temps_B = process_data_for_gaps(df['Time'].tolist(), df['Temperature_Device_B'].tolist(),
                                                 threshold_seconds)
        times_C, temps_C = process_data_for_gaps(df['Time'].tolist(), df['Temperature_Device_C'].tolist(),
                                                 threshold_seconds)

        # 8. Create the plot figure and adjust size for width
        plt.figure(figsize=(24, 8))  # Width 24 inches, height 8 inches

        # 9. Plot the three temperature curves
        plt.plot(times_A, temps_A, label='Device 1',
                 marker='o', linestyle='-', markersize=6, linewidth=3, color='blue')

        plt.plot(times_B, temps_B, label='Device 2',
                 marker='^', linestyle='--', markersize=6, linewidth=3, color='brown')

        plt.plot(times_C, temps_C, label='Device 3',
                 marker='s', linestyle=':', markersize=6, linewidth=3, color='green')

        # 10. Set plot labels, title, grid, and legend, adjust font sizes
        # plt.xlabel('Time (Hour:Minute)', fontsize=20)  # Updated X-axis label
        plt.ylabel('Temperature (°C)', fontsize=30)
        # If you don't need a main title, you can comment this out
        # plt.title('Device Temperatures Over Time', fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=25)  # Legend font size

        # --- IMPORTANT: Format X-axis ticks to show only Hour:Minute ---
        # Create a DateFormatter object for 'Hour:Minute'
        formatter = mdates.DateFormatter('%H:%M')
        # Apply the formatter to the current axes' x-axis
        plt.gca().xaxis.set_major_formatter(formatter)
        # You might also want to set major locators to ensure good tick density, e.g., every 2 hours
        # plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        # --- End of X-axis formatting ---

        plt.xticks(rotation=0, ha='center', fontsize=25)  # X-axis tick font size and rotation
        plt.yticks(fontsize=25)  # Y-axis tick font size
        plt.tight_layout()  # Adjust layout automatically to prevent labels from overlapping

        # 11. Save and display the plot
        plt.savefig(output_image_file, dpi=300)  # Save as high-resolution image
        print(f"Plot saved to {output_image_file}")

        plt.show()

    except FileNotFoundError:
        print(f"Error: Excel file not found at '{excel_file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")


# --- How to use it ---
if __name__ == '__main__':
    # Call the function to plot
    # Ensure 'experiment_data.xlsx' exists and has a sheet named 'Temperature'
    # This sheet should contain four columns: Time (e.g., '2025/7/1 10:30:09' format),
    # Device A Temp, Device B Temp, Device C Temp
    plot_temperatures_from_excel(
        excel_file_path='experiment_data.xlsx',
        sheet_name='Temperature',
        time_gap_threshold_minutes=100  # Example: break line if gap > 100 minutes
    )