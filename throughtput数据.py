import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import random # 导入 random 模块

# 你可能需要为 matplotlib 设置一个后端，例如在 MacOS 上
# 如果你使用的是标准的 Windows/Linux 环境，可能不需要这行代码，或者可以使用 'Agg'
# matplotlib.use('MacOSX')

def process_and_plot_temperatures_from_excel(excel_file_path, sheet_name=0, output_image_file='temperatures_plot.png'):
    """
    从 Excel 文件读取温度数据，添加一列随机温度，然后（理论上）绘制两列温度与时间的关系图。
    此版本侧重于生成新数据并写回 Excel。

    参数:
        excel_file_path (str): Excel 文件的路径。
        sheet_name (int 或 str, 可选): 要读取的表格名称或索引。默认为第一个表格 (0)。
        output_image_file (str, 可选): 保存图表图片的文件名。
                                       默认为 'temperatures_plot.png'。
    """
    try:
        # 读取 Excel 文件。假设第一行是表头。
        # 确保时间列 (索引 0) 被正确解析为日期时间对象
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name,
                           parse_dates=[0],
                           engine='openpyxl')

        # 检查 DataFrame 是否至少有 3 列 (时间, 温度1, 温度2)


        # 为列赋值清晰的名称。如果你的 Excel 表头不同，可能需要调整。
        # 假设第一列是时间，第二列是设备 A 的温度，第三列是设备 B 的温度。
        # 确保 'Temperature_Device_B' 对应到你 Excel 中的第二列温度数据。
        # 如果你的 Excel 没有表头，或者表头不是英文，你可能需要根据实际情况调整这里的索引或名称
        # 示例： df.columns = ['Time', '温度_设备A', '温度_设备B']
        # 为了通用性，我们先用 df.iloc 确保获取到的是第二和第三列
        # 重命名列以方便操作
        df.columns = ['Time', 'Throughput', 'Throughput_10%']


        # 根据 'Temperature_Device_B' 列生成新的温度数据
        # 遍历 'Temperature_Device_B' 列的每个值，加上一个 0.5 到 5 之间的随机浮点数
        new_temperature_data = []
        for temp_b in df['Throughput_10%']:
            random_offset = random.uniform(-0.3, 0.3)-0.75 # 生成 0.5 到 5.0 之间的随机浮点数
            new_temperature_data.append(temp_b + random_offset)

        # 将新生成的数据作为新列添加到 DataFrame
        df['Throughput_20%'] = new_temperature_data
        print("已生成新列 'Temperature_Device_C'，数据如下：")
        print(df.head()) # 打印前几行看看效果

        # 将更新后的 DataFrame 写回 Excel 文件
        # 使用同一个 ExcelWriter 对象，可以写入到不同的 sheet 或覆盖原有 sheet
        with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            # 将更新后的 DataFrame 写入原来的 sheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"\n数据已更新并写回 Excel 文件 '{excel_file_path}' 的 '{sheet_name}' 表格中。")

        # 注意：此函数版本暂时不画图，只处理数据生成和写入。
        # 如果需要画图，请在数据生成和写入完成后，调用绘图逻辑。
        # 以下是之前绘图逻辑的注释，你可以根据需要取消注释并修改。

        # # 绘图代码（如果需要，取消注释）
        # plt.figure(figsize=(12, 7))
        #
        # plt.plot(df['Time'], df['Temperature_Device_A'], label='Device A Temperature',
        #          marker='o', linestyle='-', markersize=4, linewidth=1.5)
        #
        # plt.plot(df['Time'], df['Temperature_Device_B'], label='Device B Temperature',
        #          marker='x', linestyle='--', markersize=4, linewidth=1.5)
        #
        # # 绘制新添加的温度列
        # plt.plot(df['Time'], df['Temperature_Device_C'], label='Device C Temperature (New)',
        #          marker='s', linestyle=':', markersize=4, linewidth=1.5)
        #
        # plt.xlabel('Time', fontsize=12)
        # plt.ylabel('Temperature (°C)', fontsize=12)
        # plt.title('Device Temperatures Over Time', fontsize=14)
        # plt.grid(True, linestyle='--', alpha=0.7)
        # plt.legend(fontsize=10)
        # plt.xticks(rotation=45, ha='right', fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.tight_layout()
        #
        # plt.savefig(output_image_file, dpi=300)
        # print(f"图表已保存到 {output_image_file}")
        # plt.show()

    except FileNotFoundError:
        print(f"错误: Excel 文件未找到 '{excel_file_path}'")
    except Exception as e:
        print(f"发生错误: {e}")

# --- 如何使用 ---
if __name__ == '__main__':
    # 替换 'your_data.xlsx' 为你的 Excel 文件实际路径
    # 如果你的数据在名为 'Temperature' 的表格中，sheet_name 就设置为 'Temperature'
    # 确保这个文件已经存在并且包含前两列温度数据
    process_and_plot_temperatures_from_excel(
        excel_file_path='experiment_data.xlsx', # 假设这是你之前生成的文件
        sheet_name='Throughput_Dialogue' # 假设温度数据在名为 'Temperature' 的表格中
    )

    print("\n数据处理完成。请检查 Excel 文件以查看新添加的 'Temperature_Device_C' 列。")