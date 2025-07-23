import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict # 引入defaultdict便于计数

def process_csv_file(file_path):
    """
    处理单个CSV文件，提取bname, seizure types, and channels with only 'bckg' labels.
    一个通道如果出现过任何非bckg的label，就不算作bckg only通道。
    返回：
        file_info (dict): 包含 'bname', 'seizure_type', 'channel' 的字典。
        bckg_only_channels (list): 当前文件中只包含'bckg'标签的通道列表。
        seizure_labels_in_file (list): 当前文件中检测到的所有癫痫标签列表。
    """
    try:
        # Load the CSV file, skipping the estimated rows above header.
        # This assumes the actual header is on the first row after skipping (e.g., line 6).
        # You might need to adjust 'skiprows' based on your actual file structure.
        df = pd.read_csv(file_path, skiprows=5)
    except pd.errors.ParserError as e:
        print(f"Error parsing {file_path}: {e}. Skipping this file.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}. Skipping this file.")
        return None, None, None

    # Ensure 'label' column is string type before applying filters
    if 'label' not in df.columns or 'channel' not in df.columns:
        print(f"Required columns 'label' or 'channel' not found in {file_path}. Skipping.")
        return None, None, None
        
    df['label'] = df['label'].astype(str)

    # 识别文件中所有的非'bckg'标签，这些都视为癫痫标签
    all_unique_labels_in_file = df['label'].unique()
    seizure_labels_in_file = [label for label in all_unique_labels_in_file if label != 'bckg']

    if not seizure_labels_in_file: # 如果文件中没有任何癫痫标签，则此文件不满足“含有seizure label”的条件
        return None, None, None

    bname = os.path.basename(file_path)
    
    all_channels = df['channel'].unique()
    bckg_only_channels = []
    
    for channel_val in all_channels:
        # 获取当前通道的所有唯一标签
        channel_labels = df[df['channel'] == channel_val]['label'].unique()
        
        # 检查这个通道是否包含任何癫痫标签
        has_seizure_label = False
        for label in channel_labels:
            if label in seizure_labels_in_file: # 如果这个通道的任何一个标签是非bckg的，则认为它包含癫痫
                has_seizure_label = True
                break
        
        # 如果这个通道没有任何癫痫标签 (即所有标签都是'bckg')，则将其添加到结果中
        if not has_seizure_label:
            # 进一步确认，确保它至少有'bckg'标签
            if 'bckg' in channel_labels:
                bckg_only_channels.append(channel_val)
    
    # 只有当文件确实含有seizure labels AND 找到了符合条件的bckg only channels时才返回数据
    if seizure_labels_in_file and bckg_only_channels:
        return {
            'bname': bname,
            'seizure_type': ', '.join(seizure_labels_in_file), # 包含所有癫痫类型，便于输出展示
            'channel': bckg_only_channels
        }, bckg_only_channels, seizure_labels_in_file
    else:
        return None, None, None


def main():
    # Set your root directory here
    root_directory = "E:\\TUSZ\\edf\\"  # IMPORTANT: Change this to your dataset's root directory

    # 存储所有文件中符合条件的bckg only通道，按癫痫类型分类
    # key: seizure_type (str), value: list of bckg_channel_names (list of str)
    seizure_type_bckg_channel_map = defaultdict(list)
    processed_files_info = [] # 存储每个文件的详细处理信息

    print(f"Starting to process CSV files under: {root_directory}")

    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                print(f"Processing file: {file_path}")
                
                file_info, bckg_channels_from_current_file, seizure_labels_from_current_file = process_csv_file(file_path)
                
                if file_info: # 如果文件被成功处理并找到了符合条件的数据
                    processed_files_info.append(file_info)
                    
                    # 将当前文件的bckg only通道添加到每种癫痫类型下
                    for s_label in seizure_labels_from_current_file:
                        seizure_type_bckg_channel_map[s_label].extend(bckg_channels_from_current_file)

    print("\n--- Processing Complete ---")
    print("\nFiltered results for each file:")
    if processed_files_info:
        for info in processed_files_info:
            print(f"Bname: {info['bname']}, Seizure Type: {info['seizure_type']}, Channel: {info['channel']}")
    else:
        print("No files with seizure labels and background-only channels found based on the criteria.")

    # 统计并绘制不同癫痫类型下的通道分布
    if seizure_type_bckg_channel_map:
        print("\nDistribution of background-only channels under different seizure types:")
        for seizure_type, bckg_channels_list in seizure_type_bckg_channel_map.items():
            if bckg_channels_list:
                channel_counts = pd.Series(bckg_channels_list).value_counts()
                print(f"\nSeizure Type: {seizure_type}")
                print(channel_counts)

                # Plot the bar chart for current seizure type
                plt.figure(figsize=(12, 7))
                channel_counts.plot(kind='bar')
                plt.title(f'Count of Background Channels for Seizure Type: {seizure_type}')
                plt.xlabel('Channel Name')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                chart_filename = f'background_channel_counts_{seizure_type}.png'
                plt.savefig(chart_filename)
                print(f"Bar chart saved as '{chart_filename}'")
            else:
                print(f"\nNo background channels found for seizure type: {seizure_type}")
    else:
        print("\nNo background channels found in seizure-labeled files to plot from the entire dataset.")

if __name__ == "__main__":
    main()