from dataloader import get_ids_from_filename
import os
import pandas as pd
from mne.io import read_raw_edf
import sklearn

# if __name__ == "__main__":
#     # Test get_ids_from_filename
#     bids_root = 'F:\BIDS_TUSZ' # Replace with your actual path
#     recording_ids = []
#     bckg_df, seiz_df = [], []
#     event_cnt = 0
#     for root, dirs, files in os.walk(bids_root):
#         for file in files:
#             if file.endswith(".edf"):
#                 # print(file)
#                 base = file[:-8]  # remove "_eeg.edf" from the end
#                 json_file = base + "_eeg.json"
#                 tsv_file = base + "_events.tsv"

                
#                 events_df = pd.read_csv(os.path.join(root, tsv_file), sep='\t')
#                 # Check if the relevant columns exist in the dataframe
#                 required_columns = ['onset', 'duration', 'channels']
#                 if not all(col in events_df.columns for col in required_columns):
#                     raise ValueError(f"One or more of the required columns {required_columns} not found in the file.")
                
#                 # Extract the relevant columns: onset, duration, and channel
#                 for i, row in events_df.iterrows():
#                     #save the i and row["duration"] to csv when event_type is bckg
#                     event_cnt += 1 
#                     row["subject_id"] = base.split("_")[0]
#                     row["session_id"] = base.split("_")[1]
#                     row["task_id"] = base.split("_")[2]
#                     row["run_id"] = base.split("_")[3]
#                     if row["eventType"] == "bckg":
#                         bckg_df.append(row)
#                     elif row["eventType"] == "seiz":
#                         seiz_df.append(row)
#     print(event_cnt)
#     pd.DataFrame(bckg_df).to_csv("bckg.csv", index=False)
#     pd.DataFrame(seiz_df).to_csv("seiz.csv", index=False)
                    
    

# if __name__ == "__main__":
#     # count the number of sz recording and bckg recording
#     bids_root = 'F:\BIDS_Siena' # Replace with your actual path
#     sz_cnt, bckg_cnt = 0, 0
#     for root, dirs, files in os.walk(bids_root):
#         for file in files:
#             if file.endswith(".edf"):
#                 # print(file)
#                 base = file[:-8]  # remove "_eeg.edf" from the end
#                 json_file = base + "_eeg.json"
#                 tsv_file = base + "_events.tsv"

                
#                 events_df = pd.read_csv(os.path.join(root, tsv_file), sep='\t')
#                 # Check if the relevant columns exist in the dataframe
#                 required_columns = ['onset', 'duration', 'channels']
#                 if not all(col in events_df.columns for col in required_columns):
#                     raise ValueError(f"One or more of the required columns {required_columns} not found in the file.")
                
#                 # Extract the relevant columns: onset, duration, and channel
#                 sz_per_recording = 0
#                 bckg_per_recording = 0
#                 # for i, row in events_df.iterrows():
                    
#                 #     #save the i and row["duration"] to csv when event_type is bckg
#                 #     if row["eventType"] == "bckg":
#                 #         bckg_per_recording += 1
#                 #         if bckg_per_recording > 1:
#                 #             continue
#                 #         bckg_cnt += 1 
#                 #     elif row["eventType"] == "sz":
#                 #         sz_per_recording += 1
#                 #         if sz_per_recording > 1:
#                 #             continue
#                 #         sz_cnt += 1
                
#                 # search if the eventType is bckg or sz in the events_df
#                 if "bckg" in events_df["eventType"].values:
#                     bckg_cnt += 1
#                 # if "sz" in events_df["eventType"].values:
#                 if events_df["eventType"].str.contains("sz").any():
#                     sz_cnt += 1
                    
#     print("sz_cnt:", sz_cnt)
#     print("bckg_cnt:", bckg_cnt)

import matplotlib.pyplot as plt
import numpy as np

def plot_subject_f1_scores(subject_ids, f1_scores, title='Subject-wise F1 Score Distribution',
                           xlabel='Subject ID', ylabel='F1 Score', bar_color='skyblue',
                           figsize=(12, 7), show_mean=True, show_median=True):
    """
    Generates a bar plot showing the F1 score for each subject.

    Args:
        subject_ids (list or array-like): A list of subject identifiers (e.g., ['S1', 'S2', ...]).
        f1_scores (list or array-like): A list of F1 scores corresponding to each subject.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        bar_color (str): The color of the bars.
        figsize (tuple): The size of the figure (width, height).
        show_mean (bool): If True, a horizontal line for the mean F1 score will be shown.
        show_median (bool): If True, a horizontal line for the median F1 score will be shown.
    """
    if len(subject_ids) != len(f1_scores):
        raise ValueError("subject_ids and f1_scores must have the same length.")

    # Sort subjects by F1 score for better visualization (optional)
    # sorted_indices = np.argsort(f1_scores)
    # subject_ids_sorted = np.array(subject_ids)[sorted_indices]
    # f1_scores_sorted = np.array(f1_scores)[sorted_indices]

    # For this example, we'll plot them as provided
    subject_ids_sorted = subject_ids
    f1_scores_sorted = f1_scores

    plt.style.use('seaborn-v0_8-whitegrid') # Using a pleasant style

    plt.figure(figsize=figsize)

    # Create the bar plot
    bars = plt.bar(subject_ids_sorted, f1_scores_sorted, color=bar_color, edgecolor='black')

    # Add F1 score values on top of each bar for clarity
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}',
                 ha='center', va='bottom', fontsize=9)

    # Add horizontal lines for mean and median F1 score
    if show_mean:
        mean_f1 = np.mean(f1_scores)
        plt.axhline(mean_f1, color='red', linestyle='--', linewidth=1.5, label=f'Mean F1: {mean_f1:.2f}')

    if show_median:
        median_f1 = np.median(f1_scores)
        plt.axhline(median_f1, color='green', linestyle=':', linewidth=1.5, label=f'Median F1: {median_f1:.2f}')

    # Set plot title and labels
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Set y-axis limits (typically 0 to 1 for F1 score)
    plt.ylim(0, 1.05) # Extend a bit beyond 1.0 for text visibility

    # Rotate x-axis labels if there are many subjects to prevent overlap
    if len(subject_ids) > 10:
        plt.xticks(rotation=45, ha='right', fontsize=10)
    else:
        plt.xticks(fontsize=10)

    plt.yticks(fontsize=10)

    if show_mean or show_median:
        plt.legend(fontsize=10)

    plt.tight_layout() # Adjust layout to prevent labels from being cut off
    plt.show()

# --- Example Usage ---
# if __name__ == '__main__':
#     # Example data: Replace with your actual subject IDs and their F1 scores
#     # Let's assume you have F1 scores for 15 subjects
#     num_subjects = 15
#     example_subject_ids = [f'Subject_{i+1}' for i in range(num_subjects)]
#     # Generate some random F1 scores between 0.5 and 0.95 for demonstration
#     np.random.seed(42) # for reproducibility
#     example_f1_scores = np.random.uniform(low=0.50, high=0.95, size=num_subjects)
#     example_f1_scores = np.round(example_f1_scores, 2)

#     print("Example Subject IDs:", example_subject_ids)
#     print("Example F1 Scores:", example_f1_scores)

#     # Generate the plot
#     plot_subject_f1_scores(example_subject_ids, example_f1_scores,
#                            title='Model Performance (F1 Score) per Subject on TUSZ Dataset',
#                            xlabel='Test Subject ID',
#                            ylabel='Event-wise F1 Score')

#     # Example with fewer subjects, not sorted
#     example_subject_ids_short = ['S1', 'S5', 'S3', 'S2', 'S4']
#     example_f1_scores_short = [0.85, 0.72, 0.91, 0.65, 0.78]
#     plot_subject_f1_scores(example_subject_ids_short, example_f1_scores_short,
#                            title='Subject Performance (Unsorted)',
#                            bar_color='lightcoral')

#     # You can also create a histogram to see the distribution
#     plt.figure(figsize=(8, 5))
#     plt.hist(example_f1_scores, bins=5, color='teal', edgecolor='black')
#     plt.title('Histogram of Subject F1 Scores', fontsize=15)
#     plt.xlabel('F1 Score', fontsize=12)
#     plt.ylabel('Number of Subjects', fontsize=12)
#     plt.grid(axis='y', alpha=0.75)
#     plt.tight_layout()
#     plt.show()

if __name__ == "__main__":
    BIDS_ROOT = 'F:\BIDS_TUSZ'  # Replace with your actual path
    # calcutale how many seizure recordings a subject has, print the subject_id and sz_count for each subject
    sz_count = {}
    bckg_count = {}
    seizure_subject=0
    for root, dirs, files in os.walk(BIDS_ROOT):
        for file in files:
            if file.endswith(".edf"):
                base = file[:-8]  # remove "_eeg.edf" from the end
                json_file = base + "_eeg.json"
                tsv_file = base + "_events.tsv"

                events_df = pd.read_csv(os.path.join(root, tsv_file), sep='\t')
                # Check if the relevant columns exist in the dataframe
                required_columns = ['onset', 'duration', 'channels']
                if not all(col in events_df.columns for col in required_columns):
                    raise ValueError(f"One or more of the required columns {required_columns} not found in the file.")
                
                subject_id = base.split("_")[0]
                
                # Count the number of seizure recordings for this subject, not the number of seizures

                if subject_id not in sz_count:
                    sz_count[subject_id] = 0
                if "sz" in events_df["eventType"].values:
                    sz_count[subject_id] += 1
            
                if subject_id not in bckg_count:
                    bckg_count[subject_id] = 0
                if "bckg" in events_df["eventType"].values:
                    bckg_count[subject_id] += 1
                
    for subject_id, count in sz_count.items():
        if count > 0:
            seizure_subject += 1

                
    # for subject_id, count in sz_count.items():
    #     print(f"Subject ID: {subject_id}, Seizure Count: {count}")
        
    # print the distribution of seizure counts as a figure, use bin size of 5
    sz_counts = list(sz_count.values())
    # print number of subjects with seizure recording
    print(f"Number of subjects with seizure recording: {seizure_subject}")
    plt.figure(figsize=(10, 6))
    plt.hist(sz_counts, bins=range(0, max(sz_counts) + 5, 5), color='skyblue', edgecolor='black')
    plt.title('Distribution of Seizure Recordings per Subject', fontsize=16)
    plt.xlabel('Number of Seizure Recordings', fontsize=14)
    plt.ylabel('Number of Subjects', fontsize=14)

    
    plt.xticks(range(0, max(sz_counts) + 5, 5))
    plt.grid(axis='y', alpha=0.75)  
    plt.tight_layout()
    
    plt.show()
    
    # print the total number of seizure recordings and background recordings
    total_sz_recordings = sum(sz_count.values())
    total_bckg_recordings = sum(bckg_count.values())
    print(f"Total Seizure Recordings: {total_sz_recordings}")
    print(f"Total Background Recordings: {total_bckg_recordings}")
