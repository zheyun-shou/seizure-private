import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def process_score(csv_path):
    """
    Reads the CSV file, counts how many lines contain NaN values, and
    computes average values of the scoring columns from lines without NaN.
    """
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Count how many rows contain NaN in any column
    nan_rows = df.isna().any(axis=1).sum()
    print(f"Number of rows with NaN: {nan_rows}")

    # Filter out rows containing NaN
    df_no_nan = df.dropna()

    # Compute average values from rows without NaN
    avg_sample_sensitivity = df_no_nan["sample_sensitivity"].mean()
    avg_sample_precision   = df_no_nan["sample_precision"].mean()
    avg_sample_f1          = df_no_nan["sample_f1"].mean()
    avg_sample_fpRate      = df_no_nan["sample_fpRate"].mean()
    avg_event_sensitivity  = df_no_nan["event_sensitivity"].mean()
    avg_event_precision    = df_no_nan["event_precision"].mean()
    avg_event_f1           = df_no_nan["event_f1"].mean()
    avg_event_fpRate       = df_no_nan["event_fpRate"].mean()

    print("Averages from non-NaN rows:")
    print(f"sample_sensitivity: {avg_sample_sensitivity:.3f}")
    print(f"sample_precision:   {avg_sample_precision:.3f}")
    print(f"sample_f1:          {avg_sample_f1:.3f}")
    print(f"sample_fpRate:      {avg_sample_fpRate:.3f}")
    print(f"event_sensitivity:  {avg_event_sensitivity:.3f}")
    print(f"event_precision:    {avg_event_precision:.3f}")
    print(f"event_f1:           {avg_event_f1:.3f}")
    print(f"event_fpRate:       {avg_event_fpRate:.3f}")

def process_ts(csv_path):
    # calculate the sum of the first column and the sum of the second column
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    sum_first_col = df["non_seizure_length"].sum()
    sum_second_col = df["seizure_length"].sum()
    print(f"Sum of non_seizure_length: {sum_first_col}")
    print(f"Sum of seizure_length: {sum_second_col}")
    
    
def process_predictions(file_path):
    # plot label and prediction with x-axis as time_start
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    df = pd.read_csv(file_path)
    time_starts = df["time_start"].to_numpy().astype(float)
    label = df["label"].to_numpy().astype(float)
    prediction = df["prediction"].to_numpy().astype(float)
    # save plot as png
    fig = plt.figure()
    plt.plot(time_starts, prediction, label="prediction probability") 
    plt.plot(time_starts, label, label="label")
    plt.xlabel("Time start")
    plt.ylabel("Probability")
    title = file_path.split("\\")[-1].split(".")[0]
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"{csv_folder_path}/{title}_prob.png")
    plt.close(fig)
    
def process_score_subject_wise(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    subject_id_list = []
    
    avg_sample_sensitivity_list, avg_sample_precision_list, avg_sample_f1_list = [], [], []
    avg_event_sensitivity_list, avg_event_precision_list, avg_event_f1_list = [], [], []
    avg_sample_fpRate_list, avg_event_fpRate_list = [], []
    # calculate the scores for each subject
    df = pd.read_csv(file_path)
    for i, row in df.iterrows():
        subject_id = row["subject_id"]
        if subject_id not in subject_id_list:
            subject_id_list.append(subject_id)
            # calculate the scores for this subject
            # find the rows with this subject id
            subject_df = df[df["subject_id"] == subject_id]
            # calculate the scores
            # if the score is not NaN, calculate the average
            
            subject_df = subject_df.dropna()
            avg_sample_sensitivity = subject_df["sample_sensitivity"].mean()
            avg_sample_precision   = subject_df["sample_precision"].mean()
            avg_sample_f1          = subject_df["sample_f1"].mean()
            avg_event_sensitivity  = subject_df["event_sensitivity"].mean()
            avg_event_precision    = subject_df["event_precision"].mean()
            avg_event_f1           = subject_df["event_f1"].mean()
            avg_sample_fpRate      = subject_df["sample_fpRate"].mean()
            avg_event_fpRate       = subject_df["event_fpRate"].mean()
            avg_sample_sensitivity_list.append(avg_sample_sensitivity)
            avg_sample_precision_list.append(avg_sample_precision)
            avg_sample_f1_list.append(avg_sample_f1)
            avg_event_sensitivity_list.append(avg_event_sensitivity)
            avg_event_precision_list.append(avg_event_precision)
            avg_event_f1_list.append(avg_event_f1)
            avg_sample_fpRate_list.append(avg_sample_fpRate)
            avg_event_fpRate_list.append(avg_event_fpRate)
            
    avg_score_per_subject_df = {
            "subject_id": subject_id_list,
            "avg_sample_sensitivity": avg_sample_sensitivity_list,
            "avg_sample_precision": avg_sample_precision_list,
            "avg_sample_f1": avg_sample_f1_list,
            "avg_sample_fpRate": avg_sample_fpRate_list,
            "avg_event_sensitivity": avg_event_sensitivity_list,
            "avg_event_precision": avg_event_precision_list,
            "avg_event_f1": avg_event_f1_list,
            "avg_event_fpRate": avg_event_fpRate_list
        }
    # save the average scores to csv
    avg_score_per_subject_df = pd.DataFrame(avg_score_per_subject_df)
    print(os.path.dirname(file_path))
    avg_csv_path = os.path.join(os.path.dirname(file_path), "avg_score_per_subject.csv")
    avg_score_per_subject_df.to_csv(avg_csv_path, index=False)
    
def find_duplicates(csv_path):
    """
    Find duplicate rows in the CSV file
    """
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    duplicates = df[df.duplicated(keep=False)]  # keep=False to mark all duplicates
    if not duplicates.empty:
        print("Duplicate rows found:")
        print(duplicates)
    else:
        print("No duplicate rows found.")

def plot_f1_scores_histogram(f1_scores, title='Histogram of Subject FPR',
                             xlabel='Event-wise FPR', ylabel='Number of Subjects',
                             color='dimgray', edgecolor='white', bins=10, figsize=(8,5)):
    """
    Generates a histogram of F1 scores.

    Args:
        f1_scores (list or array-like): A list of F1 scores.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        color (str): The color of the histogram bars.
        edgecolor (str): The color of the bar edges.
        bins (int or sequence): The number of bins or bin edges for the histogram.
        figsize (tuple): The size of the figure (width, height).
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=figsize)
    
    # Filter out NaN values before plotting, if any still exist
    f1_scores_cleaned = [score for score in f1_scores if not np.isnan(score)]
    if not f1_scores_cleaned:
        print("Warning: No valid F1 scores to plot after cleaning.")
        return

    plt.hist(f1_scores_cleaned, bins=bins, color=color, edgecolor=edgecolor)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    custom_x_ticks = np.arange(0, 1.01, 0.1)
    plt.xticks(custom_x_ticks)
    plt.xlim(0, 1)  # Set x-axis limits to [0, 1] for F1 scores
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    csv_file_path = "D:/seizure/results/0521_en_mini_datasize0.5odd_epoch6_TUSZ/results_w_fp.csv"

    process_score(csv_file_path)
    # process_score_subject_wise(csv_file_path)
    
    # read all filenames under D:\seizure\results\recording_predictions
    # csv_folder_path = "D:/seizure/results/probs_TUSZ_en_d_mini_multi_tusz_0406"
    # for root, dirs, files in os.walk(csv_folder_path):
    #     for file in files:
    #         if file.endswith(".csv"):
    #             process_predictions(os.path.join(root, file))
    
    ## plot subject-wise f1 scores
    # csv_file_path = 'D:/seizure/results/en_d_mini_multi_tusz_0406_TUSZ/avg_score_per_subject.csv' # <--- !!! REPLACE THIS WITH THE ACTUAL PATH TO YOUR CSV FILE !!!

    # --- Create a dummy CSV for demonstration if it doesn't exist ---
    # try:
    #     # Try to read the CSV, if it fails, create a dummy one.
    #     data = pd.read_csv(csv_file_path)
    # except FileNotFoundError:
    #     print(f"Warning: '{csv_file_path}' not found.")

    # # --- Step 3: Read the CSV file using pandas ---
    # try:
    #     data = pd.read_csv(csv_file_path)
        
    #     # --- Step 4: Extract F1 scores and handle NaN values ---

    #     # if 'avg_event_fpRate' in data.columns:
    #     #     fpr_from_csv = data['avg_event_fpRate'].dropna().tolist() # Drop NaN and convert to list
    #     #     subject_ids_from_csv = data.loc[data['avg_event_fpRate'].notna(), 'subject_id'].tolist() # Get corresponding subject_ids

    #     #     if not fpr_from_csv:
    #     #         print("No valid F1 scores found in the CSV after dropping NaN values.")
    #     #     else:
    #     #         print(f"\nRead {len(fpr_from_csv)} valid F1 scores from '{csv_file_path}'.")
    #     #         print("F1 Scores:", fpr_from_csv)

    #     #         # --- Step 5: Plot the histogram ---
    #     #         custom_bin_edges = np.arange(0, 301, 30) 
    #     #         plot_f1_scores_histogram(fpr_from_csv,
    #     #                                  title='Distribution of Subject FPR',
    #     #                                  bins=custom_bin_edges,
    #     #                                  xlim=300) # You can adjust the number of bins


    #     # else:
    #     #     print(f"Error: Column 'f1_score' not found in '{csv_file_path}'. Please check your CSV file.")
        
    #     if 'avg_event_f1' in data.columns:
    #         f1_from_csv = data['avg_event_f1'].dropna().tolist()
    #         subject_ids_from_csv = data.loc[data['avg_event_f1'].notna(), 'subject_id'].tolist()
            
    #         if not f1_from_csv:
    #             print("No valid F1 scores found in the CSV after dropping NaN values.")
    #         else:
    #             print(f"\nRead {len(f1_from_csv)} valid F1 scores from '{csv_file_path}'.")
    #             print("F1 Scores:", f1_from_csv)

    #             # --- Step 5: Plot the histogram ---
    #             custom_bin_edges = np.arange(0, 1.01, 0.1) 
    #             plot_f1_scores_histogram(f1_from_csv,
    #                                      title='Distribution of Subject F1',
    #                                      xlabel='Event-wise F1',
    #                                      bins=custom_bin_edges)

    # except FileNotFoundError:
    #     print(f"Error: The file '{csv_file_path}' was not found. Please ensure the path is correct.")
    #     print("If you intended to use the dummy CSV, it should have been created. Try running again.")
    # except pd.errors.EmptyDataError:
    #     print(f"Error: The file '{csv_file_path}' is empty.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")