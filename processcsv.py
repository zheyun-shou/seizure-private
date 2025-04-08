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
    avg_event_sensitivity  = df_no_nan["event_sensitivity"].mean()
    avg_event_precision    = df_no_nan["event_precision"].mean()
    avg_event_f1           = df_no_nan["event_f1"].mean()

    print("Averages from non-NaN rows:")
    print(f"sample_sensitivity: {avg_sample_sensitivity:.3f}")
    print(f"sample_precision:   {avg_sample_precision:.3f}")
    print(f"sample_f1:          {avg_sample_f1:.3f}")
    print(f"event_sensitivity:  {avg_event_sensitivity:.3f}")
    print(f"event_precision:    {avg_event_precision:.3f}")
    print(f"event_f1:           {avg_event_f1:.3f}")

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
    

# def process_predictions(test_seg_path, pred_path):
    
#     if not os.path.exists(test_seg_path):
#         print(f"File not found: {test_seg_path}")
#         return
#     if not os.path.exists(pred_path):
#         print(f"File not found: {pred_path}")
#         return
    
#     test_seg_df = pd.read_csv(test_seg_path)
#     predictions = pd.read_csv(pred_path)
    
    
#     new_test_seg_df = pd.DataFrame()
#     # concatenate the test_seg_df 
#     for keyword in ["label", "time_start", "time_end", "subject_id", "session_id", "task_id", "run_id"]:
#         # convert type of elements in dataframe, displayed as string, to list
#         test_seg_df[keyword] = test_seg_df[keyword].apply(
#             lambda s: list(map(str, s.strip("[]").replace("\n", " ").split()))
#         )
#         # print(test_seg_df[keyword].to_list())
#         new_test_seg_df[keyword] = np.concatenate(test_seg_df[keyword].to_list())
    
#     # find unique combination of "subject_id", "session_id", "task_id", "run_id"
#     unique_combinations = new_test_seg_df[["subject_id", "session_id", "task_id", "run_id"]].drop_duplicates()
#     # find rows in new_test_seg_df with same "subject_id", "session_id", "task_id", "run_id" with unique_combinations
#     for i, row in unique_combinations.iterrows():
#         subject_id = row["subject_id"]
#         session_id = row["session_id"]
#         task_id = row["task_id"]
#         run_id = row["run_id"]
#         # find rows in new_test_seg_df with same "subject_id", "session_id", "task_id", "run_id"
#         row_index = new_test_seg_df[(new_test_seg_df["subject_id"] == subject_id) & (new_test_seg_df["session_id"] == session_id) & (new_test_seg_df["task_id"] == task_id) & (new_test_seg_df["run_id"] == run_id)].index
#         time_starts = new_test_seg_df["time_start"].iloc[row_index].to_numpy().astype(float)
#         label = new_test_seg_df["label"].iloc[row_index].to_numpy().astype(float)
#         pred_rows = predictions.iloc[row_index, 1].to_numpy().astype(float)
#         # plot pred_rows with time_starts
#         fig = plt.figure()
#         plt.plot(time_starts, pred_rows, label="prediction")
#         plt.plot(time_starts, label, label="label")
#         plt.xlabel("Time start")
#         plt.ylabel("Prediction")
#         # set title with "subject_id", "session_id", "task_id", "run_id"
#         title = f"subject_{subject_id}_session_{session_id}_task_{task_id}_run_id_{run_id}"
#         plt.title(title)
#         plt.legend()
#         plt.savefig(f"D:/seizure/results/predictions/{title}_pred.png")
#         plt.close(fig)
    
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
            avg_sample_sensitivity_list.append(avg_sample_sensitivity)
            avg_sample_precision_list.append(avg_sample_precision)
            avg_sample_f1_list.append(avg_sample_f1)
            avg_event_sensitivity_list.append(avg_event_sensitivity)
            avg_event_precision_list.append(avg_event_precision)
            avg_event_f1_list.append(avg_event_f1)
            
    avg_score_per_subject_df = {
            "subject_id": subject_id_list,
            "avg_sample_sensitivity": avg_sample_sensitivity_list,
            "avg_sample_precision": avg_sample_precision_list,
            "avg_sample_f1": avg_sample_f1_list,
            "avg_event_sensitivity": avg_event_sensitivity_list,
            "avg_event_precision": avg_event_precision_list,
            "avg_event_f1": avg_event_f1_list
        }
    # save the average scores to csv
    avg_score_per_subject_df = pd.DataFrame(avg_score_per_subject_df)
    avg_score_per_subject_df.to_csv("D:/seizure/results/en_d_mini_multi_tusz_0406_TUSZ/avg_score_per_subject.csv", index=False)
    

if __name__ == "__main__":
    
    # csv_file_path = "D:/seizure/results/en_d_mini_multi_tusz_221_Siena_new/results.csv"
    # process_score(csv_file_path)
    #process_score_subject_wise(csv_file_path)
    # read all filenames under D:\seizure\results\recording_predictions
    csv_folder_path = "D:/seizure/results/probs_TUSZ_en_d_mini_multi_tusz_0406"
    for root, dirs, files in os.walk(csv_folder_path):
        for file in files:
            if file.endswith(".csv"):
                process_predictions(os.path.join(root, file))
    
    
#bckg：
# TP==0, FN==0

# if FP==0:
#     sen,pre,f1=nan, nan, nan
# else if FP!=0:
#     sen=nan, pre=0, f1=nan


# #seizure：
# if TP==0 and FN!=0 and FP==0:
#     sen=0, pre=nan, f1=nan
