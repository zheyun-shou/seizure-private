import os
import pandas as pd

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
    
    

if __name__ == "__main__":
    # Example usage:
    # Replace with your actual CSV path
    csv_file_path = "D:/seizure/results/en_d_mini_multi_tusz_221_0.5_2/results.csv"
    process_score(csv_file_path)