from dataloader import get_ids_from_filename
import os
import pandas as pd

if __name__ == "__main__":
    # Test get_ids_from_filename
    bids_root = 'F:\BIDS_TUSZ' # Replace with your actual path
    recording_ids = []
    bckg_df, seiz_df = [], []
    event_cnt = 0
    for root, dirs, files in os.walk(bids_root):
        for file in files:
            if file.endswith(".edf"):
                # print(file)
                base = file[:-8]  # remove "_eeg.edf" from the end
                json_file = base + "_eeg.json"
                tsv_file = base + "_events.tsv"

                
                events_df = pd.read_csv(os.path.join(root, tsv_file), sep='\t')
                # Check if the relevant columns exist in the dataframe
                required_columns = ['onset', 'duration', 'channels']
                if not all(col in events_df.columns for col in required_columns):
                    raise ValueError(f"One or more of the required columns {required_columns} not found in the file.")
                
                # Extract the relevant columns: onset, duration, and channel
                for i, row in events_df.iterrows():
                    #save the i and row["duration"] to csv when event_type is bckg
                    event_cnt += 1 
                    row["subject_id"] = base.split("_")[0]
                    row["session_id"] = base.split("_")[1]
                    row["task_id"] = base.split("_")[2]
                    row["run_id"] = base.split("_")[3]
                    if row["eventType"] == "bckg":
                        bckg_df.append(row)
                    elif row["eventType"] == "seiz":
                        seiz_df.append(row)
    print(event_cnt)
    pd.DataFrame(bckg_df).to_csv("bckg.csv", index=False)
    pd.DataFrame(seiz_df).to_csv("seiz.csv", index=False)
                    
    

    