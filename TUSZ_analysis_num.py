import pandas as pd
import os
from collections import defaultdict

def process_single_file_for_seizure_stats(file_path):
    """
    Processes a single CSV file to extract seizure information.
    Returns:
        tuple: (bname, subject_id, {seizure_type: count_in_file})
               Returns (None, None, None) if the file is invalid or contains no seizure labels.
    """
    try:
        # Load the CSV file, skipping initial rows as identified previously.
        # Adjust 'skiprows' if your files have a different header structure.
        df = pd.read_csv(file_path, skiprows=5)
    except pd.errors.ParserError as e:
        print(f"Error parsing {file_path}: {e}. Skipping this file.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}. Skipping this file.")
        return None, None, None

    if 'label' not in df.columns:
        print(f"Required column 'label' not found in {file_path}. Skipping.")
        return None, None, None
        
    df['label'] = df['label'].astype(str)

    bname = os.path.basename(file_path)
    
    # Assuming subject_id is the prefix of the filename, e.g., 'aaaaaaav_s001_t000.csv' -> 'aaaaaaav'
    # You might need to adjust this logic based on your actual filename format.
    subject_id = bname.split('_')[0] 

    seizure_events_in_file = df[df['label'] != 'bckg']
    
    if seizure_events_in_file.empty:
        return None, None, None

    # Count occurrences of each seizure label in the file
    seizure_type_counts = seizure_events_in_file['label'].value_counts().to_dict()

    return bname, subject_id, seizure_type_counts


def main():
    # Set your dataset's root directory here
    root_directory = "E:\\TUSZ\\edf\\"  # IMPORTANT: Change this to your dataset's actual root directory

    # Store statistics:
    # key: seizure_type (str)
    # value: dict { 'subjects': set(), 'recordings': set(), 'events': int }
    seizure_stats = defaultdict(lambda: {'subjects': set(), 'recordings': set(), 'events': 0})

    print(f"Starting to process CSV files under: {root_directory}")

    # Walk through the root directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                print(f"Processing file: {file_path}")
                
                bname, subject_id, seizure_type_counts = process_single_file_for_seizure_stats(file_path)
                
                if seizure_type_counts: # If the file contains seizure events
                    for seizure_type, count in seizure_type_counts.items():
                        seizure_stats[seizure_type]['subjects'].add(subject_id)
                        seizure_stats[seizure_type]['recordings'].add(bname)
                        seizure_stats[seizure_type]['events'] += count

    print("\n--- Processing Complete ---")
    print("\nSeizure Statistics:")

    if seizure_stats:
        # Convert results to a DataFrame for better presentation
        summary_data = []
        for s_type, stats in seizure_stats.items():
            summary_data.append({
                'Seizure Type': s_type,
                'Subject Count': len(stats['subjects']),
                'Recording Count': len(stats['recordings']),
                'Event Count': stats['events']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(by='Event Count', ascending=False) # Sort by event count in descending order
        
        print(summary_df.to_string(index=False)) # Print the entire DataFrame without index
    else:
        print("No seizure recordings found in the dataset.")

if __name__ == "__main__":
    main()