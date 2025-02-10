from dataloader import get_ids_from_filename
import os

if __name__ == "__main__":
    # Test get_ids_from_filename
    bids_root = 'E:\BIDS_TUSZ' # Replace with your actual path
    recording_ids = []
    for root, dirs, files in os.walk(bids_root):
        for file in files:
            if file.endswith(".edf"):
                print(file)
                base = file[:-8]  # remove "_eeg.edf" from the end
                json_file = base + "_eeg.json"
                tsv_file = base + "_events.tsv"
                if os.path.exists(os.path.join(root, tsv_file)) and os.path.exists(os.path.join(root, json_file)):
                    subject_id, session_id, task_id, run_id = get_ids_from_filename(file)
                    # create a tree-like dictionary to store the data
                    ids = {
                        'subject_id': subject_id,
                        'session_id': session_id,
                        'task_id': task_id,
                        'run_id': run_id,
                    }
                    recording_ids.append(ids)
                    #print(ids['subject_id'], ids['session_id'], ids['task_id'], ids['run_id'])

    