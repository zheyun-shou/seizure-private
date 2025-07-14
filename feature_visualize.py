from sktime.transformations.panel.catch22 import Catch22
import pandas as pd
import os
import numpy as np
from dataloader import get_ids_from_filename, get_path_from_ids, extract_event_info, extract_epochs, get_data_from_epochs
import gc

def visualize_nd(X, y, color,reducer_type='tsne', problem='Skoda', n_components=2, n_points=1000, random_state=1234, id=None, root_path=None):
    import os
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    # from umap import UMAP
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    assert reducer_type in ['tsne', 'umap', 'pca']
    assert problem in ['Skoda', 'PAMAP2', 'Oppotunity', 'USC_HAD', 'WISDM', 'WISDM2', 'EEG']
    assert n_components > 1, "n_components should be at least 2 for pairwise plotting."
    assert (reducer_type != 'tsne' or n_components < 4), "n_components should be inferior to 4 for tsne."
    random.seed(random_state)

    X_repr = X
    y_repr = y
    sample_idx = random.sample(range(len(y_repr)), min(n_points, len(y_repr)))

    # Initialize the dimensionality reducer
    # if reducer_type == 'umap':
    #     reducer = UMAP(n_components=n_components)
    if reducer_type == 'tsne':
        reducer = TSNE(n_components=n_components)
    elif reducer_type == 'pca':
        reducer = PCA(n_components=n_components)
    X_reduced = reducer.fit_transform(X_repr[sample_idx, :])

    # Create a DataFrame for easier plotting
    columns = [f'dim{i+1}' for i in range(n_components)]
    df = pd.DataFrame(X_reduced, columns=columns)
    df['class'] = y_repr[sample_idx]
    df['color'] = color[sample_idx]
    
    if n_components == 3:
        fig = plt.figure(figsize=(8, 8)) 
        sns.set_style("whitegrid", {'axes.grid' : False})
        ax = plt.axes(projection="3d")
        sc = ax.scatter(df['dim1'], df['dim2'], df['dim3'], c=df['color'], marker='o')
        plt.legend(*sc.legend_elements())
        plt.show()
    else:
        # Plot pairwise scatter plots in a matrix format
        num_plots = (n_components * (n_components - 1)) // 2
        fig, axes = plt.subplots(n_components - 1, n_components - 1, figsize=(3*n_components, 3*n_components), constrained_layout=True)
        fig.suptitle(f'{reducer_type} - {n_components}D Pairwise Scatter Plots - {id}', fontsize=16)


        # Generate pairwise scatter plots
        for i in range(n_components - 1):
            for j in range(i + 1, n_components):
                ax = axes[i, j - 1] if n_components > 2 else axes
                # sns.scatterplot(data=df, x=f'dim{i+1}', y=f'dim{j+1}', hue='class', ax=ax, legend=False)
                g = sns.scatterplot(data=df, x=f'dim{i+1}', y=f'dim{j+1}', hue='color', ax=ax, legend=False)
                ax.set_title(f'Dim {i+1} vs Dim {j+1}')
                ax.set_xlabel(f'Dim {i+1}')
                ax.set_ylabel(f'Dim {j+1}')

        # Hide unused subplots
        for i in range(n_components - 1):
            for j in range(i):
                axes[i, j].axis('off')

        # Save the figure
        
        os.makedirs(os.path.join(root_path, 'visualize'), exist_ok=True)
        fig.savefig(os.path.join(root_path, 'visualize', f'{reducer_type}_{n_components}D_{id}.png'))

if __name__ == '__main__':
    problem = "EEG"
    root_path = "D:/seizure"
    bids_root = "F:/BIDS_TUSZ"
    avg_score_per_subject_path = 'D:/seizure/results/0529_catch22_seed41_TUSZ/subject_s41_catch22.csv'
    #read subject_id from csv
    subject_ids = pd.read_csv(avg_score_per_subject_path)['subject_id'].astype(int).tolist()
    f1_list = pd.read_csv(avg_score_per_subject_path)['avg_event_f1'].astype(float).tolist()
    # for subject_id in subject_ids:
    #     if subject_id < 100 and subject_id > 9:
    #         subject_id = f'0{subject_id}'
    #     elif subject_id < 10:
    #         subject_id = f'00{subject_id}'
    #     subject_path = os.path.join(bids_root, f'sub-{subject_id}')
    #     segments = []
    #     for root, dirs, files in os.walk(subject_path):
    #         files.sort()
    #         for file in files:
    #             if file.endswith('.edf'):
    #                 subject_id, session_id, task_id, run_id = get_ids_from_filename(file)
    #                 tsv_path = os.path.join(subject_path,f'ses-{session_id}',f'eeg',f'sub-{subject_id}_ses-{session_id}_task-{task_id}_run-{run_id}_events.tsv')
    #                 edf_path = os.path.join(subject_path,f'ses-{session_id}',f'eeg',f'sub-{subject_id}_ses-{session_id}_task-{task_id}_run-{run_id}_eeg.edf')
    #                 events_info = extract_event_info(tsv_path, 10)
    #                 epochs = extract_epochs(edf_path, events_info, inference=True)
    #                 segment = get_data_from_epochs(epochs)
    #                 segments.extend(segment)
        
    #     # Process all segments after collecting them
    #     if segments:  # Only process if we have segments
    #         X_test = np.concatenate([s["epoch"] for s in segments]).astype(np.float32)
    #         y_test = np.concatenate([s["label"] for s in segments]).astype(int)
    #         del segments    
    #         gc.collect()

    #         c22_mv = Catch22(replace_nans=True)
    #         transformed_data_df = c22_mv.fit_transform(X_test)
    #         transformed_data = transformed_data_df.values


    #         visualize_nd(X=transformed_data, 
    #         y=y_test,
    #         reducer_type='tsne', # 'tsne', 'umap', 'pca'
    #         problem=problem, 
    #         n_components=2, # 2, 3
    #         n_points=1000,
    #         id = subject_id
    #         )
    #     else:
    #         print("No segments found for this subject!")   
    segments = []
    
    for subject_id in subject_ids:
        if subject_id < 100 and subject_id > 9:
            subject_id = f'0{subject_id}'
        elif subject_id < 10:
            subject_id = f'00{subject_id}'
        subject_path = os.path.join(bids_root, f'sub-{subject_id}')
        for root, dirs, files in os.walk(subject_path):
            files.sort()
            for file in files:
                if file.endswith('.edf'):
                    subject_id, session_id, task_id, run_id = get_ids_from_filename(file)
                    tsv_path = os.path.join(subject_path,f'ses-{session_id}',f'eeg',f'sub-{subject_id}_ses-{session_id}_task-{task_id}_run-{run_id}_events.tsv')
                    edf_path = os.path.join(subject_path,f'ses-{session_id}',f'eeg',f'sub-{subject_id}_ses-{session_id}_task-{task_id}_run-{run_id}_eeg.edf')
                    events_info = extract_event_info(tsv_path, 10)
                    epochs = extract_epochs(edf_path, events_info, inference=True)
                    segment = get_data_from_epochs(epochs)
                    segments.extend(segment)
        
    
    if segments:  # Only process if we have segments
        #save segments to numpy file
        try:
            np.save(os.path.join(root_path, 'visualize', f'segments_21subwithseizure.npy'), segments)
        except:
            print(f"Error saving segments for subject {subject_id}")

        X_test = np.concatenate([s["epoch"] for s in segments]).astype(np.float32)
        y_test = np.concatenate([s["label"] for s in segments]).astype(int)
        subject_id_list = np.concatenate([s["subject"] for s in segments]).astype(int)
        # assign f1 score to subject_id_list from f1_list, note that f1_list has the same length as subject_ids, different length as subject_id_list
        f1_list_dict = dict(zip(subject_ids, f1_list))
        subject_id_list_f1 = [f1_list_dict[subject_id] for subject_id in subject_id_list]
        color_list = []
        # if f1=1.0f and y_test=1,color=red; if f1=1.0f and y_test=0,color=yellow
        # if f1<1.0f and y_test=1,color=green; if f1<1.0f and y_test=0,color=blue
        for f1, y in zip(subject_id_list_f1, y_test):
            if f1 == 1.0 and y == 1:
                color_list.append('red')
            elif f1 == 1.0 and y == 0:
                color_list.append('yellow')
            elif f1 < 1.0 and y == 1:
                color_list.append('yellow') #should be green
            elif f1 < 1.0 and y == 0:
                color_list.append('blue')
            else:
                print("gray points occur")
                color_list.append('gray')
        color = np.array(color_list)

        del segments    
        gc.collect()

        c22_mv = Catch22(replace_nans=True)
        transformed_data_df = c22_mv.fit_transform(X_test)
        transformed_data = transformed_data_df.values

        # save transformed_data, y_test, color to a single npz file
        np.savez(os.path.join(root_path, 'visualize', f'transformed_data_6sub_f1_0.9.npz'), 
            transformed_data=transformed_data, 
            y_test=y_test, 
            color=color)


        visualize_nd(X=transformed_data, 
        y=y_test,
        color=color,
        reducer_type='tsne', # 'tsne', 'umap', 'pca'
        problem=problem, 
        n_components=2, # 2, 3
        n_points=10000,
        id = 'f1_0.9',
        root_path=root_path
        )
    else:
        print("No segments found for this subject!")   