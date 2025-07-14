from sktime.transformations.panel.catch22 import Catch22
import pandas as pd
import os
import numpy as np
from dataloader import get_ids_from_filename, get_path_from_ids, extract_event_info, extract_epochs, get_data_from_epochs
import gc
from feature_visualize import visualize_nd

if __name__ == "__main__":
    root_path = 'D:/seizure'

    npz_file = np.load(os.path.join(root_path, 'visualize', f'transformed_data_6sub_f1_0.9.npz'))
    transformed_data = npz_file['transformed_data']
    y_test = npz_file['y_test']
    color = npz_file['color']
    problem = 'EEG'

    visualize_nd(X=transformed_data, 
    y=y_test,
    color=color,
    reducer_type='tsne', # 'tsne', 'umap', 'pca'
    problem=problem, 
    n_components=3, # 2, 3
    n_points=10000,
    id ='f0.9',
    root_path=root_path
    )