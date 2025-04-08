from pathlib import Path
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
import torch
from dataloader import read_dataset
from sktime.classification.kernel_based import RocketClassifier

from tsai.imports import default_device
from fastai.metrics import accuracy
from fastai.callback.tracker import ReduceLROnPlateau
from tsai.data.all import get_ts_dls, TSStandardize, TSClassification
from tsai.learner import ts_learner, Learner

from tsai.models.utils import build_ts_model, accuracy
from tsai.models.MINIROCKET_Pytorch import MiniRocket, MiniRocketFeatures, MiniRocketHead, get_minirocket_features
from tsai.inference import load_learner
from analysis import Analyzer



if __name__ == "__main__":
    # dname = os.path.dirname(os.path.abspath(__file__))
    # bids_root = dname + '\BIDS_Siena' 
    bids_root = 'E:\BIDS_Siena' # Replace with your actual path
    #bids_root = 'E:\BIDS_CHB-MIT'

    event_infos, segments = read_dataset(bids_root, max_workers=2)

    #only in desired channels?
    X = np.concatenate([s['epoch'] for s in segments]).astype(np.float32)
    y = np.concatenate([s['label'] for s in segments]).astype(int)
    X = X[:, np.newaxis, :]
    print(X.shape, y.shape)

    del segments

    train_size = 0.8
    split_train, split_test = train_test_split(range(len(y)), train_size=train_size, random_state=42, stratify=y)
    splits = (split_train, split_test)

    print("cuda available: ", torch.cuda.is_available())

    # references:
    # https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/10_Time_Series_Classification_and_Regression_with_MiniRocket.ipynb
    # https://timeseriesai.github.io/tsai/tslearner.html
    # https://timeseriesai.github.io/tsai/models.minirocket_pytorch.html

    mrf = MiniRocketFeatures(X.shape[1], X.shape[2]).to(default_device()) # mrf: MiniRocketFeatures
    X_train = X[splits[0]] # X[split_train]
    mrf.fit(X_train)
    X_feat = get_minirocket_features(X, mrf, chunksize=1024, use_cuda=torch.cuda.is_available(), to_np=True)
    tfms = [None, TSClassification()] # ?
    batch_tfms = [TSStandardize(by_sample=True)] # batch transforms are applied after the dataset is created
    # batch_tfms = [TSStandardize(by_sample=True), TSMagScale(), TSWindowWarp()] # Data argumentation examples

    
    start_time = time.time()

    # offline feature calculation: calculate all features at once
    dls = get_ts_dls(X_feat, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=256) # dataloaders

    ## seems to be old fashion of creating a model...
    model = build_ts_model(MiniRocketHead, dls=dls) # build a time-series model      
    learn = Learner(dls, model, metrics=accuracy) # Learner; add callback cbs=[ShowGraph()] to see the plot

    #learn = ts_learner(dls, MiniRocketHead, metrics=accuracy) # idn what is kernel_size

    # # Online feature calculation:
    # # MiniRocket can also be used online, re-calculating the features each minibatch. In this scenario, you do not calculate fixed features one time. The online mode is a bit slower than the offline scanario, but offers more flexibility. Here are some potential uses:

    # You can experiment with different scaling techniques (no standardization, standardize by sample, normalize, etc).
    # You can use data augmentation is applied to the original time series.
    # Another use of online calculation is to experiment with training the kernels and biases. To do this requires modifications to the MRF code.
    # get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
    # learn = ts_learner(dls, MiniRocket, metrics=accuracy) # idn what is kernel_size

    # the learner is not fine-tuned;
    # learn.lr_find() # find the best learning rate
    # there should be other hyperparameters to tune: https://timeseriesai.github.io/tsai/optuna.html
    learn.fit(n_epoch=1000, lr=1e-3, cbs=ReduceLROnPlateau(factor=0.5, min_lr=1e-8, patience=10))

    end_time = time.time()
    print(f"Model training took: {end_time - start_time:.3f} seconds")

    # Save the feature encoder and backbone model
    PATH = Path("./models/MRF.pt")
    PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mrf.state_dict(), PATH)

    PATH = Path('./models/MRL.pkl')
    PATH.parent.mkdir(parents=True, exist_ok=True)
    learn.export(PATH)

    # Load the feature encoder and backbone model
    mrf = MiniRocketFeatures(X.shape[1], X.shape[2]).to(default_device())
    PATH = Path("./models/MRF.pt")
    mrf.load_state_dict(torch.load(PATH))

    PATH = Path('./models/MRL.pkl')
    learn = load_learner(PATH, cpu=False) # change cpu to True if you don't have a GPU
    
    # Inference (prediction)
    start_time = time.time()

    mrf = MiniRocketFeatures(X.shape[1], X.shape[2]).to(default_device())
    PATH = Path("./models/MRF.pt")
    mrf.load_state_dict(torch.load(PATH))

    new_feat = get_minirocket_features(X[splits[1]], mrf, chunksize=1024, to_np=True)
    probas, _, preds = learn.get_X_preds(new_feat)
    # turn list preds to int
    preds = np.array([int(p) for p in preds])

    print(f"Predicted probabilities: {probas}, Predicted labels: {preds}")

    y_test = y[splits[1]]
    analyzer = Analyzer(print_conf_mat=True)
    analyzer.analyze_classification(preds, y_test, ['normal', 'seizure'])
    accuracy = np.mean(preds == y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    end_time = time.time()
    print(f"Model prediction took: {end_time - start_time:.2f} seconds")




