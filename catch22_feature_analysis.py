import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sktime

from sktime.transformations.panel.catch22 import Catch22
from sktime.classification.feature_based import Catch22Classifier
import joblib

if __name__ == "__main__":
    # load saved Catch22Classifier model
    model_path = "D:/seizure/models/0529_catch22_seed41.pkl"
    model = joblib.load(model_path)

    importances = model.estimator.feature_importances_
    catch22_feature_names = model.estimator.feature_names_in_
    feature_importance_df = pd.DataFrame({
    'Feature': catch22_feature_names,
    'Importance': importances
})


    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print(feature_importance_df)
    