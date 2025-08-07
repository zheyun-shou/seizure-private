import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sktime
# import cuml
from cuml.ensemble import RandomForestClassifier
from sktime.base._base import _clone_estimator
from sktime.classification._delegate import _DelegatedClassifier
from sktime.pipeline import make_pipeline
from sktime.transformations.panel.catch22 import Catch22
from sktime.classification.feature_based import Catch22Classifier
import joblib

class Catch22CumlClassifier(_DelegatedClassifier):
    _tags = {
        # packaging info
        # --------------
        "authors": ["MatthewMiddlehurst", "RavenRudi", "fkiraly"],
        "maintainers": ["RavenRudi"],
        "python_dependencies": "numba",
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:multithreading": True,
        "capability:predict_proba": True,
        "classifier_type": "feature",
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(
        self,
        outlier_norm=False,
        replace_nans=True,
        estimator=None,
        batch_size=128,
        n_jobs=1,
        random_state=None,
    ):
        self.outlier_norm = outlier_norm
        self.replace_nans = replace_nans
        self.estimator = estimator
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

        self.transformer_ = Catch22(
            outlier_norm=self.outlier_norm, 
            replace_nans=self.replace_nans
        )

        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=200)

        self.estimator_ = _clone_estimator(estimator, random_state)

        m = getattr(estimator, "n_jobs", None)
        if m is not None:
            self.estimator_.n_jobs = self._threads_to_use

    def _batch_transform(self, X):
        """Perform batch transformation to avoid memory issues."""
        if len(X) <= self.batch_size:
            return self.transformer_.transform(X)
        
        batch_features = []
        for i in tqdm.tqdm(range(0, len(X), self.batch_size), desc=f"Transform with batch={self.batch_size}"):
            batch = X[i:i+self.batch_size]
            batch_features.append(self.transformer_.transform(batch))

        # from concurrent.futures import ProcessPoolExecutor, as_completed
        # with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
        #     # use tqdm to show progress
        #     tasks = [executor.submit(self.transformer_.transform, X[i:i+self.batch_size]) for i in range(0, len(X), self.batch_size)]
        #     pbar = tqdm.tqdm(as_completed(tasks), total=len(future_to_recording), desc="", ncols=100)
        #     for res in pbar:
        #         batch_features.append(res.result())
        
        return np.concatenate(batch_features, axis=0)

    def fit(self, X, y):
        """Override fit to handle batch processing."""
        # Fit the transformer first
        self.transformer_.fit(X, y)
        
        # Transform data in batches
        X_transformed = self._batch_transform(X)
        
        # Fit the estimator
        self.estimator_.fit(X_transformed, y)
        
        return self

    def predict(self, X):
        """Override predict to handle batch processing."""
        X_transformed = self._batch_transform(X)
        return self.estimator_.predict(X_transformed)

    def predict_proba(self, X):
        """Override predict_proba to handle batch processing."""
        X_transformed = self._batch_transform(X)
        return self.estimator_.predict_proba(X_transformed)

    def fit_transform(self, X, y):
        """Implement fit_transform with batch processing."""
        self.fit(X, y)
        return self._batch_transform(X)

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
    