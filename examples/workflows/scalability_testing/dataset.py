import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, LabelEncoder

class HousingDataset(object):

    def get(preprocess=True):
        X, y = fetch_california_housing(return_X_y=True)
        
        # Preprocess data
        if preprocess:
            num_features = 8
            num_idx = []
            cat_dims = []
            cat_idx = []
            for i in range(num_features):
                if cat_idx and i in cat_idx:
                    le = LabelEncoder()
                    X[:, i] = le.fit_transform(X[:, i])            
                    cat_dims.append(len(le.classes_))
                else:
                    num_idx.append(i)    
                
            scaler = StandardScaler()
            X[:, num_idx] = scaler.fit_transform(X[:, num_idx])
        xy = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        return xy
