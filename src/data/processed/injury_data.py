import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

class PrepocessedInjuryData:
    
    def __init__(self):
        df = pd.read_csv('data/processed/injury_data.csv', header=0)
        
        train, test = np.split(df.sample(frac=1), [int(0.8*len(df))])
        
        self.train = train
        self.test = test
        
    def scale_dataset(self, dataframe, oversample=False):
        # This selects all columns in the DataFrame except the last one as the features.
        X = dataframe[dataframe.columns[:-1]].values

        # This selects the last column in the DataFrame as the target.
        y = dataframe[dataframe.columns[-1]].values

        # This removes the mean and scaling to unit variance
        # Known as standardization. Basically removes outliers.
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        """
            Make both x and y sets equal sets as appropriate.

            RandomOverSampler is important in cases where there is alot more features vector of a
            specific output. 
            
            Example if you have a dataset with 100 rows with output as "Yes" and 20
            rows with "No". 
            You can see that our datasets would be biased towards the output with "Yes".
            To solve this, RandomOverSampler strategically duplicates rows with "No" so the dataset ends up
            having 100 rows with "Yes" and 100 with "No" outputs.

            This is called over-sampling.
        """
        if oversample:
            ros = RandomOverSampler()
            X, y = ros.fit_resample(X, y)

        # Stack horizontally
        # Reshape y and concatenate it with X
        # This simply means attaching each feature vector with the appropriate output.
        data = np.hstack((X, np.reshape(y, (-1, 1))))

        return data, X, y
    
    def get_data(self, dataset_type, oversample=False):
        if(dataset_type == 'train'):
            return self.scale_dataset(self.train, oversample)
        if(dataset_type == 'test'):
            return self.scale_dataset(self.test, oversample)
        return self.scale_dataset(self.train)

    