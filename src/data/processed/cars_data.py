import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

class PrepocessedCarsData:
    
    def __init__(self):
        # Import the car data file and use the first column as the title
        df = pd.read_csv('data/processed/CarsData.csv', header=0)
        
        # Convert each column with nominal data to numbers from 0, 1, 2...
        df["model"], _ = pd.factorize(df["model"])
        df["fuelType"], _ = pd.factorize(df["fuelType"])
        df["transmission"], _ = pd.factorize(df["transmission"])
        
        
        # Create a map using the unique values array above.
        mapping = {
            'hyundi': 0, 
            'volkswagen': 1, 
            'BMW': 2, 
            'skoda': 3, 
            'ford': 4, 
            'toyota': 5, 
            'merc': 6,
            'vauxhall': 7, 
            'Audi': 8,
        }

        # Replace the values
        df['Manufacturer'] = df['Manufacturer'].replace(mapping).infer_objects(copy=False)

        """
            Split dataset into training and test.

            df.sample: randomizes the array so the training data get all feature vectors.
            np.split: splits an array in multiple sub arrays.
            0.8*len(df): this makes the split to be 0-80% for training and, 20% for test.
        """
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

    