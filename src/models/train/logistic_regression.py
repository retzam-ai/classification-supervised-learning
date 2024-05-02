from src.features.build_features import BuildFeatures
from sklearn.linear_model import LogisticRegression
from joblib import dump

class TrainLogisticRegressionModel:
    
    def __init__(self, dataset):
        self.model = LogisticRegression()
        self.prepocessed_data = BuildFeatures(dataset)
        self.dataset = dataset
        
    def train(self):
        train, X_train, y_train = self.prepocessed_data.get_data('train', oversample=True)
        self.model.fit(X_train, y_train)
        
        # save the trained model
        model_name = f"models/{self.dataset}/{self.dataset}-logistic-regression-model.joblib"
        dump(self.model, model_name)
