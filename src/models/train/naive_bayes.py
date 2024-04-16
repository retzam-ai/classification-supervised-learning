from sklearn.naive_bayes import GaussianNB
from src.features.build_features import BuildFeatures
from joblib import dump

class TrainNaiveBayesModel:
    
    def __init__(self, dataset):
        self.model = GaussianNB()
        self.prepocessed_data = BuildFeatures(dataset)
        self.dataset = dataset
        
    def train(self):
        train, X_train, y_train = self.prepocessed_data.get_data('train', oversample=True)
        self.model.fit(X_train, y_train)
        
        # save the trained model
        model_name = f"models/{self.dataset}/{self.dataset}-naive-bayes-model.joblib"
        dump(self.model, model_name)
