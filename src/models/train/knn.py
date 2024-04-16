from sklearn.neighbors import KNeighborsClassifier
from src.features.build_features import BuildFeatures
from joblib import dump

class TrainKNNModel:
    
    def __init__(self, dataset, k=5):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.prepocessed_data = BuildFeatures(dataset)
        self.dataset = dataset
        
    def train(self):
        train, X_train, y_train = self.prepocessed_data.get_data('train', oversample=True)
        self.model.fit(X_train, y_train)
        
        # save the trained model
        model_name = f"models/{self.dataset}/{self.dataset}-knn-model.joblib"
        dump(self.model, model_name)
