import argparse
from src.models.constants import DATASETS, MODELS
from src.models.train.knn import TrainKNNModel
from src.models.train.logistic_regression import TrainLogisticRegressionModel
from src.models.train.naive_bayes import TrainNaiveBayesModel
from src.models.train.random_forest import TrainRandomForestModel
from src.models.train.svm import TrainSVMModel

def main(args):
    model_name = args.model
    dataset_name = args.dataset
    
    print("dataset_name: ", dataset_name)
    print("datasets: ", DATASETS)
    
    if model_name not in MODELS:
        return print(f"Unsupported model: {model_name}")
    if  dataset_name not in DATASETS:
        return print(f"Unsupported dataset: {dataset_name}")
    
    if model_name == 'knn':
        model = TrainKNNModel(args.dataset)
        model.train()
        return print(f"KNN model successfully trained with {args.dataset}.")

    if model_name == 'naive_bayes':
        model = TrainNaiveBayesModel(args.dataset)
        model.train()
        return print(f"Naive Bayes model successfully trained with {args.dataset}.")
    
    if model_name == 'logistic_regression':
        model = TrainLogisticRegressionModel(args.dataset)
        model.train()
        return print(f"Logistic Regression model successfully trained with {args.dataset}.")
    
    if model_name == 'svm':
        model = TrainSVMModel(args.dataset)
        model.train()
        return print(f"SVM model successfully trained with {args.dataset}.")

    if model_name == 'random_forest':
        model = TrainRandomForestModel(args.dataset)
        model.train()
        return print(f"Random Forest model successfully trained with {args.dataset}.")
    
    print("No model specificied for training.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset type.')
    parser.add_argument('--model', type=str, help='Model type.')
    args = parser.parse_args()

    main(args)