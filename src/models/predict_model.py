import argparse
from sklearn.metrics import classification_report
from src.features.build_features import BuildFeatures
from joblib import load

class PredictModel:
    
    def __init__(self, dataset, model):
        self.model = load(model)
        self.prepocessed_data = BuildFeatures(dataset)
        test, X_test, y_test = self.prepocessed_data.get_data('test', oversample=False)
        self.X_test = X_test
        self.y_test = y_test
        
    def predict(self, X=None):
        if X is not None:
            return self.model.predict(X)
        return self.model.predict(self.X_test)
    
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        return classification_report(self.y_test, y_pred)
    

def main(args):
    print('args: ', args)
    if args.model == 'knn':
        model = f"models/{args.dataset}/{args.dataset}-knn-model.joblib"
    elif args.model == 'naive_bayes':
        model = f"models/{args.dataset}/{args.dataset}-naive-bayes-model.joblib"
    elif args.model == 'logistic_regression':
        model = f"models/{args.dataset}/{args.dataset}-logistic-regression-model.joblib"
    elif args.model == 'svm':
        model = f"models/{args.dataset}/{args.dataset}-svm-model.joblib"
    elif args.model == 'random_forest':
        model = f"models/{args.dataset}/{args.dataset}-random-forest-model.joblib"
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model = PredictModel(args.dataset, model)
    prediction = model.predict()
    evaluation = model.evaluate()
    return print(prediction), print(evaluation)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset type.')
    parser.add_argument('--model', type=str, help='Model type.')
    args = parser.parse_args()

    main(args)