import argparse
from src.models.train.knn import TrainKNNModel
from src.models.train.naive_bayes import TrainNaiveBayesModel

def main(args):
    print('args: ', args)
    if args.model == 'knn':
        model = TrainKNNModel(args.dataset, k=5)
        model.train()
        return print(f"KNN model successfully trained with {args.dataset}.")

    if args.model == 'naive_bayes':
        model = TrainNaiveBayesModel(args.dataset)
        model.train()
        return print(f"Naive Bayes model successfully trained with {args.dataset}.")
    
    print("No model specificied for training.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset type.')
    parser.add_argument('--model', type=str, help='Model type.')
    args = parser.parse_args()

    main(args)