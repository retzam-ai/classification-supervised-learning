from src.data.processed.cars_data import PrepocessedCarsData
from src.data.processed.injury_data import PrepocessedInjuryData
from src.data.processed.machine_data import PrepocessedMachineData
class BuildFeatures:
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def get_data(self, dataset_type, oversample=False):
        if self.dataset == 'cars':
            prepocessed_data = PrepocessedCarsData()
        
        if self.dataset == 'injury':
            prepocessed_data = PrepocessedInjuryData()
            
        if self.dataset == 'machines':
            prepocessed_data = PrepocessedMachineData()
            
        if dataset_type == 'test':
            test, X_test, y_test = prepocessed_data.get_data(dataset_type, oversample)
            return test, X_test, y_test
        
        train, X_train, y_train = prepocessed_data.get_data(dataset_type, oversample)
        
        return train, X_train, y_train

    