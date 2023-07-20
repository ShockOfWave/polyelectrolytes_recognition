from src.data.make_dataset import create_barcodes, calc_bottleneck_dist, create_dataset
from src.train.tune_model_optuna import tune_model_optuna

def main():
    create_barcodes()
    calc_bottleneck_dist()
    create_dataset()
    tune_model_optuna()
    
if __name__ == '__main__':
    main()