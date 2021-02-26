# Chang Wei Tan, Christoph Bergmeir, Francois Petitjean, Geoff Webb
# https://github.com/xiaolongwu0713/TS-Extrinsic-Regression
# @article{Tan2020Time,
#   title={Time Series Regression},
#   author={Tan, Chang Wei and Bergmeir, Christoph and Petitjean, Francois and Webb, Geoffrey I},
#   journal={arXiv preprint arXiv:2006.12672},
#   year={2020}
# }

## tested on one 15s period, very bad
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import getopt
import sys
import numpy as np
from utils.data_loader import load_from_tsfile_to_dataframe
from utils.regressor_tools import process_data, fit_regressor, calculate_regression_metrics
from utils.tools import create_directory
from utils.transformer_tools import fit_transformer
from grasp.utils import read_fbanddata, plot_on_test, plot1, plot_on_train,plotloss,read_rawdata,preprocess,itermove
from torch.utils.data import Dataset, DataLoader

module = "RegressionExperiment"
data_path = "data/"
problem = "PPGDalia"  # see data_loader.regression_datasets
regressor_name = "rocket"  # see regressor_tools.all_models
transformer_name = "none"  # see transformer_tools.transformers
itr = 1
norm = "none"  # none, standard, minmax

# transformer parameters
flatten = False  # if flatten, do not transform per dimension
n_components = 10  # number of principal components
n_basis = 10  # number of basis functions
bspline_order = 4  # bspline order

# parse arguments
try:
    opts, args = getopt.getopt(sys.argv[1:], "hd:p:r:i:n:m:",
                               ["data_path=", "problem=", "regressor=", "iter=", "norm="])
except getopt.GetoptError:
    print("demo.py -d <data_path> -p <problem> -r <regressor> -i <iteration> -n <normalisation>")
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print("demo.py -d <data_path> -p <problem> -s <regressor> -i <iteration> -n <normalisation>")
        sys.exit()
    elif opt in ("-d", "--data"):
        data_path = arg
    elif opt in ("-p", "--problem"):
        problem = arg
    elif opt in ("-r", "--regressor"):
        regressor_name = arg
    elif opt in ("-i", "--iter"):
        itr = arg
    elif opt in ("-n", "--norm"):
        norm = arg

# start the program
if __name__ == '__main__':
    # preprocess()
    trainset = itermove(root_dir='/Users/long/BCI/matlab_scripts/force/data/SEEG_Data/dataloader/train')
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
    #testset = itermove(root_dir='/Users/long/BCI/matlab_scripts/force/data/SEEG_Data/dataloader/test')
    #testloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)

    data=trainset[0]
    x_train = data['data']
    y_train = data['force']
    data = trainset[1]
    x_test = data['data']
    y_test = data['force']
    #permutate axes:
    aa = x_train.swapaxes(0, 1)
    x_train=aa.swapaxes(1,2)
    aa = x_test.swapaxes(0, 1)
    x_test = aa.swapaxes(1, 2)

    # create output directory
    output_directory = "/Users/long/BCI/python_scripts/examples/TSER/output/regression/"
    if norm != "none":
        output_directory = "/Users/long/BCI/python_scripts/examples/TSER/output/regression_{}/".format(norm)
    output_directory = output_directory + regressor_name + '/' + problem + '/itr_' + str(itr) + '/'
    #create_directory(output_directory)

    print("=======================================================================")
    print("[{}] Starting Holdout Experiments".format(module))
    print("=======================================================================")
    print("[{}] Data path: {}".format(module, data_path))
    print("[{}] Output Dir: {}".format(module, output_directory))
    print("[{}] Iteration: {}".format(module, itr))
    print("[{}] Problem: {}".format(module, problem))
    print("[{}] Regressor: {}".format(module, regressor_name))
    print("[{}] Transformer: {}".format(module, transformer_name))
    print("[{}] Normalisation: {}".format(module, norm))

    # transform the data if needed
    if transformer_name != "none":
        if transformer_name == "pca":
            kwargs = {"n_components": n_components}
        elif transformer_name == "fpca":
            kwargs = {"n_components": n_components}
        elif transformer_name == "fpca_bspline":
            kwargs = {"n_components": n_components,
                      "n_basis": n_basis,
                      "order": bspline_order,
                      "smooth": "bspline"}
        else:
            kwargs = {}
        x_train, transformer = fit_transformer(transformer_name, x_train, flatten=flatten, **kwargs)
        x_test = transformer.transform(x_test)

    print("[{}] X_train: {}".format(module, x_train.shape))
    print("[{}] X_test: {}".format(module, x_test.shape))

    # fit the regressor
    regressor = fit_regressor(output_directory, regressor_name, x_train, y_train, x_test, y_test, itr=itr)

    # start testing
    y_pred = regressor.predict(x_test)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.savefig('y_testandy_pred')
    plt.close(fig)

    df_metrics = calculate_regression_metrics(y_test, y_pred)

    print(df_metrics)

    # save the outputs
    df_metrics.to_csv(output_directory + 'regression_experiment.csv', index=False)
