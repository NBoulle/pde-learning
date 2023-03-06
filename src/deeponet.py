import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, io
from sklearn.preprocessing import StandardScaler
import deepxde as dde
from deepxde.backend import tf
import random
import multiprocessing

def get_data(filename, ndata, train=True):
    r = 15
    s = 29

    # Data is of the shape (number of samples, grid size = 421x421)
    data = io.loadmat(filename)
    if train:
        i_train = random.sample(range(1000), ndata)
        x_branch = data["force"][i_train, ::r, ::r].astype(np.float32) * 0.1 - 0.75
        y = data["sol"][i_train, ::r, ::r].astype(np.float32) * 100
    else:
        x_branch = data["force"][:ndata, ::r, ::r].astype(np.float32) * 0.1 - 0.75
        y = data["sol"][:ndata, ::r, ::r].astype(np.float32) * 100
    grids = []
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    x_branch = x_branch.reshape(ndata, s * s)
    x = (x_branch, grid)
    y = y.reshape(ndata, s * s)
    return x, y

def dirichlet(inputs, output):
    x_trunk = inputs[1]
    x, y = x_trunk[:, 0], x_trunk[:, 1]
    return 20 * x * (1 - x) * y * (1 - y) * (output + 1)

def DON_main(run_index, ntrain):
    
    print("ntrain = %d, index = %d" %(ntrain, run_index))
    x_train, y_train = get_data('../data/train.mat', ntrain, train=True)
    x_test, y_test = get_data('../data/test.mat', 200, train=False)
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    m = 29 ** 2
    activation = "relu"
    branch = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(m,)),
            tf.keras.layers.Reshape((29, 29, 1)),
            tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Conv2D(128, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=activation),
            tf.keras.layers.Dense(128),
        ]
    )
    
    branch.summary()
    net = dde.maps.DeepONetCartesianProd(
        [m, branch], [2, 128, 128, 128, 128], activation, "Glorot normal"
    )
    
    scaler = StandardScaler().fit(y_train)
    std = np.sqrt(scaler.var_.astype(np.float32))

    def output_transform(inputs, outputs):
        return outputs * std + scaler.mean_.astype(np.float32)

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile(
        "adam",
        decay=("inverse time", 1, 1e-4),
        lr = 3e-4,
        metrics=["mean l2 relative error"],
    )
    
    losshistory, train_state = model.train(epochs=10**5, batch_size=None)
    test_l2  = train_state.best_metrics[0]
    
    with open('result_DNO.csv','a+') as file:
        file.write("%d,%d,%e"%(run_index, ntrain, test_l2))
        file.write('\n')

if __name__ == "__main__":

    N = np.floor(10**(np.linspace(np.log10(2),np.log10(10**3),20)))
    for run_index in range(10):
        for ntrain in N:        
            process_train = multiprocessing.Process(target=DON_main, args=(run_index, int(ntrain),))
            process_train.start()
            process_train.join()
