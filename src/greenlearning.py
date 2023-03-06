import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from rational import *
import matplotlib.pyplot as plt
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
import scipy
import random
import multiprocessing

class GL(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(GL, self).__init__()
        self.fc1 = nn.Linear(4,50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,50)
        self.fc4 = nn.Linear(50,50)
        self.fc5 = nn.Linear(50,1)
        self.R1 = Rational()
        self.R2 = Rational()
        self.R3 = Rational()
        self.R4 = Rational()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.R1(x)
        x = self.fc2(x)
        x = self.R2(x)
        x = self.fc3(x)
        x = self.R3(x)
        x = self.fc4(x)
        x = self.R4(x)
        x = self.fc5(x)
        return x

def GL_main(save_index, ntrain):
    
    print("ntrain = %d, index = %d" %(ntrain, save_index))
    ################################################################
    # configs
    ################################################################
    TRAIN_PATH = '../data/train.mat'
    TEST_PATH = '../data/test.mat'
    
    ntest = 200
    
    if ntrain >= 20:
        batch_size = 20
    else:
        batch_size = ntrain
    learning_rate = 0.001
    
    epochs = 2500
    step_size = 100
    gamma = 0.9
    
    modes = 12
    width = 32
    
    s = 29
    r = 15
    
    ################################################################
    # load data and data normalization
    ################################################################
    reader = MatReader(TRAIN_PATH)
    i_train = random.sample(range(1000), ntrain)
    x_train = reader.read_field('force')[i_train,::r,::r][:,:s,:s]
    y_train = reader.read_field('sol')[i_train,::r,::r][:,:s,:s]
    
    reader.load_file(TEST_PATH)
    x_test = reader.read_field('force')[:ntest,::r,::r][:,:s,:s]
    y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]
    
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    
    grids = []
    grid_all = np.linspace(0, 1, 421).reshape(421, 1).astype(np.float64)
    grids.append(grid_all[::r,:])
    grids.append(grid_all[::r,:])
    grids.append(grid_all[::r,:])
    grids.append(grid_all[::r,:])
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = torch.tensor(grid, dtype=torch.float)
    
    ################################################################
    # training and evaluation
    ################################################################
    model = GL(modes, modes, width).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    start_time = default_timer()
    myloss = LpLoss(size_average=False)
    
    # Create tensors
    x_train = x_train.reshape(ntrain, s**2)
    x_test = x_test.reshape(ntest, s**2)
    grid = grid.cuda()
    
    y_normalizer.cuda()
    for ep in range(epochs):
        x = x_train
        y = y_train
        x, y = x.cuda(), y.cuda()
        curr_size = x.shape[0]
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_mse = 0
        # Get G(x,y)
        optimizer.zero_grad()
        out = model(grid)
        out = out.reshape(s**2,s**2)
        # Multiply by f(y)
        out = torch.matmul(x, out) * (1/s**2)
        out = out.reshape(ntrain, s, s)
        mse = F.mse_loss(out.reshape(ntrain, -1), y.reshape(ntrain, -1), reduction='mean')
        mse.backward()
            
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        loss = myloss(out.reshape(ntrain,-1), y.reshape(ntrain,-1))
        # loss.backward()
    
        optimizer.step()
        train_mse += mse.item()
        train_l2 += loss.item()
    
        scheduler.step()
    
        model.eval()
        with torch.no_grad():
            x = x_test
            y = y_test
            x, y = x.cuda(), y.cuda()
            
            curr_size = x.shape[0]
            out = model(grid)
            out = out.reshape(s**2,s**2)
            # Multiply by f(y)
            out = torch.matmul(x, out) * (1/s**2)
            out = out.reshape(curr_size, s, s)
            out = y_normalizer.decode(out)
            test_l2 = myloss(out.reshape(curr_size,-1), y.reshape(curr_size,-1)).item()
    
        #train_mse /= ntrain
        train_l2/= ntrain
        test_l2 /= ntest
    
        t2 = default_timer()
        # print(ep, t2-t1, train_l2, test_l2)
        if ep %10 == 0:
            print("Epoch: %d, time: %.3f, Train Loss: %.3e, Train l2: %.4f, Test l2: %.4f" 
                  % ( ep, t2-t1, train_mse, train_l2, test_l2) )

    elapsed = default_timer() - start_time
    print("\n=============================")
    print("Training done...")
    print('Training time: %.3f'%(elapsed))
    print("=============================\n")
    
    ################################################################
    # testing
    ################################################################
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    
    index = 0
    t1 = default_timer()
    test_l2 = 0
    
    with torch.no_grad():
        out = model(grid)
        out = out.reshape(s**2,s**2)
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            # Multiply by f(y)
            out_f = torch.matmul(x, out) * (1/s**2)
            out_f = out_f.reshape(1, s, s)
            out_f = y_normalizer.decode(out_f)
    
            test_l2 += np.linalg.norm(out_f.reshape(1, -1).cpu().numpy() 
                                      - y.reshape(1, -1).cpu().numpy()) / np.linalg.norm(y.reshape(1, -1).cpu().numpy())
            index = index + 1
    t2 = default_timer()
    testing_time = t2-t1
     
    test_l2 = test_l2/index
    print("\n=============================")
    print('Testing error: %.3e'%(test_l2))
    print("=============================\n")
    
    with open('result_gl.csv','a+') as file:
        file.write("%d,%d,%e"%(save_index, ntrain, test_l2))
        file.write('\n')

if __name__ == "__main__":
    
    multiprocessing.set_start_method('spawn')
    N = np.floor(10**(np.linspace(np.log10(2),np.log10(10**3),20)))
    for ntrain in N:
        for run_index in range(10):
            process_train = multiprocessing.Process(target=GL_main, args=(run_index, int(ntrain),))
            process_train.start()
            process_train.join()
