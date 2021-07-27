#  #################################################################
#  Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks
#
#  This file contains the main code of DROO. It loads the training samples saved in ./data/data_#.mat, splits the samples into two parts (training and testing data constitutes 80% and 20%), trains the DNN with training and validation samples, and finally tests the DNN with test data.
#
#  Input: ./data/data_#.mat
#    Data samples are generated according to the CD method presented in [2]. There are 30,000 samples saved in each ./data/data_#.mat, where # is the user number. Each data sample includes
#  -----------------------------------------------------------------
#  |       wireless channel gain           |    input_h            |
#  -----------------------------------------------------------------
#  |       computing mode selection        |    output_mode        |
#  -----------------------------------------------------------------
#  |       energy broadcasting parameter   |    output_a           |
#  -----------------------------------------------------------------
#  |     transmit time of wireless device  |    output_tau         |
#  -----------------------------------------------------------------
#  |      weighted sum computation rate    |    output_obj         |
#  -----------------------------------------------------------------
#
#
#  References:
#  [1] 1. Liang Huang, Suzhi Bi, and Ying-Jun Angela Zhang, "Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks," in IEEE Transactions on Mobile Computing, early access, 2019, DOI:10.1109/TMC.2019.2928811.
#  [2] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” IEEE Trans. Wireless Commun., vol. 17, no. 6, pp. 4177-4190, Jun. 2018.
#
# version 1.0 -- July 2018. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################


import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy

# for tensorflow2
from memoryTF2 import MemoryDNN
# from optimization import bisection
from mytest_new import Algo1_NUM, racian_mec

import math

import time


def plot_rate( rate_his, rolling_intv = 50, ylabel='Normalized Computation Rate'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)


    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15,8))
#    rolling_intv = 20

#    plt.plot(np.arange(len(rate_array))+1, df.rolling(rolling_intv, min_periods=1).mean(), 'b')
#    plt.fill_between(np.arange(len(rate_array))+1, df.rolling(rolling_intv, min_periods=1).min()[0], df.rolling(rolling_intv, min_periods=1).max()[0], color = 'b', alpha = 0.2)
    plt.plot(np.arange(len(rate_array))+1, df.rolling(rolling_intv, min_periods=1).mean())
    plt.fill_between(np.arange(len(rate_array))+1, df.rolling(rolling_intv, min_periods=1).min()[0], df.rolling(rolling_intv, min_periods=1).max()[0], alpha = 0.2)
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.show()

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
        Adaptive K is implemented. K = max(K, K_his[-memory_size])
    '''

    N =10                     # number of users
    n = 10000                     # number of time frames
    K = N                   # initialize K = N
    decoder_mode = 'OPN'    # the quantization mode could be 'OP' (Order-preserving) or 'KNN' or 'OPN' (Order-Preserving with noise)
    Memory = 1024          # capacity of memory structure
    Delta = 32             # Update interval for adaptive K
    CHFACT = 10**10       # The factor for scaling channel value   
    energy_thresh = np.ones((N))*0.08; # energy comsumption threshold in J per time slot
    nu = 1000; # energy queue factor;
#    w = np.ones((N));      # weights for each user
    w = [1.5 if i%2==0 else 1 for i in range(N)]
    V = 20
#    arrival_lambda =30*np.ones((N))/N; # average data arrival in Mb, sum of arrival over all 'N' users is a constant
    lambda_param = 3
    arrival_lambda = lambda_param*np.ones((N)); # 3 Mbps per user

    print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))
    # Load data
    channel_test = sio.loadmat('./data/Qdata_%d' %N)['input_h']
    channel = np.zeros((n,N))
    dataA = np.zeros((n,N))
#    dataQ = sio.loadmat('./data/Qdata_%d' %N)['data_queue']
    dataA_test = sio.loadmat('./data/Qdata_%d' %N)['data_arrival']
#    energyY = sio.loadmat('./data/Qdata_%d' %N)['energy_queue']
#    energy = sio.loadmat('./data/Qdata_%d' %N)['energy_consumption']
#    obj_rate = sio.loadmat('./data/Qdata_%d' %N)['rate']
#    obj_val = sio.loadmat('./data/Qdata_%d' %N)['objective']
    state = sio.loadmat('./data/s')['s']
    # equal arrival rate for single user
    dataA_test = dataA_test*lambda_param*N/30
    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    channel_test = channel_test * CHFACT
    if N == 20 or N == 30:
        # my mistake in stored Qdata_20.mat and Qdata_30.mat
        channel_test = channel_test * CHFACT
    
    
    # generate channel
    dist_v = np.linspace(start = 120, stop = 255, num = N);
    Ad = 3
    fc = 915*10**6
    loss_exponent = 3; # path loss exponent
    light = 3*10**8
    h0 = np.ones((N))    
    for j in range(0,N):
        h0[j] = Ad*(light/4/math.pi/fc/dist_v[j])**(loss_exponent)

    # generate the train and test data sample index
    # data are splitted as 80:20
    # training data are randomly sampled with duplication if n > total data size

#    split_idx = int(.8* len(channel))
#    num_test = min(len(channel) - split_idx, n - int(.8* n)) # training data size
    num_test = 10000    
    test_idx = n - num_test


    mem = MemoryDNN(net = [N*3, 256, 128, N],
                    learning_rate = 0.01,
                    training_interval=20,
                    batch_size=128,
                    memory_size=Memory
                    )

    start_time=time.time()

    rate_his = []
    rate_his_ratio = []
    mode_his = []
    k_idx_his = []
    K_his = []
    Q = np.zeros((n,N)) # data queue in MbitsW
    Y = np.zeros((n,N)) # virtual energy queue in mJ
    Obj = np.zeros(n) # objective values after solving problem (26)
    energy = np.zeros((n,N)) # energy consumption
    rate = np.zeros((n,N)); # achieved computation rate
    
    # special test
#    Q = dataQ
#    Y = energyY
    

    for i in range(n):
        
        if i % (n//10) == 0:
            print("%0.1f"%(i/n))
            # plot_rate(Q, 100, 'Data Queue')
            # plot_rate(Y, 100, 'Energy Queue')
            # plot_rate(energy, 100, 'Energy Consumption')
        if i> 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(np.array(k_idx_his[-Delta:-1])%K) +1;
            else:
                max_k = k_idx_his[-1] +1;
            K = min(max_k +1, N)

        i_idx = i
        
#        if i < n - num_test:
#            # training
#            i_idx = i % split_idx
#        else:
#            # test
#            i_idx = i - n + num_test + split_idx
        
        if i_idx > test_idx:
            # test data
            h = channel_test[i_idx - test_idx,:]
            channel[i,:] = h;
            dataA[i,:] = dataA_test[i_idx - test_idx,:]
            # ON-OFF with constant arrival 
#            dataA[i,:] = state[:,i]*arrival_lambda*2
#            dataA[i,:] = state[:,i]*np.random.exponential(arrival_lambda*2)
#            dataA[i,:] = state[:,i]*np.random.binomial(1,0.5,N)*arrival_lambda*4
            # Pareto distribution with mean = 3            
#            dataA[i,:] = np.random.pareto(4/3,10)
        else:
            #real-time channel generation
            h_tmp = racian_mec(h0,0.3)
            h = h_tmp*CHFACT
            channel[i,:] = h;
            # real-time arrival generation
            dataA[i,:] = np.random.exponential(arrival_lambda)
            # ON-OFF with constant arrival 
#            dataA[i,:] = state[:,i]*arrival_lambda*2
#            dataA[i,:] = state[:,i]*np.random.exponential(arrival_lambda*2)
#            dataA[i,:] = state[:,i]*np.random.binomial(1,0.5,N)*arrival_lambda*4
            # Pareto distribution with mean = 3
#            dataA[i,:] = np.random.pareto(4/3,10)
        
        if i_idx > 0:
            # update queues
            Q[i_idx,:] = Q[i_idx-1,:] + dataA[i_idx-1,:] - rate[i_idx-1,:] # current data queue
            # assert Q is positive due to float error
            Q[i_idx,Q[i_idx,:]<0] =0
            Y[i_idx,:] = np.maximum(Y[i_idx-1,:] + (energy[i_idx-1,:]- energy_thresh)*nu,0); # current energy queue
            # assert Y is positive due to float error 
            Y[i_idx,Y[i_idx,:]<0] =0
        
        # scale Q and Y to 1
#        nn_input =np.concatenate( (h, Q[i_idx,:]/100,dataA[i_idx-1,:]/100,Y[i_idx,:]/100)) # arrival rate matters
        nn_input =np.vstack( (h, Q[i_idx,:]/10000,Y[i_idx,:]/10000)).transpose().flatten()
        
        # I prefer to the following one since arrival feature matters.
        # nn_input =np.concatenate( (h, Q[i_idx,:]/100-dataA[i_idx-1,:]/100,dataA[i_idx-1,:]/100,Y[i_idx,:]/100))

        # the action selection must be either 'OP' or 'KNN'
        m_list = mem.decode(nn_input, K, decoder_mode)

        r_list = [] # all results of candidate offloading modes
        v_list = [] # the objective values of candidate offloading modes
        for m in m_list:
            r_list.append(Algo1_NUM(m,h,w,Q[i_idx,:],Y[i_idx,:],V))
#            r_list.append(Algo1_NUM(m,h,dataA[i_idx-1,:],Q[i_idx,:],Y[i_idx,:]))
            v_list.append(r_list[-1][0]) 
#            v_list.append(sum(r_list[-1][1]))

        # record the index of largest reward
        k_idx_his.append(np.argmax(v_list))
        # memorize the largest reward
        rate_his.append(r_list[k_idx_his[-1]][1])
#        rate_his_ratio.append(sum(rate_his[-1]) / sum(obj_rate[i_idx]))
        
        # record K in case of adaptive K
        K_his.append(K)
        
        # encode the mode with largest reward
        mem.encode(nn_input, m_list[k_idx_his[-1]])
        mode_his.append(m_list[k_idx_his[-1]])
        
        # store max result
        Obj[i_idx],rate[i_idx,:],energy[i_idx,:]  = r_list[k_idx_his[-1]]
        
        


    total_time=time.time()-start_time
    mem.plot_cost()
    plot_rate(Q, 100, 'Data Queue')
    plot_rate(Q.sum(axis=1)/N, 100, 'Average Data Queue')
    plot_rate(Y, 100, 'Energy Queue')
    plot_rate(Y.sum(axis=1)/N, 100, 'Average Energy Queue')
    plot_rate(energy, 100, 'Energy Consumption')
#    plot_rate(rate_his_ratio, 100)
#    plot_rate(Obj[1:n]/obj_val[0][1:n], 100, 'Normalized Objective values')
    

#    print("Averaged normalized computation rate:", sum(rate_his_ratio[-num_test: -1])/num_test)
#    print("Averaged normalized objective value:", sum((Obj[-num_test:-1]/obj_val[0][-num_test:-1]))/num_test)
    print("Averaged energy consumption:", sum(energy[-num_test:-1])/num_test)
    print('Total time consumed:%s'%total_time)
    print('Average time per channel:%s'%(total_time/n))

    # save data into txt
    save_to_txt(k_idx_his, "k_idx_his.txt")
    save_to_txt(K_his, "K_his.txt")
    save_to_txt(mem.cost_his, "cost_his.txt")
#    save_to_txt(rate_his_ratio, "rate_his_ratio.txt")
    save_to_txt(mode_his, "mode_his.txt")
    
    sio.savemat('./result_%d.mat'%N, {'input_h': channel/CHFACT,'data_arrival':dataA,'data_queue':Q,'energy_queue':Y,'off_mode':mode_his,'rate':rate,'energy_consumption':energy,'data_rate':rate,'objective':Obj})
