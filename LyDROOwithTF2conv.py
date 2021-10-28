#  #################################################################
#
#  This file contains the main code of LyDROO.
#
#  References:
#  [1] Suzhi Bi, Liang Huang, Hui Wang, and Ying-Jun Angela Zhang, "Lyapunov-guided Deep Reinforcement Learning for Stable Online Computation Offloading in Mobile-Edge Computing Networks," IEEE Transactions on Wireless Communications, 2021, doi:10.1109/TWC.2021.3085319.
#  [2] Liang Huang, Suzhi Bi, and Ying-Jun Angela Zhang, "Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks," in IEEE Transactions on Mobile Computing, vol. 19, no. 11, pp. 2581-2593, November 2020.
#  [3] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” IEEE Trans. Wireless Commun., vol. 17, no. 6, pp. 4177-4190, Jun. 2018.
#
# version 1.0 -- July 2020. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################


import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy

# for tensorflow2
from memoryTF2conv import MemoryDNN
# from optimization import bisection
from ResourceAllocation import Algo1_NUM

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

    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color = 'b', alpha = 0.2)
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.show()

# generate racian fading channel with power h and Line of sight ratio factor
# replace it with your own channel generations when necessary
def racian_mec(h,factor):
    n = len(h)
    beta = np.sqrt(h*factor) # LOS channel amplitude
    sigma = np.sqrt(h*(1-factor)/2) # scattering sdv
    x = np.multiply(sigma*np.ones((n)),np.random.randn(n)) + beta*np.ones((n))
    y = np.multiply(sigma*np.ones((n)),np.random.randn(n))
    g = np.power(x,2) +  np.power(y,2)
    return g


if __name__ == "__main__":
    '''
        LyDROO algorithm composed of four steps:
            1) 'Actor module'
            2) 'Critic module'
            3) 'Policy update module'
            4) ‘Queueing module’ of
    '''

    N =10                     # number of users
    n = 10000                     # number of time frames
    K = N                   # initialize K = N
    decoder_mode = 'OPN'    # the quantization mode could be 'OP' (Order-preserving) or 'KNN' or 'OPN' (Order-Preserving with noise)
    Memory = 1024          # capacity of memory structure
    Delta = 32             # Update interval for adaptive K
    CHFACT = 10**10       # The factor for scaling channel value
    energy_thresh = np.ones((N))*0.08 # energy comsumption threshold in J per time slot
    nu = 1000 # energy queue factor;
#    w = np.ones((N))      # weights for each user
    w = [1.5 if i%2==0 else 1 for i in range(N)]
    V = 20
#    arrival_lambda =30*np.ones((N))/N # average data arrival in Mb, sum of arrival over all 'N' users is a constant
    lambda_param = 3
    arrival_lambda = lambda_param*np.ones((N)) # 3 Mbps per user

    print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))

    # initialize data
    channel = np.zeros((n,N)) # chanel gains
    dataA = np.zeros((n,N)) # arrival data size

    # generate channel
    dist_v = np.linspace(start = 120, stop = 255, num = N)
    Ad = 3
    fc = 915*10**6
    loss_exponent = 3 # path loss exponent
    light = 3*10**8
    h0 = np.ones((N))
    for j in range(0,N):
        h0[j] = Ad*(light/4/math.pi/fc/dist_v[j])**(loss_exponent)


    mem = MemoryDNN(net = [N*3, 256, 128, N],
                    learning_rate = 0.01,
                    training_interval=20,
                    batch_size=128,
                    memory_size=Memory
                    )

    start_time=time.time()
    mode_his = [] # store the offloading mode
    k_idx_his = [] # store the index of optimal offloading actor
    Q = np.zeros((n,N)) # data queue in MbitsW
    Y = np.zeros((n,N)) # virtual energy queue in mJ
    Obj = np.zeros(n) # objective values after solving problem (26)
    energy = np.zeros((n,N)) # energy consumption
    rate = np.zeros((n,N)) # achieved computation rate

    for i in range(n):

        if i % (n//10) == 0:
            print("%0.1f"%(i/n))
        if i> 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(np.array(k_idx_his[-Delta:-1])%K) +1
            else:
                max_k = k_idx_his[-1] +1
            K = min(max_k +1, N)

        i_idx = i



        #real-time channel generation
        h_tmp = racian_mec(h0,0.3)
        # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
        h = h_tmp*CHFACT
        channel[i,:] = h
        # real-time arrival generation
        dataA[i,:] = np.random.exponential(arrival_lambda)


        # 4) ‘Queueing module’ of LyDROO
        if i_idx > 0:
            # update queues
            Q[i_idx,:] = Q[i_idx-1,:] + dataA[i_idx-1,:] - rate[i_idx-1,:] # current data queue
            # assert Q is positive due to float error
            Q[i_idx,Q[i_idx,:]<0] =0
            Y[i_idx,:] = np.maximum(Y[i_idx-1,:] + (energy[i_idx-1,:]- energy_thresh)*nu,0) # current energy queue
            # assert Y is positive due to float error
            Y[i_idx,Y[i_idx,:]<0] =0

        # scale Q and Y to 1
        nn_input =np.vstack( (h, Q[i_idx,:]/10000,Y[i_idx,:]/10000)).transpose().flatten()


        # 1) 'Actor module' of LyDROO
        # generate a batch of actions
        m_list = mem.decode(nn_input, K, decoder_mode)

        r_list = [] # all results of candidate offloading modes
        v_list = [] # the objective values of candidate offloading modes
        for m in m_list:
            # 2) 'Critic module' of LyDROO
            # allocate resource for all generated offloading modes saved in m_list
            r_list.append(Algo1_NUM(m,h,w,Q[i_idx,:],Y[i_idx,:],V))
            v_list.append(r_list[-1][0])

        # record the index of largest reward
        k_idx_his.append(np.argmax(v_list))

        # 3) 'Policy update module' of LyDROO
        # encode the mode with largest reward
        mem.encode(nn_input, m_list[k_idx_his[-1]])
        mode_his.append(m_list[k_idx_his[-1]])

        # store max result
        Obj[i_idx],rate[i_idx,:],energy[i_idx,:]  = r_list[k_idx_his[-1]]


    total_time=time.time()-start_time
    mem.plot_cost()

    plot_rate(Q.sum(axis=1)/N, 100, 'Average Data Queue')
    plot_rate(energy.sum(axis=1)/N, 100, 'Average Energy Consumption')

    print('Average time per channel:%s'%(total_time/n))

    # save all data
    sio.savemat('./result_%d.mat'%N, {'input_h': channel/CHFACT,'data_arrival':dataA,'data_queue':Q,'energy_queue':Y,'off_mode':mode_his,'rate':rate,'energy_consumption':energy,'data_rate':rate,'objective':Obj})
