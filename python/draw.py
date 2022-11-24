import pickle
from recorder import Recoder
import matplotlib.pyplot as plt
import torch
import time
def show_sigle_window(v_mat,step='Step1995',task='Task3',save=False):


    rmean = v_mat.mean(axis=1)
    rmean = torch.unsqueeze(rmean, dim=1)
    v_mat = v_mat - rmean

    rmax = v_mat.max(axis=1).values
    rmax = torch.unsqueeze(rmax, dim=1)
    v_mat = v_mat / rmax

    plt.cla()
    plt.imshow(v_mat)
    if save:
        plt.savefig(step + '_' + task + '_.png')

    plt.show()
    plt.pause(0.01)

    #plt.show()

def dynamic_window(v_mat,window_len=100,step=2):
    n_frame=int((v_mat.shape[1]-window_len)/step)
    plt.figure(1)
    plt.ion()

    for i in range(n_frame):
        frame=v_mat[:,i*step:i*step+window_len]
        show_sigle_window(frame)


if __name__=='__main__':
    recorder=Recoder()
    recorder.load_spike_v()
    print('2')


    step = 'Step1000'
    task = 'Task1'

    v_mat=recorder.v[step][task].T
    show_sigle_window(v_mat)

    model=recorder.load_model(1200)
    w_rec=model.w_rec






