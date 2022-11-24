# -*- coding: utf-8 -*-


import torch
import numpy as np
import numpy.random as rd
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from recorder import Recoder

class CueAccumulationDataset(torch.utils.data.Dataset):
    """Adapted from the original TensorFlow e-prop implemation from TU Graz, available at https://github.com/IGITUGraz/eligibility_propagation"""

    def __init__(self, args, type):

        in_class=4
        n_class = 2
        n_cues     = 7
        f0         = 40
        t_cue      = 100
        t_wait     = 200
        n_symbols  = 4
        p_group    = 0.3
        
        self.dt         = 1e-3
        self.t_interval = 150
        self.seq_len    = 154 + t_wait # n_cues*self.t_interval
        self.n_in       = 40
        self.n_out      = 2    # This is a binary classification task, so using two output units with a softmax activation redundant
        n_channel       = self.n_in // n_symbols
        prob0           = f0 * self.dt
        t_silent        = self.t_interval - t_cue
        
        if (type == 'train'):
            all_length = args.train_len
        else:
            all_length = args.test_len

        y_lable=[0,1,0,1]
        for clas in range(in_class):
            length=int(all_length/in_class)

            color_img = cv2.imread(str(clas)+".png")
            reImg = cv2.resize(color_img, (154,20), interpolation=cv2.INTER_CUBIC)
            gray_img = cv2.cvtColor(reImg, cv2.COLOR_RGB2GRAY).T
            _,bina_img=cv2.threshold(gray_img, 255/2, 1, cv2.THRESH_BINARY)
            # Randomly assign group A and B
            prob_choices = np.array([p_group, 1 - p_group], dtype=np.float32)
            idx = rd.choice([0, 1], length)
            probs = np.zeros((length, 2), dtype=np.float32)
            # Assign input spike probabilities
            probs[:, 0] = prob_choices[idx]
            probs[:, 1] = prob_choices[1 - idx]

            cue_assignments = np.zeros((length, n_cues), dtype=np.int)
            # For each example in batch, draw which cues are going to be active (left or right)
            for b in range(length):
                cue_assignments[b, :] = rd.choice([0, 1], n_cues, p=probs[b])

            # Generate input spikes
            input_spike_prob = np.zeros((length, self.seq_len, self.n_in))
            t_silent = self.t_interval - t_cue
            for b in range(length):
                input_spike_prob[b, :154, :20] = bina_img*prob0*10

            # Recall cue and background noise
            input_spike_prob[:, -self.t_interval:, 2*n_channel:3*n_channel] = prob0 #提示输出的线索信号
            input_spike_prob[:, :, 3*n_channel:] = prob0/4.   #噪声部分
            input_spikes = generate_poisson_noise_np(input_spike_prob,freezing_seed=args.seed)
            try:
                self.x = torch.cat([self.x,torch.tensor(input_spikes).float()],0)
            except:
                self.x = torch.tensor(input_spikes).float()

            # Generate targets
            target_nums = np.zeros((length, self.seq_len), dtype=np.int)

            target_nums[:, :] = np.transpose(np.tile(np.array([y_lable[clas] for i in range(length)]), (self.seq_len, 1)))

            try:
                self.y = torch.cat([self.y,torch.tensor(target_nums).long()],0)
            except:
                self.y = torch.tensor(target_nums).long()


            #打乱前两类
            if clas==1 and type == 'train':
                ind=torch.randperm(self.x.size(0))
                self.x=self.x[ind]
                self.y = self.y[ind]
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]



def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    rd.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)  # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    else:
        torch.manual_seed(args.seed)


    if args.dataset == "cue_accumulation":
        print("=== Loading cue evidence accumulation dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_cue_accumulation(args, kwargs)
    else:
        print("=== ERROR - Unsupported dataset ===")
        sys.exit(1)
        
    print("Training set length: "+str(args.full_train_len))
    print("Test set length: "+str(args.full_test_len))

    recorder=Recoder()
    return (device, train_loader, traintest_loader, test_loader,recorder)

def _init_fn(worker_id):
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)

def load_dataset_cue_accumulation(args, kwargs):

    trainset = CueAccumulationDataset(args,"train")
    testset  = CueAccumulationDataset(args,"test")

    train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=args.shuffle,worker_init_fn=_init_fn , **kwargs)
    traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False       ,worker_init_fn=_init_fn , **kwargs)
    test_loader      = torch.utils.data.DataLoader(testset , batch_size=args.test_batch_size, shuffle=False       ,worker_init_fn=_init_fn , **kwargs)
    
    args.n_classes      = trainset.n_out
    args.n_steps        = trainset.seq_len
    args.n_inputs       = trainset.n_in
    args.dt             = trainset.dt
    args.classif        = True
    args.full_train_len = len(trainset)
    args.full_test_len  = len(testset)
    args.delay_targets  = trainset.t_interval
    args.skip_test      = False
    
    return (train_loader, traintest_loader, test_loader)


def generate_poisson_noise_np(prob_pattern, freezing_seed=0):
    if isinstance(prob_pattern, list):
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    shp = prob_pattern.shape

    if not(freezing_seed is None): rng = rd.RandomState(freezing_seed)
    else: rng = rd.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return spikes
