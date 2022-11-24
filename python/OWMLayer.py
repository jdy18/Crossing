# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor  # run on GPU


class OWMLayer:

    def __init__(self,  shape, alpha=0):

        self.input_size = shape[0]
        self.output_size = shape[1]
        self.alpha = alpha
        self.input=None
        with torch.no_grad():
            self.P = Variable((1.0/self.alpha)*torch.eye(self.input_size).type(dtype))
    def record_input(self,input):
        self.input=torch.cat()
    def force_learn(self, w, input_, learning_rate, alpha=1.0):  # w(outdim,indim) input_(batch,input_size)
        self.r = torch.mean(input_, 0, True)
        self.k = torch.mm(self.P, torch.t(self.r))
        self.c = 1.0 / (alpha + torch.mm(self.r, self.k))  # 1X1
        self.P.sub_(self.c*torch.mm(self.k, torch.t(self.k)))
        #w.data -= (learning_rate * torch.mm(self.P.data, w.grad.data.T)).T
        #w.grad.data.zero_()
        w.grad.data=(learning_rate * torch.mm(self.P.data, w.grad.data.T)).T

    def predit_lable(self, input_, w,):
        return torch.mm(input_, w)
