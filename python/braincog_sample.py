import torch
from torch import nn
import matplotlib.pyplot as plt
from braincog.base.node.node import BaseNode, LIFNode, IzhNode
from braincog.base.strategy.surrogate import *
from braincog.base.utils.visualization import spike_rate_vis, spike_rate_vis_1d


x = torch.rand(1, 10, 10)

lif = LIFNode(threshold=0.6, tau=2000.)
## Reset Before use
lif.n_reset()
#spike = lif(x)
mem = []
spike = []
for t in range(150):
    # x = torch.tensor(0.31)
    x = torch.rand(5,100)*2000
    spike.append(lif(x))
    mem.append(lif.mem)

mem = torch.stack(mem)
spike = torch.stack(spike)

outputs = torch.max(mem[:,0,1], spike[:,0,1]).detach().cpu().numpy()

plt.plot(outputs)
plt.show()
#spike_rate_vis(x)
