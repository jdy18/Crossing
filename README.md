

A Python package used for simulating spiking neural networks (SNNs) on CPUs or GPUs using [PyTorch](http://pytorch.org/) `Tensor` functionality.



## Requirements

- Python >=3.7.10,<3.10

## Setting things up

## Using Pip
To install the most recent stable release from the GitHub repository

```
pip install git+https://github.com/BindsNET/bindsnet.git
```

## Getting started

To run a near-replication of the SNN from [this paper](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full#), issue

```
cd examples/mnist
python eth_mnist.py
```

There are a number of optional command-line arguments which can be passed in, including `--plot` (displays useful monitoring figures)
## Running the tests

Issue the following to run the tests:

```
python -m pytest test/
```

Some tests will fail if Open AI `gym` is not installed on your machine.


## Result

<p align="middle">
<img src="data/task2/result" alt="BindsNET%20Benchmark"  width="503" height="403">
</p>

All simulations run on Ubuntu 16.04 LTS with Intel(R) Xeon(R) CPU E5-2687W v3 @ 3.10GHz, 128Gb RAM @ 2133MHz, and two GeForce GTX TITAN X (GM200) GPUs. Python 3.6 is used in all cases. Clock time was recorded for each simulation run. 



## Contributors

- Daniel Saunders ([email](mailto:djsaunde@cs.umass.edu))
