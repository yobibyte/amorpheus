## My Body is a Cage: the Role of Morphology in Graph-Based Incompatible Control
### ICLR 2021
#### [OpenReview](https://openreview.net/forum?id=N3zUDGN5lO) | [Arxiv](https://arxiv.org/abs/2010.01856)

[Vitaly Kurin](https://twitter.com/y0b1byte), [Maximilian Igl](https://twitter.com/MaxiIgl), [Tim Rocktäschel](https://twitter.com/_rockt), [Wendelin Boehmer](https://twitter.com/WendelinBoehmer), [Shimon Whiteson](https://twitter.com/shimon8282)

### TL;DR 

Providing morphological structure as an input graph is not a useful inductive bias in Graph-Based Incompatible Control.
If we let the structural information go, we can do better with transformers.

```
@inproceedings{
kurin2021my,
title={My Body is a Cage: the Role of Morphology in Graph-Based Incompatible Control},
author={Vitaly Kurin and Maximilian Igl and Tim Rockt{\"a}schel and Wendelin Boehmer and Shimon Whiteson},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=N3zUDGN5lO}
}
```

![](https://i.guim.co.uk/img/media/b251ae63d78acf9389a8fce146580483ecdd2253/57_6_1416_849/master/1416.jpg?width=645&quality=45&auto=format&fit=max&dpr=2&s=e2908d58f0687b4bded76519f006dbd1)

## Setup

All the experiments are done in a Docker container.
To build it, run `./docker_build.sh <device>`, where `<device>` can be `cpu` or `cu101`. It will use CUDA by default.

To build and run the experiments, you need a MuJoCo license. Put it to the root folder before running `docker_build.sh`. 


## Running

```
./docker_run <device_id> # either GPU id or cpu
cd amorpheus             # select the experiment to replicate
bash cwhh.sh             # run it on a task
```

We were using [Sacred](https://github.com/IDSIA/sacred) with a remote MongoDB for experiment management.
For release, we changed Sacred to log to local files instead.
You can change it back to MongDB if you provide credentials in `modular-rl/src/main.py`. 

## Acknowledgement

- The code is built on top of [SMP](https://github.com/huangwl18/modular-rl) repository. 
- NerveNet Walkers environment are
taken and adapted from [the original repo](https://github.com/WilsonWangTHU/NerveNet).
- Initial implementation of the transformers was taken from the [official Pytorch tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) and modified thereafter. 
