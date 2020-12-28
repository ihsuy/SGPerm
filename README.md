# SGPerm

This repository contains the code for **Stochastic Gradient Permutation (SGPerm)** mentioned in:

> Yushi Qiu, Reiji Suda. [*Permute to Train: A New Dimension to Training Deep Neural Networks*](https://arxiv.org/abs/2003.02570v4), CoRR, 2020.

SGPerm trains DNNs like solving picture puzzles. 

Below is an analogy:

A picture puzzle game involves a picture frame and a set of picture fragments (the puzzle pieces) given in random
orders. The goal of this game is to permute this set of puzzle pieces within the picture frame to obtain a complete picture. As demonstrated in the figure below, we first propose a fourstep routine to solve such puzzles using human intuitions. 

<p align="left" style="float: left;">
  <img src="https://github.com/ihsuy/SGPerm/blob/main/img/puzzle.png?raw=true" height="600">
</p> 

- **Step 1**: Provide recommendation: For each slot on the picture
frame, we provide intuitive recommendations for moving puzzle
pieces over. In other words, for each slot, we list a set of puzzle
piece candidates which we would like to move to this slot.

- **Step 2**: Graph building: We connect these intuitions to form a
directed graph which contains all recommended movements as
shown below.

- **Step 3**: Cycle finding: We find appropriate cycles in the directed
graph built in Step 2 as shown below.

- **Step 4**: Permutation: We perform permutation following the
cycles.

If we are satisfied with the resulting picture from Step 4, we
return it, otherwise we repeat this routine from Step 1 continuing
from this resulting picture.

SGPerm uses the aforementioned
simple four-step routine to train DNNs, except that the intuitive
recommendations (Step 1) are computed using gradient-based
information, and the puzzle pieces are the weighted connections
associated with each neuron. Nevertheless, the “puzzle of a
DNN” is often high dimensional and has no straightforward
answer. In order to efficiently train DNNs, every step of this
routine needs to be elaborated.

<p align="left" style="float: left;">
  <img src="https://github.com/ihsuy/SGPerm/blob/main/img/graph%20building.png?raw=true" height="600">
</p> 

<p align="left" style="float: left;">
  <img src="https://github.com/ihsuy/SGPerm/blob/main/img/cycle%20finding.png?raw=true" height="600">
</p> 

Please refer to our [paper](https://arxiv.org/abs/2003.02570v4) for more details.

## Licence
MIT
