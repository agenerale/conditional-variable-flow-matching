<div align="center">

# Conditional Variable Flow Matching


</div>

## Description

Conditional Variable Flow Matching (CVFM) is a robust extension to Flow Matching for training amortized conditional continuous normalizing flows (CNF). CVFM enables the disentanglement of conditional dynamics from unpaired training data, requiring only an $(x,y)$ pair observed at a given time. 

<p align="center">
<img src="imgs/2-moons_to_2-moons.gif" width="800"/>
</p>

The trajectories and densities for mapping from a continuous conditioning mapping from two moons to two moons rotated about the origin by 270 degrees. Trajectories are colored by the conditioning variable associated with the mapping.

The test cases presented above, along with additional 2D mapping found in the paper (insert paper) can be found in the notebook `examples/cvfm_tutorial.ipynb`: [![notebook](https://img.shields.io/static/v1?label=Run%20in&message=Google%20Colab&color=orange&logo=Google%20Cloud)](https://colab.research.google.com/github/agenerale/conditional-variable-flow-matching/blob/main/examples/cvfm_tutorial.ipynb). Code to specifically generate the animation presented above can be found in `/toy_demos/2moons_2moons.py`. 

Code corresponding to the other 2D example problems can be found in `/toy_demos/8gauss_2moon.py` and `/toy_demos/8gauss_8gauss.py`.


