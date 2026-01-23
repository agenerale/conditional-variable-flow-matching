<div align="center">

# Conditional Variable Flow Matching


</div>

## Description

Conditional Variable Flow Matching (CVFM) is a robust extension to Flow Matching for training amortized conditional continuous normalizing flows (CNF). CVFM enables the disentanglement of conditional dynamics from unpaired training data, requiring only an $(x,y)$ pair observed at a given time. 

<p align="center">
<img src="imgs/2-moons_to_2-moons.gif" width="800"/>
</p>

The trajectories and densities for mapping from a continuous conditioning mapping from two moons to two moons rotated about the origin by 90 degrees. Trajectories are colored by the conditioning variable associated with the mapping.

The test cases presented above, along with additional 2D mapping found in the paper [![CVFM Preprint](http://img.shields.io/badge/paper-arxiv.2302.00482-B31B1B.svg)](https://arxiv.org/abs/2411.08314) can be found in the notebook `examples/cvfm_tutorial.ipynb`: [![notebook](https://img.shields.io/static/v1?label=Run%20in&message=Google%20Colab&color=orange&logo=Google%20Cloud)](https://colab.research.google.com/github/agenerale/conditional-variable-flow-matching/blob/main/examples/cvfm_tutorial.ipynb). Code to specifically generate the animation presented above can be found in `/toy_demos/2moons_2moons.py`. 

Code corresponding to the other 2D example problems can be found in `/toy_demos/8gauss_2moon.py` and `/toy_demos/8gauss_8gauss.py`, while the presented materials microstructure evolution case study main code can be found in `/phase_field/main.py`.

## How to cite

This repository contains the code to reproduce the main experiments for CVFM:

[Conditional Variable Flow Matching: Transforming Conditional Densities with Amortized Conditional Optimal Transport](https://arxiv.org/abs/2411.08314). This work introduces **Conditional Variable Flow Matching** (CVFM), an extension to CFM which enables the disentanglement and modeling of conditional dynamics from unpaired training data.

If you find this code useful in your research, please cite us (expand for BibTeX):

<details>
<summary>
A. Generale, A. Robertson, S. Kalidindi. Conditional Variable Flow Matching: Transforming Conditional Densities with Amortized Conditional Optimal Transport, 2024.
</summary>

```bibtex
@article{generale2024cvfm,
	title = {Conditional {Variable} {Flow} {Matching}: {Transforming} {Conditional} {Densities} with {Amortized} {Conditional} {Optimal} {Transport}},
	shorttitle = {Conditional {Variable} {Flow} {Matching}},
	url = {http://arxiv.org/abs/2411.08314},
	doi = {10.48550/arXiv.2411.08314},
	abstract = {Forecasting stochastic nonlinear dynamical systems under the influence of conditioning variables is a fundamental challenge repeatedly encountered across the biological and physical sciences. While flow-based models can impressively predict the temporal evolution of probability distributions representing possible outcomes of a specific process, existing frameworks cannot satisfactorily account for the impact of conditioning variables on these dynamics. Amongst several limitations, existing methods require training data with paired conditions and are developed for discrete conditioning variables. We propose Conditional Variable Flow Matching (CVFM), a framework for learning flows transforming conditional distributions with amortization across continuous conditioning variables - permitting predictions across the conditional density manifold. This is accomplished through several novel advances. In particular, simultaneous sample conditioned flows over the main and conditioning variables. In addition, motivated by theoretical analysis, a conditional Wasserstein distance combined with a loss reweighting kernel facilitating conditional optimal transport. Collectively, these advances allow for learning system dynamics provided measurement data whose states and conditioning variables are not in correspondence. We demonstrate CVFM on a suite of increasingly challenging problems, including discrete and continuous conditional mapping benchmarks, image-to-image domain transfer, and modeling the temporal evolution of materials internal structure during manufacturing processes. We observe that CVFM results in improved performance and convergence characteristics over alternative conditional variants.},
	urldate = {2024-12-08},
	publisher = {arXiv},
	author = {Generale, Adam P., Robertson, Andreas E., Kalidindi, Surya R.},
	month = nov,
	year = {2024},
	note = {arXiv:2411.08314 [cs]},
	keywords = {Computer Science - Machine Learning},
}
```
</details>

## Project Structure

The directory structure looks like this:

```
│
├── examples              <- Jupyter example notebooks
│
│── phase_field           <- Material phase field evolution experiments
│   ├── models            <- Model for material dynamics
|   ├── main.py           <- Main execution script for training / evaluation
|
│── toy_demos             <- 2D discrete and continuous conditioning experiments
│   ├── 2moons_2moons.py  
│   ├── 8gauss_8gauss.py  
│   ├── 8gauss_2moon.py  
|
└── README.md
```

## License

Conditional Variable Flow Matching is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 Adam Generale

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
