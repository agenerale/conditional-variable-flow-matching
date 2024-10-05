
# Conditional Variable Flow Matching

## Description

Conditional Flow Matching (CFM) is a fast way to train continuous normalizing flow (CNF) models. CFM is a simulation-free training objective for continuous normalizing flows that allows conditional generative modeling and speeds up training and inference. CFM's performance closes the gap between CNFs and diffusion models. To spread its use within the machine learning community, we have built a library focused on Flow Matching methods: TorchCFM. TorchCFM is a library showing how Flow Matching methods can be trained and used to deal with image generation, single-cell dynamics, tabular data and soon SO(3) data.

<p align="center">
<img src="assets/169_generated_samples_otcfm.png" width="600"/>
<img src="assets/8gaussians-to-moons.gif" />
</p>

The density, vector field, and trajectories of simulation-free CNF training schemes: mapping 8 Gaussians to two moons (above) and a single Gaussian to two moons (below). Action matching with the same architecture (3x64 MLP with SeLU activations) underfits with the ReLU, SiLU, and SiLU activations as suggested in the [example code](https://github.com/necludov/jam), but it seems to fit better under our training setup (Action-Matching (Swish)).

The models to produce the GIFs are stored in `examples/models` and can be visualized with this notebook: [![notebook](https://img.shields.io/static/v1?label=Run%20in&message=Google%20Colab&color=orange&logo=Google%20Cloud)](https://colab.research.google.com/github/atong01/conditional-flow-matching/blob/master/examples/notebooks/model-comparison-plotting.ipynb).

We also have included an example of unconditional MNIST generation in `examples/notebooks/mnist_example.ipynb` for both deterministic and stochastic generation. [![notebook](https://img.shields.io/static/v1?label=Run%20in&message=Google%20Colab&color=orange&logo=Google%20Cloud)](https://colab.research.google.com/github/atong01/conditional-flow-matching/blob/master/examples/notebooks/mnist_example.ipynb).

## The torchcfm Package

In our version 1 update we have extracted implementations of the relevant flow matching variants into a package `torchcfm`. This allows abstraction of the choice of the conditional distribution `q(z)`. `torchcfm` supplies the following loss functions:

- `ConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = q(x_0) q(x_1)$
- `ExactOptimalTransportConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = \\pi(x_0, x_1)$ where $\\pi$ is an exact optimal transport joint. This is used in \[Tong et al. 2023a\] and \[Poolidan et al. 2023\] as "OT-CFM" and "Multisample FM with Batch OT" respectively.
- `TargetConditionalFlowMatcher`: $z = x_1$, $q(z) = q(x_1)$ as defined in Lipman et al. 2023, learns a flow from a standard normal Gaussian to data using conditional flows which optimally transport the Gaussian to the datapoint (Note that this does not result in the marginal flow being optimal transport).
- `SchrodingerBridgeConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = \\pi\_\\epsilon(x_0, x_1)$ where $\\pi\_\\epsilon$ is an entropically regularized OT plan, although in practice this is often approximated by a minibatch OT plan (See Tong et al. 2023b). The flow-matching variant of this where the marginals are equivalent to the Schrodinger Bridge marginals is known as `SB-CFM` \[Tong et al. 2023a\]. When the score is also known and the bridge is stochastic is called \[SF\]2M \[Tong et al. 2023b\]
- `VariancePreservingConditionalFlowMatcher`: $z = (x_0, x_1)$ $q(z) = q(x_0) q(x_1)$ but with conditional Gaussian probability paths which preserve variance over time using a trigonometric interpolation as presented in \[Albergo et al. 2023a\].

