# Numerical inversion of precision matrix in Gaussian graphical model

## Summary

[See also the LaTeX version of the docs.](doc/doc.pdf)

The goal of this library is to solve the problem of calculating the inverse of a symmetric positive definite matrix (e.g. a covariance matrix) when the constraints are mixed between the covariance matrix `\Sigma` and the precision matrix `B = \Sigma^{-1}`. In particular, constraints of the following form are considered:
```
\Sigma_{ij} = \theta_{ij}
B_{kl} = 0
```
where `\theta_{ij}` are some given numerical values. It is also considered that the number of constraints matches the number of degrees of freedom. An `N x N` symmetric matrix has `N + (N choose 2) = N * (N+1) / 2` degrees of freedom.

* If all the constraints were on `\Sigma`, this problem would be trivially solved by computing the inverse `B = \Sigma^{-1}`.
* For small matrices, this problem can be solved analytically - for large matrices this is infeasible.
* In this library, the problem is solved by optimization.

## Two approaches

There are two main approaches:

1. Apply root finding to the matrix equation system
```
B * \Sigma - I = 0
```

2. Minimize the L2 loss:
```
\sum_{kl} ((B^{-1})_{kl} - \Sigma_{kl})^2
```

The first approach is generally the preferred one, using Newton's root finding method. See the [Newton's root finding method example](test/src/root_find_newton_5d.cpp).

The second one also works well, but typically you need a good optimizer like ADAM. Currently, only first order methods (in the gradients) are included. Two classes of optimizers are supported:
* Optimizers from the [Optim library](https://github.com/kthohr/optim).
* Several home-grown optimizers, including gradient descent (GD) and ADAM.
See the [ADAM L2 loss minimzer example](test/src/l2_adam_5d.cpp).

## Example figures

Minimization of the residuals from Newton's root finding method:
<img src="readme_figures/residuals.png" alt="drawing" width="400"/>

Convergence of the covariance matrix during minimizing the L2 loss:
<img src="readme_figures/cov.png" alt="drawing" width="400"/>

Convergence of the precision matrix during minimizing the L2 loss:
<img src="readme_figures/prec.png" alt="drawing" width="400"/>

## Building

Build out of a dedicated build directory;
```
mkdir build
cd build
cmake ..
make
make install
```

## Tests

See the [test](test) directory, which can be built in the same way:
```
cd test
mkdir build
cd build
cmake ..
```