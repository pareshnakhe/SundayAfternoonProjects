# Metropolis-Hastings Algorithm

All too often I use Bayesian inference algorithms (PyMC3) to capture the uncertainties in my results.
In this tiny project here, I attempt to lay bare the simplest of the algorithms used for inference.

I will not be saying much about the *how* and *what* here. I have used two blog posts to help me understand
the underlying concepts:
 1. https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/ (Focus on the math)
 2. https://twiecki.io/blog/2015/11/10/mcmc-sampling/ (Focus on the algorithm description)

 Some quick comments:

 MCMC sampling algorithms are often spoken of in context of inference algorithms. This is quite expected
 since the distributions we are trying to infer often don't have clean analytical forms. But I found to 
 helpful to think of this sampling algorithm in isolation, i.e. without the complications brought in by
 the Bayesian approach.

 Stated concisely, we can use MCMC sampling algorithm to approximate a probability distribution to which
 we only have a black-box access: given *x we see *Pr(x)*, the probability density at the point.