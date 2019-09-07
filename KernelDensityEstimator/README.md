# Posterior Probability Computation

## Motivation

Suppose you organize comedy shows. The tickets for a given show goes on sale about 3 weeks before the event is supposed to take place. Also, assume we have sales data for a lot of past shows, like when and how many tickets for a given show were sold. Suppose for the upcoming show, one week before, your boss asks how many tickets do we expect to sell in the last week. How would you go about that?

## Mathematical Description

At its core, the method implemented here works with vectors. Each data sample, for example, is a vector capturing the number of tickets sold in each of the weeks before the event, (10, 25, 12) for example. This method assumes that such a tuple is drawn from an unknown *true* distribution.

The solution strategy is to approximate the distribution using the data available. We do this using the **KernelDensity** method from **sklearn.neighbours**. For a new incomplete data sample, say (15, 19, _), we use this approximated distribution together with available data (i.e. the number of tickets sold in the first and second week) to compute a posterior distribution on the number of tickets sold in the third week.

I have implemented a toy example in this post to demonstrate the idea.