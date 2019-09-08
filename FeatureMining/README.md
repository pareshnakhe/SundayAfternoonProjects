# Finding Similar Features in Your Data

## Motivation

Let's suppose you work for Spotify in the song recommendation team. For every user, you have access to the complete history of songs played. For purposes of analytics, you want to find out the *defining set of feature* that play a major role in what the user listens to.

Specifically, say we have a data set with $n$ row and $m$ features. Features could be for example, language, region, genre, date-of-release... *Assuming* there exists some features which are *similar* across all data points, how will you find them.

## Disclaimer

The code snippet in this repository is my first approach to see if some of my ideas could actually work. This is not a complete solution. I am still working on it.
