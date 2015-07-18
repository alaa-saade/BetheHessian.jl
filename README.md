BetheHessian.jl
===============

[![Build Status](https://travis-ci.org/alaa-saade/BetheHessian.jl.svg?branch=master)](https://travis-ci.org/alaa-saade/BetheHessian.jl)

A Julia package for applications of the Bethe Hessian operator in inference. 
Only one application is currently supported, namely matrix completion. Support for community detection will be added in the future. 

# Matrix Completion

Two functions are provided : demo_MC and complete. Both accept keyword arguments. Note that complete doesn't do any preprocessing. In particular, the matrix to be completed should be previously **centered**. demo_MC is just a wrapper around complete that applies complete to a randomly generated matrix.  

For usage and a list of the keyword arguments, type ?demo_MC or ?complete. 

# Community Detection

To do

# Ressources 

Matrix Completion: 
* *Matrix Completion from Fewer Entries: Spectral Detectability and Rank Estimation*, A Saade, F Krzakala, L Zdeborová - arXiv preprint arXiv:1506.03498, 2015 

Community Detection:
* *Spectral Clustering of Graphs with the Bethe Hessian*, A Saade, F Krzakala, L Zdeborová - NIPS2014
* *Spectral detection in the censored block model*, A Saade, F Krzakala, M Lelarge, L Zdeborová - ISIT2015