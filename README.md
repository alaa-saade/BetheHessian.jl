BetheHessian.jl
===============

[![Build Status](https://travis-ci.org/alaa-saade/BetheHessian.jl.svg?branch=master)](https://travis-ci.org/alaa-saade/BetheHessian.jl)

A Julia package for applications of the Bethe Hessian operator in inference. 
Only one application is currently supported, namely matrix completion. Support for community detection will be added in the future. 

# Matrix Completion

Two functions are provided : demo and complete. Both accept keyword arguments. Note that complete doesn't do any preprocessing. In particular, the matrix to be completed should be previously **centered**.

For usage and a list of the keyword arguments, type ?demo or ?complete. 