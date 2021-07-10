.. evaLEs documentation master file, created by
   sphinx-quickstart on Mon Jul  5 16:08:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to evaLEs's documentation!
==================================

Introduction
------------
evaLEs is a package based on Python scientific ecosystem allowing 
to calculate the lyapunov spectrum of a generic Dynamical System described 
by ODEs or a map.

In order to do this the package implement the algorithm described by Benedettin et al.

Install
--------
Install via pip::

    pip install evaLEs

Contents
--------

Basically this package have one function: computeLE. This function is implemented in two version: 

.. toctree::
   :maxdepth: 2
   :glob:

   evaLEs

To Do
-----

In order to evaluate the Lyapunov spectrum the algorithm integrate the dynamical system 
with a Runge Kutta method of 4th order. The next goal is to leave to user to choose 
is own integration method (for example using one of the integrators implemented in 
scipy.integrate)


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
