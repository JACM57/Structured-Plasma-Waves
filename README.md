# Structured-Plasma-Waves
This repository contains the scripts developed during a research internship at GoLP (Instituto Superior Técnico), under the supervision of Jorge Vieira and Rafael Almeida.

This work focused on studying and simulating 1D and 2D electrostatic plasma waves. Initially, a theoretical Python code was developed to explore wavepacket formation and propagation with arbitrary plane wave distributions and parameters. Then, this tool was transposed to a more realistic simulation environment, using the particle-in-cell (PIC) code ZPIC.

Brief description of each file:

Structured_Plasma_Waves.pdf -> Internship report containing a brief overview and discussion of some results.

1D_Theory.py -> Visualization of the theoretical (no particles are used) evolution of the density, electric field, and velocity wavepackets in 1D for an arbitrary thermal velocity.

2D_Theory.py -> Visualization of the theoretical evolution of the density, electric field, and velocity wavepackets in 2D for an arbitrary thermal velocity.

1D_ZPIC_SingleWave_Parameter_Optimization.ipynb -> Performs parameter sweeps for a 1D ZPIC single-wave setup. The goal is to optimize the initial conditions to maximize the duration over which a plasma wave maintains its overall structure.

1D_ZPIC_SingleWave_Parameter_Optimization_FFT.ipynb -> Performs parameter sweeps for the optimization of a 1D ZPIC single-wave setup, based on the analysis of the Electric field's FFT.

1D_ZPIC_Wavepacket.ipynb -> Visualization of the evolution of the density, electric field, and velocity wavepackets in a 1D ZPIC setup with arbitrary thermal velocity.

2D_ZPIC_Wavepacket.ipynb ->Visualization of the evolution of the density, electric field, and velocity wavepackets in a 2D ZPIC setup with arbitrary thermal velocity. The wavepackets are invariant along the y-direction.

ZPIC_DispersionRelation.ipynb -> Visualization of the 1D and 2D numerical dispersion relations using ZPIC.
