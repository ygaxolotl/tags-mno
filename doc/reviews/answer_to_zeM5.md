# Answer to zeM5

First of all, thank you so much for taking the time and giving us such extensive feedback. I appreciate that you value our contribution in augmenting classical solvers.
It seems that your biggest comments are regarding the resolution-independence and a comparison with a pure ML based surrogate model.  

- Regarding resolution indepence, I agree that our current implementation of FNO does not work on irregular meshes. We explicitly mention in the limitation section that our implementation assumes fixed spectral resolution and an equispaced grid. I will clarify that we define „resolution-independent“ to refer to the independence of step size in equispaced grids. As a side note, FNO could be extended to non-equispaced grids by implementing Fourier transforms on non-equispaced grids (Dutt & Rokhlin, 1993).
- I really appreciate that you are asking for a comparison with a pure ML based model. With your feedback, we trained & tested a pure FNO model to forecast the next global large-scale state, X(t+1), given the previous global large-scale state, X(t), on quasi-geostrophic turbulence. The FNO uses the same hyperparameters and train/test split as MNO. We then compare the RMSE of the predicted vs. the ground-truth large-scale solution, 2 and 10 time steps ahead. When rolling out those two models, MNO is ~3-times more accurate than the pure FNO after 2 time steps and 27-times more accurate after 10 time steps. The RMSE for 2-time steps is ('mno': 0.0046,'null': 0.0051, 'pure-fno': 0.0140) and for 10 time steps ('mno': 0.0125, 'null': 0.0243, 'pure-fno': 0.3389). It is not officially allowed to include new results in the rebuttal, so please consider this information as interesting insight outside of your score. We will robustify this experiment with a deeper hyperparameter search of the FNO and update the paper.


## Questions:
- According to equation 1, can we understand the approach as learning a residual for inaccurate low-resolution simulation?
  - Yes. MNO can be interpreted as learning a residual or bias term that corrects a low-resolution simulation on every time-step. 
- what is the missing figure in Figure 7?
  - This is a bit confusing indeed. The 2nd figure from the right (mid-right) is a baseline model that predicts the parametrization to be the mean over the full dataset. In our set-up of QG turbulence the mean is zero, so this models predicts a constant zero. The model is also called ‚null‘ parametrization as mentioned in Fig. 7 caption. 
- what is the unit of t in Figure 6? how many timesteps for that?
  - Good catch. In multiscale Lorenz96, one model time unit is 200 time steps (step size, Δt = 0.005). It’s mentioned in the appendix, but I will add this to the figure caption.  
- what is the error for just doing low-resolution simulation, without modeling the fine-resolution dynamic?
  - This is a good question. The ‚null‘ parametrization is a model that just runs the low-resolution simulation. For QG turbulence, ‚null‘ and ‚climatology‘ parametrizations and the results are reported in Section 4.4. For multiscale Lorenz96, I should add the ‚null‘ parametrization experiment. Thank you for pointing this out. 