# paired_design_codes

Codes for my MA thesis: Paired 2x2 Factorial Design for Treatment Effect Identification and Estimation
  in the Presence of Paired Interference and Noncompliance.
  
  -- paired_design.py: contains the implementation of the average treatment effect estimators
  -- assignment_mechanism.py: illustrates how individual i.i.d. assignement converges to pairs being assigned in proportion 25-25-25-25%
  -- symbolic_inversion.py: does the symbolic inversion needed fro the proff of Theorem [Full-sample Identification Strategy]
  -- mc_symmetry.py: does the Monte Carlo simulation for consistency-check when symmetry assumptions are violated
                     saves results to biases_*.c and variances_*.c cloudpickle dictionaries
  -- mc_symmetry_plots.py: creates plot from biases_*.c and variances_*.c
