grid_name               (str)               name of your grid
test                    (bool)              if true, the code will run in test mode
remove_previous         (bool)              if true, the code will delete files associates with previous runs with this grid_name
n_cores                 (null or int)       number of cores to use for the run, uses all but one if set as null
grain_a_filename        (str)               name of the dust optical constant file for first grain type
grain_b_filename        (str)               " second grain type
grain_c_filename        (str)               " third grain type
grain_d_filename        (str)               " fourth grain type
effective_temp_grid     (array)             effective temperature(s) of BB used for model
inner_dust_temp_grid    (array)             inner dust temperature(s) of model
grain_type_a_grid       (array)             grain fraction (totalling 100%) for first grain type   
grain_type_b_grid       (array)             " second grain type
grain_type_d_grid       (array)             " fourth grain type, where the fraction for the third is the remainder
tau_number              (int)               number of individual optical depths to consider
tau_max                 (float)             maximum optical depth at 10um for models



DUSTY instructions for changing wavelength sampling

Change lambda_grid.dat to change wavelength sampling
Change userpar.inc
Recompile with gfortran dustyV2.f -std=legacy -o dusty

run srg_dusty_grid

