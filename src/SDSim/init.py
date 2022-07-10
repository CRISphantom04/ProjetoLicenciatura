from setup import setup
import numpy as np



if __name__ == "__main__":
    # env_variables: environmental variables.
    # n: quantity of environmental variables.
    # delta: maps sample.
    # ticks: number of iterations.
    # path: environmental variables path.
    # user_path: simulation results path.
    # sd: standard deviation of each environmental variables
    # u: mean of each environmental variables.


    ticks = 20
    delta = 1 # delta < ticks
    death_rate = 0.25 # 0 < death rate < 1 
    birth_rate = 0.5 # 0 < birth rate < 1 
    spread_rate = 0.4 # 0 < spread rate < 1 
    path = r'Maps10'
    user_path = r'tmp'
    specie_name = 'apis'
    n = 4

    env_variables = np.array(['Present_tann_10km.asc', 'Present_rfseas_10km.asc',
                              'Present_mntcm_10km.asc', 'Present_mxtwm_10km.asc'])

    sd = np.array([2.5685735, 12.7725651, 2.6066717, 3.3386669])
    u = np.array([17.3, 47.1037037, 4.2, 29.7044444])


    setup(specie_name, ticks, env_variables, u, sd, n, death_rate, birth_rate, spread_rate, path, delta, user_path)
        
    
