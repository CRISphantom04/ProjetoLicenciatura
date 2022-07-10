import rasterio
from SDSim import species
from SDSim.species import create_suitability_map, plot_suitability
import numpy as np
import os
from sympy import *
import copy
from moviepy.editor import *

def setup(specie_name, ticks, env_variables, u, sd, n, death_rate, birth_rate, spread_rate, path, delta, user_path,
          agents_quantity=500, max_quantity=100, model=0, neighbourhood=1):
           
    print('Read Data')
    env_data = []
    for variable in env_variables:
        value = species.read_map(path, variable)
        transform = value.transform
        value = value.read(1, masked=True)
        value = value.astype(np.float)
        value.filled(np.nan)
        env_data.append(value.filled(np.nan))


    converted = species.converted(env_data, u, sd)
    fitness = species.compute_model(converted, model)
    normalize = species.normalize(fitness, model, n)

    create_suitability_map(normalize, path+'/suitability.asc', transform)
    plot_suitability(normalize, path)

    rows = len(normalize)
    cols = len(normalize[0])

    patches = species.create_patches(rows, cols, normalize)
    previous_patches = species.create_patches(rows, cols, normalize)
    previous_patches = species.fill_patches_random(previous_patches, agents_quantity, max_quantity)
    patches = copy.deepcopy(previous_patches)

    
    for k in range(0, ticks):
        if neighbourhood == 0:
            species.run(patches, previous_patches, birth_rate, death_rate, spread_rate, max_quantity)
        else:
            species.run_weighted(patches, previous_patches, birth_rate, death_rate, spread_rate, max_quantity)

        if k % delta == 0:
            prev = species.create_first_distribution(
                previous_patches, max_quantity)
            current = species.create_distribution_file(
                patches, specie_name, k, max_quantity, path)
            species.create_map(specie_name+str(k)+'.txt', specie_name +
                              str(k), transform, k, specie_name, path, max_quantity)

        previous_patches = copy.deepcopy(patches)
        print(f'Iteration {k}')
 

