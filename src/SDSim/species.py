import rasterio
from rasterio import plot
import os
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr
from sympy import *
import numpy as np
import math
from matplotlib import pyplot
import random
import matplotlib
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('Agg')


class Patch:
    # initialize the attributes: coordinates, suitability value,
    # and the environment variables from the maps (temp, prec, elev...)
    def __init__(self, quantity, fitness):
        self.quantity = quantity
        self.fitness = fitness

    def get_total_quantity(self):
        return sum(self.quantity)

    def find_neighbours(self, patches, x, y):
        if x == 0:
            if y == 0:
                neighbours = np.array(
                    [patches[x + 1][y], patches[x + 1][y + 1], patches[x][y+1]])
                neighbours_direction = np.array([180, 135, 90])
            elif y == len(patches[0])-1:
                neighbours = np.array(
                    [patches[x + 1][y], patches[x][y - 1], patches[x+1][y-1]])
                neighbours_direction = np.array([180, 270, 225])
            else:
                neighbours = np.array(
                    [patches[x+1][y], patches[x+1][y-1], patches[x][y-1], patches[x][y+1], patches[x+1][y+1]])
                neighbours_direction = np.array([180, 255, 270, 90, 135])
        elif x == len(patches)-1:
            if y == 0:
                neighbours = np.array(
                    [patches[x][y + 1], patches[x-1][y], patches[x-1][y+1]])
                neighbours_direction = np.array([90, 0, 45])
            elif y == len(patches[0]) - 1:
                neighbours = np.array(
                    [patches[x][y - 1], patches[x-1][y], patches[x-1][y-1]])
                neighbours_direction = np.array([270, 0, 315])
            else:
                neighbours = np.array(
                    [patches[x-1][y], patches[x-1][y-1], patches[x][y-1], patches[x][y+1], patches[x-1][y+1]])
                neighbours_direction = np.array([0, 315, 270, 90, 45])
        else:
            if y == 0:
                neighbours = np.array(
                    [patches[x][y+1], patches[x+1][y], patches[x-1][y], patches[x+1][y+1], patches[x-1][y+1]])
                neighbours_direction = np.array([90, 180, 0, 135, 45])
            elif y == len(patches[0]) - 1:
                neighbours = np.array(
                    [patches[x][y-1], patches[x+1][y], patches[x-1][y], patches[x+1][y-1], patches[x-1][y-1]])
                neighbours_direction = np.array([270, 180, 0, 225, 315])
            else:
                neighbours = np.array([patches[x+1][y+1], patches[x][y+1], patches[x-1][y+1], patches[x-1]
                                       [y], patches[x+1][y], patches[x-1][y-1], patches[x][y-1], patches[x+1][y-1]])
                neighbours_direction = np.array(
                    [135, 90, 45, 0, 180, 315, 270, 225])

        my_neighbours = []
        my_neighbours_direction = []

        for index, neighbour in enumerate(neighbours):
            if not math.isnan(neighbour.fitness):
                my_neighbours.append(neighbour)
                my_neighbours_direction.append(neighbours_direction[index])
        return my_neighbours, my_neighbours_direction

    def select_neighbour(self, neighbours):
        fitness = sum([neighbour.fitness for neighbour in neighbours])
        r = random.uniform(0, 1)
        s = 0
        for neighbour in neighbours:
            s += neighbour.fitness/fitness
            if s >= r:
                return neighbour

    def reproduce_weighted(self, previous_self, birth_rate, death_rate, spread_rate, max_quantity, previous_neighbours, current_neighbours):
        self.quantity += previous_self.quantity * birth_rate * previous_self.fitness - \
            previous_self.quantity * death_rate * (2 - previous_self.fitness)
        s = sum([n.fitness for n in previous_neighbours[0]])

        for neighbour in current_neighbours[0]:
            neighbour.quantity += previous_self.quantity * spread_rate * neighbour.fitness/s
            neighbour.quantity = threshold(neighbour.quantity, 0, max_quantity)

        self.quantity -= previous_self.quantity * spread_rate
        self.quantity = threshold(self.quantity, 0, max_quantity)

    def reproduce_direction_weighted(self, previous_self, birth_rate, death_rate, spread_rate, max_quantity, previous_neighbours, current_neighbours, dm, model):
        self.quantity += previous_self.quantity * birth_rate * previous_self.fitness - \
            previous_self.quantity * death_rate * (2 - previous_self.fitness)

        k = [(180.0 - min(np.abs(dm-k), np.abs(dm+(360-k))))/180.0
             for k in previous_neighbours[1]]

        # If Additive.
        if model == 0:
            s = sum([n.fitness + k[i]
                     for i, n in enumerate(previous_neighbours[0])])

            for i, neighbour in enumerate(current_neighbours[0]):
                fitness = neighbour.fitness + k[i]
                neighbour.quantity += previous_self.quantity * spread_rate * fitness / s
                neighbour.quantity = threshold(
                    neighbour.quantity, 0, max_quantity)

        # If Multiplicative
        else:
            s = sum([n.fitness * k[i]
                     for i, n in enumerate(previous_neighbours[0])])
            for i, neighbour in enumerate(current_neighbours[0]):
                fitness = neighbour.fitness * k[i]
                neighbour.quantity += self.quantity * spread_rate * fitness / s
                neighbour.quantity = threshold(
                    neighbour.quantity, 0, max_quantity)

        self.quantity -= previous_self.quantity * spread_rate
        self.quantity = threshold(self.quantity, 0, max_quantity)

    def reproduce(self, previous_self, birth_rate, death_rate, spread_rate, max_quantity, previous_neighbours, current_neighbours):
        self.quantity += previous_self.quantity * birth_rate * previous_self.fitness - \
            previous_self.quantity * death_rate * (2 - previous_self.fitness)

        for neighbour in current_neighbours[0]:
            neighbour.quantity += previous_self.quantity * \
                spread_rate / len(current_neighbours[0])
            neighbour.quantity = threshold(neighbour.quantity, 0, max_quantity)

        self.quantity -= previous_self.quantity * spread_rate
        self.quantity = threshold(self.quantity, 0, max_quantity)


def threshold(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def read_map(path, file):
    fp = os.path.join(path, file)
    return rasterio.open(fp)


def converted(env_variables, u, sd):
    for i, env_variable in enumerate(env_variables):
        env_variables[i] = (env_variables[i] - u[i]) / sd[i]
    return env_variables


def compute_model(env_variables, model):
    if model == 0:
        s = 0
    else:
        s = 1
    function = lambdify(
        sympify('x'), '1/(sqrt(2*pi)) * exp( - x**2 / 2)', 'numpy')
    for i, env_variable in enumerate(env_variables):
        if model == 0:
            s += function(env_variable)
        else:
            s *= function(env_variable)
    return s


def normalize(fitness, model, n):
    f = 1/(math.sqrt(2*math.pi)) * math.exp(- 0**2 / 2)
    if model == 0:
        maximum = f*n
    else:
        maximum = f**n
    return fitness / maximum


def create_patches(rows, cols, fitness):
    patches = np.zeros(shape=(rows, cols), dtype='object')
    for i in range(rows):
        patch = np.zeros(cols, dtype='object')
        for j in range(cols):
            if math.isnan(fitness[i][j]):
                patch[j] = Patch(float('nan'), fitness[i][j])
            else:
                patch[j] = Patch(0, fitness[i][j])
        patches[i] = patch
    return patches


def fill_patches_random(patches, agents_quantity, max_quantity):
    # np.random.seed(13)
    for _ in range(agents_quantity):
        while True:
            row = patches[np.random.randint(patches.shape[0])]
            patch = np.random.choice(row)
            if not math.isnan(patch.quantity):
                break
        patch.quantity = random.randrange(1, max_quantity)
    return patches


def create_map(typo, file_name, transform, iteration, fig_name, user_path, max_quantity):
    typo = np.loadtxt(user_path+'/'+typo)
    dataset = rasterio.open(user_path+'/'+file_name, 'w', driver='AAIGrid', height=typo.shape[0], width=typo.shape[1], count=1, dtype=str(
        typo.dtype), crs='+proj=latlong', transform=transform)
    dataset.write(typo, 1)
    dataset.close()
    raster = rasterio.open(user_path+'/'+file_name)
    fig, (axr) = pyplot.subplots()
    ax = pyplot.gca()
    ax.set_facecolor('cornflowerblue')

    # pyplot.imshow(raster.read(1), transform=None, adjust='linear', cmap='BuPu', vmax=max_quantity)
    rasterio.plot.show(raster, with_bounds=True, contour=False, contour_label_kws=None, ax=axr,
                       title='iteration: '+str(iteration), transform=None, adjust='linear', cmap='jet', vmax=max_quantity)
    pyplot.savefig(user_path+'/maps/'+str(fig_name)+'_'+str(iteration))
    pyplot.clf()


def create_distribution_file(patches, specie_name, iteration, max_quantity, user_path):
    data = np.zeros((patches.shape[0], patches.shape[1]))
    for i, row in enumerate(patches):
        for j, patch in enumerate(row):
            data[i][j] = patch.quantity

    np.savetxt(user_path+'/'+specie_name+str(iteration)+'.txt', data)
    return data / max_quantity


def create_first_distribution(patches, max_quantity):
    data = np.zeros((patches.shape[0], patches.shape[1]))
    for i, row in enumerate(patches):
        for j, patch in enumerate(row):
            data[i][j] = patch.quantity

    return data / max_quantity


def run_weighted(patches, previous_patches, birth_rate, death_rate, spread_rate, max_quantity):
    for i, row in enumerate(patches):
        for j, patch in enumerate(row):
            if not math.isnan(patch.fitness):
                previous_neighbours = patch.find_neighbours(
                    previous_patches, i, j)
                current_neighbours = patch.find_neighbours(patches, i, j)
                patch.reproduce_weighted(previous_patches[i][j], birth_rate, death_rate,
                                         spread_rate, max_quantity, previous_neighbours, current_neighbours)


def run(patches, previous_patches, birth_rate, death_rate, spread_rate, max_quantity):
    for i, row in enumerate(patches):
        for j, patch in enumerate(row):
            if not math.isnan(patch.fitness):
                previous_neighbours = patch.find_neighbours(
                    previous_patches, i, j)
                current_neighbours = patch.find_neighbours(patches, i, j)
                patch.reproduce(previous_patches[i][j], birth_rate, death_rate,
                                spread_rate, max_quantity, previous_neighbours, current_neighbours)


def run_parts(patches, previous_patches, rows, cols, pipe, birth_rate, death_rate, spread_rate, max_quantity, step, top, bottom):
    _, output = pipe
    for k in range(step):
        for i in range(rows):
            for j in range(cols):
                if not math.isnan(patches[i][j].fitness):
                    previous_neighbours = patches[i][j].find_neighbours(
                        previous_patches, i, j)
                    current_neighbours = patches[i][j].find_neighbours(
                        patches, i, j)
                    patches[i][j].reproduce(previous_patches[i][j], birth_rate, death_rate,
                                            spread_rate, max_quantity, previous_neighbours, current_neighbours)
        previous_patches = copy.deepcopy(patches)
    output.send(patches[top:len(patches)-bottom])
    output.close()


def run_direction_weighted(patches, previous_patches, birth_rate, death_rate, spread_rate, max_quantity, direction_map, model):
    for i, row in enumerate(patches):
        for j, patch in enumerate(row):
            if not math.isnan(patch.fitness):
                previous_neighbours = patch.find_neighbours(
                    previous_patches, i, j)
                current_neighbours = patch.find_neighbours(patches, i, j)
                patch.reproduce_direction_weighted(previous_patches[i][j], birth_rate, death_rate,
                                                   spread_rate, max_quantity, previous_neighbours, current_neighbours, direction_map[i][j], model)


def create_suitability_map(fitness, file_name, transform):
    dataset = rasterio.open(file_name, 'w', driver='AAIGrid', height=fitness.shape[0], width=fitness.shape[1], count=1, dtype=str(fitness.dtype), crs='+proj=latlong', transform=transform)
    dataset.write(fitness, 1)
    dataset.close()

def plot_suitability(data, user_path):
    ax = pyplot.subplot()
    im = ax.imshow(data, cmap='jet', vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad= 0.05)
    pyplot.colorbar(im, cax=cax)
    pyplot.savefig(user_path+'/suitability_map')
    pyplot.clf()

    