import sys
import tensorflow as tf

from modified_NM_jeremy_code import annealingSurfaceCode

# Define physical parameters
L = 4  # Length of lattice along one side

# Define training parameters
N_warmup = 1000  # Number of warmup steps
N_anneal = 10000  # Number of annealing steps
N_train = 5  # Number of training steps

N_hidden = 40  # Number of hidden/memory units
N_samples = 100  # Number of samples
minibatch_size = 10

seed = 0  # Seed for RNG
max_change = 1.0

T0 = 1

# I guess just call it?
annealingSurfaceCode(
    Nwarm=N_warmup,
    Nsteps=N_anneal,
    Nequilibrium=N_train,
    Nqubits=L,
    Nunits=N_hidden,
    numsamples=N_samples,
    minibatch_size=minibatch_size,
    seed=seed,
    max_change=max_change,
    # T0=1,
    T0=T0,
)
