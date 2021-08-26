#%tensorflow_version 1.x
import tensorflow as tf
import numpy as np
import random
from random import choices

import time


def ErrorGeneration(pErrorX, N, numsamples):
    """
    Purpose:
        - Generates the configurations of X errors on the lattice,
        given the number of samples and the error probability.
        Note that the error is assumed to be independent on
        each site.

    Arguments:
        - pErrorX (float between 0 and 1): Physical error rate on the lattice.
        - N (odd integer): The number of qubits along one side of the lattice.
        - numsamples (integer): Number of error configurations to generate.

    Output:
        - xErrors (an array of size (numsamples, N*N) ): A vector of error configurations.
    """

    values = [1, 0]  # The two possible values for our probability distribution.
    probability = [pErrorX, 1 - pErrorX]

    xErrors = [0] * Nsamples

    for i in range(Nsamples):
        xErrors[i] = choices(
            values, probability, k=N ** 2
        )  # For each i, gives a list of N**2 values chosen from the probability distribution.

    return xErrors


def ImportantZSites(N):
    """
    Purpose:
      - Generates the list of important lattice sites in order to compute the Z syndrome later on.

    Arguments:
      - N (odd integer): The number of qubits along one side of the square lattice.

    Output:
      - BoundaryListZ (list): A list of important sites along the boundary of the lattice.
      - BulkListZ (list): A list of important sites in the bulk of the lattice.
      - FullListZ (list): BoundaryListZ + BulkListZ, sorted numerically.
    """

    BoundaryP = (N - 1) // 2  # Number of boundary plaquettes per side
    BulkP = (N - 1) ** 2  # Number of bulk plaquettes

    BoundaryListZ = []
    BulkListZ = []

    # The following loop creates a list of the important spins in the bulk for the RNN.

    for i in range(N, N ** 2):  # Skips the first row to generate the important sites.
        if (i % 2 != 0) and (i % N != 0):
            BulkListZ.append(i)

    # The next loop does the same, but for the boundaries.

    for i in range(BoundaryP):
        BoundaryListZ.append(1 + 2 * i)  # The top boundary
        BoundaryListZ.append((N ** 2 - 1) - 2 * i)  # The bottom boundary

    # Sort the boundary lists so we can locate which plaquette we're on

    BoundaryListZ.sort()

    FullListZ = BoundaryListZ + BulkListZ
    FullListZ.sort()

    return BoundaryListZ, BulkListZ, FullListZ


def ImportantXSites(N):
    """
    Purpose:
      - Generates the list of important lattice sites in order to compute the X syndrome later on.

    Arguments:
      - N (odd integer): The number of qubits along one side of the square lattice.

    Output:
      - BoundaryListX (list): A list of important sites along the boundary of the lattice.
      - BulkListX (list): A list of important sites in the bulk of the lattice.
      - FullListX (list): BoundaryListX + BulkListX, sorted numerically.
    """

    BoundaryP = (N - 1) // 2  # Number of boundary plaquettes per side
    BulkP = (N - 1) ** 2  # Number of bulk plaquettes

    BoundaryListX = []
    BulkListX = []

    # The following loop creates a list of the important spins in the bulk for the RNN.

    for i in range(N, N ** 2):  # Skips the first row to generate the important sites.
        if (i % 2 == 0) and (i % N != 0):
            BulkListX.append(i)

    # The next loop does the same, but for the boundaries.

    for i in range(BoundaryP):
        BoundaryListX.append(2 * N * (i + 1) - 1)  # The right boundary
        BoundaryListX.append(2 * N * (i + 1))  # The left boundary

    # Sort the boundary lists so we can locate which plaquette we're on

    BoundaryListX.sort()

    FullListX = BoundaryListX + BulkListX
    FullListX.sort()

    return BoundaryListX, BulkListX, FullListX


def SnakeImportantXSites(N):
    """
    Purpose:
      - Generates the list of important lattice sites in order to compute the X syndrome later on.
      - This is the "Snake" version, which will store the bottom-right sites on the even rows, and
        the bottom-left sites on the odd rows.
      - Example for a 5x5 lattice:
          [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

          As a lattice:
          [00,01,02,03,04]
          [05,06,07,08,09]
          [10,11,12,13,14]
          [15,16,17,18,19]
          [20,21,22,23,24]

          Becomes:
          BoundaryListX: [09,10,19,20]
          BulkListX:     [07,05,12,14,17,15,22,24]
          FullListX:     [05,07,09,10,12,14,15,17,19,20,22,24]

    Arguments:
      - N (odd integer): The number of qubits along one side of the square lattice.

    Output:
      - BoundaryListX (list): A list of important sites along the boundary of the lattice.
      - BulkListX (list): A list of important sites in the bulk of the lattice.
      - FullListX (list): BoundaryListX + BulkListX, sorted numerically.
    """

    BoundaryP = (N - 1) // 2  # Number of boundary plaquettes per side
    BulkP = (N - 1) ** 2  # Number of bulk plaquettes

    BoundaryListX = []
    BulkListX = []

    # The following loop creates a list of the important spins in the bulk for the RNN.

    k = 1
    for i in range(N, N ** 2):  # Skips the first row to generate the important sites.
        if i % N == 0 and i != N:
            k += 1  # Flips the site that gets saved from bottom-right to bottom-left.

        if k % 2 == 1:  # Saving bottom-left sites
            if (i % 2 == 1) and ((i + 1) % N != 0):
                BulkListX.append(i)
        else:  # Saving bottom-right sites
            if (i % 2 == 0) and (i % N != 0):
                BulkListX.append(i)

    # The next loop does the same, but for the boundaries.

    for i in range(BoundaryP):
        BoundaryListX.append(2 * N * (i + 1) - 1)  # The right boundary
        BoundaryListX.append(2 * N * (i + 1))  # The left boundary

    # Sort the boundary lists so we can locate which plaquette we're on

    BoundaryListX.sort()
    BulkListX.sort()

    FullListX = BoundaryListX + BulkListX
    FullListX.sort()

    return BoundaryListX, BulkListX, FullListX


def SyndromeZ(errors, BoundaryListZ, BulkListZ):
    """
    Purpose:
      - Computes the entire Z syndrome for a given error configuration.

    Arguments:
      - errors (list of size N**2): Configuration of Z errors.
      - BoundaryListZ (list): Bottom-right sites along the boundary to check for syndromes.
      - BulkListZ (list): Bottom-right sites along the bulk to check for syndromes.

    Output:
      - syndrome (list of size (1/2) * (N**2 - 1) ): Syndrome for a given error configuration.
    """

    N = len(errors)
    syndrome = []  # This is the syndrome vector for Z errors
    for i in BoundaryListZ[: len(BoundaryListZ) // 2]:
        syndrome.append((errors[i] + errors[i - 1]) % 2)

    for i in BulkListZ:
        syndrome.append(
            (errors[i] + errors[i - 1] + errors[i - 1 - N] + errors[i - N]) % 2
        )

    for i in BoundaryListZ[(len(BoundaryListZ) // 2) :]:
        syndrome.append((errors[i] + errors[i - 1]) % 2)

    return syndrome


def OldSyndromeX(errors, N):
    """
    Purpose:
      - Computes the entire X syndrome for a given error configuration.
      Note that this is the syndrome caused by X errors, so they show up
      on the Z plaquettes.
      - Note: This is an older version that computes the syndrome individually.

    Arguments:
      - errors (list of size N**2): Configuration of X (spin flip) errors.
      - N (odd integer): The number of qubits along one side of the square lattice.

    Output:
      - syndrome (list of size (1/2) * (N**2 - 1) ): Syndrome for a given error configuration.
    """

    BoundaryListX, BulkListX, FullListX = ImportantXSites(N)

    syndrome = []  # This is the syndrome vector for X errors

    for site in FullListX:
        if site in BoundaryListX:
            syndrome.append((errors[site] + errors[site - N]) % 2)
        else:
            syndrome.append(
                (
                    errors[site]
                    + errors[site - 1]
                    + errors[site - 1 - N]
                    + errors[site - N]
                )
                % 2
            )

    return syndrome


def SyndromeX(
    sess,
    Errors,
    base_log_probs,
    new_samples_placeholder,
    new_log_probs_tensor,
    errors,
    N,
    B,
    BoundaryListX,
    BulkListX,
    FullListX,
):
    """
    Purpose:
      - Computes the entire X syndrome for a given error configuration.
      This is for annealing, so you need to feed in the indices of the important sites.
      Note that this is the syndrome caused by X errors, so they show up
      on the Z plaquettes. This is the "new" function to incorporate
      parallel calculation.

    Arguments:
      - Errors is the tensorflow RNN
      - errors (list of size [numsamples, N**2]): Configuration of X (spin flip) errors.
      - N (odd integer): The number of qubits along one side of the square lattice.
      - BoundaryListX (list): A list of important sites along the boundary of the lattice.
      - BulkListX (list): A list of important sites in the bulk of the lattice.
      - FullListX (list): BoundaryListX + BulkListX, sorted numerically.

    Output:
      # - syndrome (array of size [numsamples, (1/2) * (N**2 - 1)] ): Syndrome for a given error configuration.
      - HACK: H_per_sample (array of size [numsamples] ): Energy for each sample
    """
    L = N
    N = L ** 2
    num_samples = errors.shape[0]
    H_per_sample = np.zeros(num_samples)

    #####################
    ### OFF-DIAG HERE ###
    #####################

    time1 = time.time()

    flipped_spins = np.expand_dims(errors, axis=2)  # spin-flip configs along new axis
    flipped_spins = np.repeat(flipped_spins, N, axis=2)
    ref_spins = np.copy(flipped_spins)
    for i in range(N):  # flip spins
        flipped_spins[:, i, i][ref_spins[:, i, i] == 1] = 0
        flipped_spins[:, i, i][ref_spins[:, i, i] == 0] = 1

    for i in range(N):  # calculate probs one at a time
        log_probs = sess.run(
            new_log_probs_tensor,
            feed_dict={new_samples_placeholder: flipped_spins[:, :, i]},
        )

        # print(log_probs.shape)
        # print(base_log_probs.shape)
        # H_per_sample += -B * np.sum(
            # np.exp(0.5 * log_probs - 0.5 * base_log_probs), axis=0
        # )
        H_per_sample += -B * np.exp(0.5 * log_probs - 0.5 * base_log_probs)
        # print("off-diag contrib is ", -B * np.exp(0.5 * log_probs - 0.5 * base_log_probs))

        # print("off diagonal part is", log_probs)
        # print("diagonal part is ", base_log_probs)

        # print("H_per_sample is ", H_per_sample)


    time2 = time.time()

    #####################
    ### OFF-DIAG ENDS ###
    #####################

    for first_index in range(N):
        # Determine indices for summing using rhombic lattice (see repo image)
        second_index = (first_index + 1) % L + L * (first_index // L)
        third_index = (first_index + L) % L ** 2
        H_per_sample += (  # slice 0 to get binary and get H with Ising spins
            (2 * errors[:, first_index] - 1)
            * (2 * errors[:, second_index] - 1)
            * (2 * errors[:, third_index] - 1)
        ) / 2
        # print("diag contrib is ", (2 * errors[:, first_index] - 1) * (2 * errors[:, second_index] - 1) * (2 * errors[:, third_index] - 1) / 2)

    time3 = time.time()

    print("off-diag time is ", time2 - time1)
    print("diag time is ", time3 - time2)

    return H_per_sample


def ParityCheck(site, samples, syndromes, N, BoundaryListX, FullListX):
    """
    Purpose:
      - Given a lattice site in FullListX, add up the errors at the other sites and compare to the syndrome.

    What's happening:
      - The code takes in the samples and the syndromes, then computes the parity at a given site in parallel
      by adding the previous errors on a plaquette with the syndrome at that site, and computing the result
      modulo 2.

    Arguments:
      - site (integer): A site in FullListX.
      - samples (tensor of shape (numsamples, N**2)): A list of the error configurations on the lattice.
      - syndromes (list of shape (numsamples, (N**2 - 1) / 2): A list of the syndromes.
      - N (odd integer): Number of qubits along one side of the lattice.
      - BoundaryListX (list): A list of boundary sites for the lattice.
      - FullListX (list): BoundaryListX + FullListX.

    Output:
      - A tensor of shape (numsamples) with elements 0 or 1.
    """

    # samplesTensor = tf.cast(samples, dtype = tf.int64)

    if site in BoundaryListX:
        return (
            samples[(site - N)] + tf.cast(syndromes, tf.int64)[:, FullListX.index(site)]
        ) % 2

    else:
        return (
            samples[(site - 1)]
            + samples[(site - 1 - N)]
            + samples[(site - N)]
            + tf.cast(syndromes, tf.int64)[:, FullListX.index(site)]
        ) % 2


def SnakeParityCheck(
    site, yCoordinate, samples, syndromes, N, BoundaryListX, FullListX
):
    """
    Purpose:
      - Given a lattice site in FullListX, add up the errors at the other sites and compare to the syndrome.
      - This is the "Snake" version of the function, since we need to take in the row of the lattice.

    What's happening:
      - The code takes in the samples and the syndromes, then computes the parity at a given site in parallel
      by adding the previous errors on a plaquette with the syndrome at that site, and computing the result
      modulo 2.

    Arguments:
      - site (integer): A site in FullListX.
      - samples (tensor of shape (numsamples, N**2)): A list of the error configurations on the lattice.
      - syndromes (list of shape (numsamples, (N**2 - 1) / 2): A list of the syndromes.
      - N (odd integer): Number of qubits along one side of the lattice.
      - BoundaryListX (list): A list of boundary sites for the lattice. This is generated using SnakeImportantXSites.
      - FullListX (list): BoundaryListX + FullListX.

    Output:
      - A tensor of shape (numsamples) with elements 0 or 1.
    """

    # samplesTensor = tf.cast(samples, dtype = tf.int64)

    if site in BoundaryListX:
        return (
            samples[(site - N)] + tf.cast(syndromes, tf.int64)[:, FullListX.index(site)]
        ) % 2

    else:
        if yCoordinate % 2 == 0:
            return (
                samples[(site - 1)]
                + samples[(site - 1 - N)]
                + samples[(site - N)]
                + tf.cast(syndromes, tf.int64)[:, FullListX.index(site)]
            ) % 2
        else:
            return (
                samples[(site + 1)]
                + samples[(site + 1 - N)]
                + samples[(site - N)]
                + tf.cast(syndromes, tf.int64)[:, FullListX.index(site)]
            ) % 2


def PlaquetteParityAtSite(site, N, errors, BulkListZ, BoundaryListZ):
    """
    Purpose:
      - Take in the site, the size of the lattice, and the errors,
        and output the sum of the three (bulk) or one (boundary) sites
        on the plaquette. This tells us if we need to flip the error
        on the final site.

    Arguments:
      - site (integer): Index on the lattice, and part of FullListZ.
      - N (odd integer): Number of qubits along one edge of the lattice.
      - errors (array): Array of errors on the lattice.
      - BulkListZ (list): Bottom-right sites along the bulk to check syndromes.
      - BoundaryListZ (list): Bottom-right sites along the boundary to check for syndromes.

    Output:
      - Plaquette parity (integer modulo 2): The sum of the previous spins for a plaquette, telling us if we need to flip the last one.
    """

    if site in BoundaryListZ:
        return errors[site - 1]

    if site in BulkListZ:
        return (errors[site - 1] + errors[site - N] + errors[site - 1 - N]) % 2


def WilsonLoopH(errors, N, numsamples):
    """
    Purpose:
      - Computes the horizontal Wilson loop to check for logical failure due to X errors.

    Arguments:
      - errors (tensor of shape (numsamples, N**2)): list of X errors on the lattice.
      - N (odd integer): number of qubits along one side of the lattice.
      - numsamples (integer): Number of samples for the errors and recovery chains.

    Ouput:
      - list of float between 0 (no logical failure) or 1 (logical failure).
    """

    # return tf.reduce_mean((tf.reduce_sum( tf.reshape(total, [numsamples, N, N]), axis = 2 ) % 2), axis = 1) # Sums across the rows and takes the average mod 2.
    return np.mean(
        (np.sum(np.reshape(errors, [numsamples, N, N]), axis=2) % 2), axis=1
    )  # Sums across the rows and takes the average mod 2.
