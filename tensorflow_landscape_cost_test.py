import os
import numpy as np
import tensorflow as tf

import math


def visualize_landscape_cost(
    sess,
    #### OLD STUFF HERE ####
    cost,
    errors_placeholder,
    # output_chain_Train,
    freeEnergy_placeholder,
    freeEnergy,
    #### NOW NEW STUFF ####
    getFreeEnergy,
    Nqubits,
    errors_prediction, ## NOTE: GET RID OF LATER
    # output_chain_Train,
    log_probs_tensor,
    # log_probs, ## NOTE: GET RID OF LATER
    #### END NEW STUFF ####
    T,
    num_samples,
    max_range,
    num_points,
    deltas,
    etas,
    fname,
):
    """
    sess:           tensorflow session thing
                    tensorflow session thing
    cost:           tensorflow cost thing
                    tensorflow cost thing

    errors_placeholder, output_chain_Train, freeEnergy_placeholder, freeEnergy:
    THESE ARE TENSORFLOW SHIT

    T:              float
                    the current temperature
    num_samples:    int
                    number of autoregressive samples used to calculate F
    max_range:      float
                    the maximum value that alpha or beta take
    num_points:     int
                    the number of points between -alpha/-beta and +alpha/+beta
    delta:          dictof (int torch.Tensor)
                    dictionary of (parameter_number, parameters tensor) for delta
    eta:            dictof (int torch.Tensor)
                    dictionary of (parameter_number, parameters tensor) for eta
    fname:          str
                    name under which to save the file

    returns:        None
                    saves file of free energy values over the range
    """


    s_alphas = [0, 0.5]
    s_betas = [0, 0.5]

    # Defining where to hold F values and alpha/beta
    F_vals = np.zeros((num_points, num_points))

    alpha_vals = np.linspace(-max_range, max_range, num_points)
    beta_vals = np.linspace(-max_range, max_range, num_points)

    # Create dictionary to store minimum parameters and their dimensions, and
    # create placeholders for variables and assignment operations plus dicts when to use
    curr_params = {}
    placeholders_dict = {}
    assignments_dict = {}
    vars = tf.trainable_variables()
    for var_num, variable in enumerate(tf.trainable_variables()):
        value = sess.run(variable.name)
        curr_params[var_num] = value
        placeholders_dict[var_num] = tf.compat.v1.placeholder(
            dtype=tf.float64, shape=variable.get_shape()
        )
        assignments_dict[var_num] = tf.compat.v1.assign(
            variable, placeholders_dict[var_num]
        )

    for i in np.arange(num_points):  # alpha
        alpha = alpha_vals[i]
        for j in np.arange(num_points):  # beta
            beta = beta_vals[j]
            # print("alpha = {0}, beta = {1}".format(alpha, beta))

            output_chain_Train = sess.run(errors_prediction)

            for var_num, variable in enumerate(tf.trainable_variables()):
                new_val = (
                    curr_params[var_num]
                    + alpha * deltas[var_num]
                    + beta * etas[var_num]
                )
                sess.run(
                    assignments_dict[var_num],
                    feed_dict={placeholders_dict[var_num]: new_val},
                )

            for s_alpha in s_alphas:
                for s_beta in s_betas:
                    if math.isclose(abs(alpha), s_alpha) and math.isclose(abs(beta), s_beta):
                        samples = []
                        for sample in output_chain_Train:
                            samples.append(sample)
                        samples = np.array(samples)
                        H_per_sample = np.zeros(samples.shape[0])
                        L = 3
                        for first_index in range(L ** 2):
                            # Determine indices for summing using rhombic lattice (see repo image)
                            second_index = (first_index + 1) % L + L * (first_index // L)
                            third_index = (first_index + L) % L ** 2
                            # H_per_sample += (  # slice 0 to get binary and get H with Ising spins
                                # (2 * samples[:, first_index] - 1)
                                # * (2 * samples[:, second_index] - 1)
                                # * (2 * samples[:, third_index] - 1) / 2
                            # )
                            first_part = 2 * samples[:, first_index] - 1
                            second_part = 2 * samples[:, second_index] - 1
                            third_part = 2 * samples[:, third_index] - 1
                            H_per_sample += (first_part * second_part * third_part / 2)

                        print(samples)
                        print(H_per_sample)
                        print(sum(H_per_sample) / len(H_per_sample))

                        # log_probs = sess.run(log_probs)
                        # freeEnergy = sess.run(freeEnergy)

                        # print("log_probs is ", tf.reduce_mean(log_probs))
                        # print("freeEnergy is ", tf.reduce_mean(tf.stop_gradient(freeEnergy)))
                        # print("combined term is ", tf.reduce_mean(tf.multiply(log_probs, tf.stop_gradient(freeEnergy))))


            curr_loss = sess.run(
                cost,
                feed_dict={
                    errors_placeholder: output_chain_Train,
                    freeEnergy_placeholder: freeEnergy,
                },
            )
            print("alpha = {0}, beta = {1}, loss = {2}".format(alpha, beta, curr_loss))
            F_vals[i, j] = curr_loss

    for var_num, variable in enumerate(tf.trainable_variables()):
        new_val = curr_params[var_num]
        sess.run(tf.compat.v1.assign(variable, new_val))

    np.savetxt(fname, F_vals)
