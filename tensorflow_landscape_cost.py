import os
import numpy as np
import tensorflow as tf


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
    # errors_prediction,
    output_chain_Train,
    log_probs_tensor,
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
