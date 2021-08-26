#%tensorflow_version 1.x
import tensorflow as tf
import numpy as np
import os
import time
import math
import random
from random import choices

from modified_MDRNNfunction import RNNfunction
from modified_HelperFunctionsandDataGeneration import *

from tensorflow_landscape_F import visualize_landscape_F
# from tensorflow_landscape_cost import visualize_landscape_cost
from tensorflow_landscape_cost_test import visualize_landscape_cost


###################################################
########## DEFINE VISUALIZATION STUFF HERE ########
###################################################

N_POINTS = 21


def getFreeEnergy(
    sess,
    Errors,
    Nqubits,
    samples,
    log_probs,
    # old_log_probs_tensor,
    # old_errors_placeholder,
    new_samples_placeholder,
    new_log_probs_tensor,
    T,
    BoundaryListX,
    BulkListX,
    FullListX,
):
    """
    - Purpose: Get the variational free energy for a bunch of samples.
    - Inputs:
        - Nqubits (odd int): Number of qubits along one side of the lattice.
        - samples (tensor of size (numsamples, Nqubits**2)): The errors on the lattice.
        - log_probs (tensor): The log probabilities from the RNN.
        - T (float): The temperature of the annealing algorithm.
        - BoundaryListX (list): The indices for the boundary sites to calculate the syndrome.
        - BulkListX (list): The indices for the bulk sites to calculate the syndrome.
        - FullListX (list): The indices for the bulk and boundary sites to calculate the syndrome.
    - Outputs:
        - freeEnergy: Tensor of shape (numsamples,) corresponding to the free energy for the variational neural annealing algorithm.
    """

    # Calculation of the energy
    energy = SyndromeX(
        sess,
        Errors,
        log_probs,
        new_samples_placeholder,
        new_log_probs_tensor,
        samples,
        Nqubits,
        T,
        BoundaryListX,
        BulkListX,
        FullListX,
    )

    # Calculation of the free energy using log_probs
    # freeEnergy = energy + T * log_probs

    # return freeEnergy
    return energy


def annealingSurfaceCode(
    Nwarm=100,
    Nsteps=10 ** 3,
    Nequilibrium=5,
    Nqubits=5,
    T0=1.0,
    Nunits=5,
    Nlayers=1,
    numsamples=100,
    minibatch_size=10,
    activation=tf.nn.elu,
    learningrate=1e-3,
    learningdecay=0.0,
    learningdecaytime=50,
    seed=111,
    max_change=1,
    savechains=False,
):
    """
    Purpose:
        - Use variational neural annealing to bring the surface code back to its ground state.

    Arguments:
        - Nwarm (integer): Number of warm up steps before lowering the temperature.
        - Nsteps (integer): Number of annealing steps in the RNN algorithm.
        - Nequilibrium (integer): Number of training steps at a given temperature.
        - T0 (float): Starting temperature for the annealing process.
        - Nqubits (odd integer): Number of qubits along one side of the lattice.
        - Nunits (integer): Size of hidden layer.
        - Nlayers (integer): Number of hidden layers (here, we just have 1).
        - numsamples (integer): Number of samples.
        - minibatch_size (integer): Number of training samples to use to update weights. Must divide numsamples.
        - activation (string): Activation function used.
        - learningrate (float): Learning rate for the training.
        - learningdecay (float): Power that the learning rate decays by.
        - learningdecaytime (integer): Number of steps until the learning rate asymptotes.
        - seed (integer): Seed for consistent stochastic processes.
        - saveinterval (integer): Frequency to save the model and results.
        - savechains (Boolean): Save the test recovery chains for plotting purposes.

    Output: None
    """

    # Define directories to store results
    results_path = "results/NM_L{0}_results".format(Nqubits)
    savelocation = "{0}/L{1}_T{2}_nh{3}_lr{4}_Ns{5}_Nw{6}_Nt{7}_Na{8}_seed{9}".format(
        results_path,
        Nqubits,
        T0,
        Nunits,
        learningrate,
        numsamples,
        Nwarm,
        Nequilibrium,
        Nsteps,
        seed,
    )

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(savelocation):
        os.makedirs(savelocation)

    # Seed TensorFlow

    tf.reset_default_graph()
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Define array to store relevant values
    # dimension is total steps x 4 (step, T, cost, free energy, energy)
    save_data_array = np.zeros((Nwarm + Nequilibrium * Nsteps + 1, 5))

    # Define when to make loss landscape visualizations
    mid_anneal_num_viz = 5

    # Start up the RNN

    # units = [Nunits] * Nlayers #For now, we only support Nlayers = 1, which I think would be enough, but let me know if you would like to test your code more layers.
    input_dim = 2  # Errors can be either on or off.
    initialNumSamples = 20  # Just to start the RNN.

    Errors = RNNfunction(
        Nqubits, num_units=Nunits, activation=activation, seed=seed
    )  # This creates the RNN blocks
    sampling = Errors.sample(initialNumSamples, input_dim)

    with Errors.graph.as_default():
        samples_placeholder = tf.placeholder(
            dtype=tf.int32, shape=[initialNumSamples, Nqubits ** 2]
        )
        global_step = tf.Variable(0, trainable=False)
        learningrate_placeholder = tf.placeholder(dtype=tf.float64, shape=[])
        learningrate_withexpdecay = tf.train.exponential_decay(
            learningrate_placeholder,
            global_step=global_step,
            decay_steps=100,
            decay_rate=1.0,
            staircase=True,
        )  # Here, there is no exp decay as decay_rate = 1
        log_probabilities = Errors.log_probability(samples_placeholder, input_dim)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learningrate_withexpdecay
        )  # This uses the AdamOptimizer, though I could change it later.
        init = tf.global_variables_initializer()

    # Starting the session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(graph=Errors.graph, config=config)
    sess.run(init)

    # Naming the training session

    print(
        "N = "
        + str(Nqubits)
        + ", numsamples = "
        + str(numsamples)
        + ", Nunits = "
        + str(Nunits)
        + ", lr = "
        + str(learningrate)
        + ", lr decay = "
        + str(learningdecay)
        + ", lr decay time = "
        + str(learningdecaytime)
        + ", minibatch_size = "
        + str(minibatch_size)
        + ", activation = "
        + str(activation)
        + ", MDRNN: On, "
        + "Warm up = "
        + str(Nwarm)
        + ", Annealing steps = "
        + str(Nsteps)
    )

    # Counting the number of parameters

    with Errors.graph.as_default():
        variables_names = [v.name for v in tf.trainable_variables()]

        sum = 0
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            v1 = tf.reshape(v, [-1])
            # print(k,v1.shape)
            sum += v1.shape[0]
        print("The number of parameters of the RNN function is {0}".format(sum))
        print("\n")

    # Learning rate
    lr = np.float64(learningrate)

    with tf.variable_scope(Errors.scope, reuse=tf.AUTO_REUSE):
        with Errors.graph.as_default():
            errors_placeholder = tf.placeholder(
                dtype=tf.int32, shape=[None, Nqubits ** 2]
            )  # Define a placeholder for the errors
            freeEnergy_placeholder = tf.placeholder(dtype=tf.float64, shape=[None])
            log_probs_ = Errors.log_probability(
                errors_placeholder, 2
            )  # You need to feed in the syndromes

            # The cost function
            cost = 2 * tf.reduce_mean(
                tf.multiply(log_probs_, tf.stop_gradient(freeEnergy_placeholder))
            ) - tf.reduce_mean(
                tf.stop_gradient(freeEnergy_placeholder)
            ) * tf.reduce_mean(
                0.5 * log_probs_
            )

            print("Created cost function placeholder")

            # Calculate Gradients---------------

            gradients, variables = zip(*optimizer.compute_gradients(cost))

            ## Gradient clipping
            # gradients, _ = tf.clip_by_global_norm(gradients, 1)

            # End calculate Gradients---------------

            optstep = optimizer.apply_gradients(
                zip(gradients, variables), global_step=global_step
            )
            sess.run(
                tf.variables_initializer(optimizer.variables()),
                feed_dict={learningrate_placeholder: lr},
            )
            saver = tf.train.Saver()

            print("Finished the optimizer")

            ### Loading models

            fileNameModel = "{0}/{1}".format(savelocation, "Model")
            # fileNameModelLast = savelocation + "LastModel.ckpt"
            fileNameTest = "{0}/{1}".format(savelocation, "pFailTest.npy")
            fileNameTrain = "{0}/{1}".format(savelocation, "pFailTrain.npy")
            fileNameCost = "{0}/{1}".format(savelocation, "cost.npy")

            energy = []
            temperatures = []
            free_energy = []
            modelCost = []

    # ----------------------------------------------------------------

    ### Training the RNN function
    with tf.variable_scope(Errors.scope, reuse=tf.AUTO_REUSE):
        with Errors.graph.as_default():

            # errors_placeholderTrain = Errors.sample(numsamples = numsamples, inputdim = 2, syndromes = xSyndromes)
            # errors_placeholderTest = Errors.sample(numsamples = numtestsamples, inputdim = 2, syndromes = xSyndromesTest)

            print("Starting the training")
            BoundaryListX, BulkListX, FullListX = ImportantXSites(Nqubits)

            permut = np.arange(numsamples)

            start = time.time()
            saveinterval = 1

            # initialStep = len(wilsonLoop) * saveinterval

            # if len(wilsonLoop) == Nsteps:
            # print("Already completed!")
            # initialStep == Nsteps  # This will make sure the code doesn't continue.

            ## Allocate tensors while not making them too big

            maxbatchsize = 50000  # Prevents memory overload issues. Make numtestsamples > maxbatchsize

            # Training samples
            # print("Creating the errors placeholder")
            if numsamples > maxbatchsize:
                errors_prediction = Errors.sample(numsamples=maxbatchsize, inputdim=2)
            else:
                errors_prediction = Errors.sample(numsamples=numsamples, inputdim=2)

            output_chain_Train = np.zeros((numsamples, Nqubits ** 2), np.int32)
            freeEnergy = np.zeros((numsamples), np.int32)

            # print("Creating the log probs placeholder")
            log_probs_tensor = Errors.log_probability(errors_placeholder, inputdim=2)
            log_probs = np.zeros((Nqubits ** 2) * numsamples, dtype=np.float64)

            print("Starting warm up")
            T = T0
            c = 0  # Counter for the equlibrium steps

            ##########################################################
            ############# EXPERIMENTING WITH SHIT HERE  ##############
            ##########################################################

            # variables_names = [v.name for v in tf.trainable_variables()]
            # values = sess.run(variables_names)
            # print(values)
            # values = sess.run(tf.assign(values, values + 99))
            # print(values)

            # print(Errors.Ul)
            # print(sess.run(Errors.Ul.name))

            # for var_num, variable in enumerate(tf.trainable_variables()):
            # value = sess.run(variable.name)
            # print("init value is")
            # print(value)
            # var_shape = variable.get_shape()
            # to_add = tf.convert_to_tensor(99 * np.ones(var_shape))
            # sess.run(tf.compat.v1.assign_add(variable, to_add))
            # new_value = sess.run(variable.name)
            # print("new value is")
            # print(new_value)
            # print(value)
            # new_value = value + 99 * np.ones(var_shape)
            # print(new_value)
            # tf.assign(value, tf.convert_to_tensor(new_value))
            # print("can assign as required")
            # # new_value = sess.run(tf.assign(value, value + 99 * np.ones(var_shape)))
            # print(new_value)

            #####################################################
            ############# VISUALIZATION STUFF HERE ##############
            #####################################################

            # Define a delta and eta first
            deltas = {}
            etas = {}
            for var_num, variable in enumerate(tf.trainable_variables()):
                var_shape = variable.get_shape().as_list()
                if len(var_shape) == 1:
                    deltas[var_num] = np.random.randn(var_shape[0])
                    etas[var_num] = np.random.randn(var_shape[0])
                if len(var_shape) == 2:
                    deltas[var_num] = np.random.randn(var_shape[0], var_shape[1])
                    etas[var_num] = np.random.randn(var_shape[0], var_shape[1])

            #######################################################
            ############# UNUSED VISUALIZATION HERE  ##############
            #######################################################

            #####################################################
            ############# VISUALIZATION STUFF ENDS ##############
            #####################################################

            ### Define placeholder and log_probs tensor to use in free energy calculation
            new_samples_placeholder = tf.placeholder(
                dtype=tf.int32, shape=[None, Nqubits ** 2]
            )
            new_log_probs_tensor = Errors.log_probability(
                new_samples_placeholder, inputdim=2
            )

            print("Defined placeholder and log_probs_tensor")

            for step in range(Nwarm + Nequilibrium * Nsteps + 1):
                TotalCost = 0
                if step == Nwarm:
                    print("Done warm up")
                    print("Starting annealing")
                if step > Nwarm - 1:
                    if c % Nequilibrium == 0:
                        T = T0 * (
                            1 - (step - Nwarm) / (Nequilibrium * Nsteps)
                        )  # Linear schedule
                    c += 1

                trainsteps = (
                    numsamples // maxbatchsize
                )  # Need this to be a multiple of maxbatchsize
                if trainsteps > 0:
                    for i in range(trainsteps):
                        cut = slice(i * maxbatchsize, (i + 1) * maxbatchsize)

                        output_chain_Train[cut] = sess.run(errors_prediction)
                        log_probs[cut] = sess.run(
                            log_probs_tensor,
                            feed_dict={errors_placeholder: output_chain_Train[cut]},
                        )

                        freeEnergy[cut] = getFreeEnergy(
                            sess,
                            Errors,
                            Nqubits,
                            output_chain_Train[cut],
                            log_probs[cut],
                            new_samples_placeholder,
                            new_log_probs_tensor,
                            T,
                            BoundaryListX,
                            BulkListX,
                            FullListX,
                        )

                        TotalCost += sess.run(
                            cost,
                            feed_dict={
                                errors_placeholder: output_chain_Train[cut],
                                freeEnergy_placeholder: freeEnergy[cut],
                            },
                        )
                    TotalCost = TotalCost / trainsteps
                else:
                    # print("Getting the output of the RNN")
                    # outtime = time.time()
                    output_chain_Train = sess.run(errors_prediction)
                    # print("Getting the log probs from the RNN")
                    log_probs = sess.run(
                        log_probs_tensor,
                        feed_dict={errors_placeholder: output_chain_Train},
                    )
                    # end = time.time()
                    # freeEnergy = getFreeEnergy(
                    # sess,
                    # Errors,
                    # Nqubits,
                    # output_chain_Train,
                    # log_probs,
                    # new_samples_placeholder,
                    # new_log_probs_tensor,
                    # T,
                    # BoundaryListX,
                    # BulkListX,
                    # FullListX,
                    # )
                    freeEnergy = getFreeEnergy(
                        sess,
                        Errors,
                        Nqubits,
                        output_chain_Train,
                        log_probs,
                        # log_probs_tensor,
                        # errors_placeholder,
                        new_samples_placeholder,
                        new_log_probs_tensor,
                        T,
                        BoundaryListX,
                        BulkListX,
                        FullListX,
                    )
                    # print("Calculating the cost")
                    TotalCost += sess.run(
                        cost,
                        feed_dict={
                            errors_placeholder: output_chain_Train,
                            freeEnergy_placeholder: freeEnergy,
                        },
                    )

                #####################################################
                ############# VISUALIZATION STUFF HERE ##############
                #####################################################

                names_dict = {
                    0: "init",
                    Nwarm: "warmup",
                    Nwarm + Nequilibrium * Nsteps: "completion",
                }

                make_viz_step = (Nequilibrium * Nsteps) // (mid_anneal_num_viz + 1)

                for viz_num in range(1, mid_anneal_num_viz + 1):
                    names_dict[Nwarm + viz_num * make_viz_step] = (
                        "mid_viz{0}_of{1}".format(viz_num, mid_anneal_num_viz)
                    )

                if step in names_dict.keys():
                    fname_F = "{0}/F_landscape_viz_{1}_max{2}_Npoints{3}.txt".format(
                        savelocation, names_dict[step], max_change, N_POINTS
                    )
                    fname_cost = (
                        "{0}/cost_landscape_viz_{1}_max{2}_Npoints{3}.txt".format(
                            savelocation, names_dict[step], max_change, N_POINTS
                        )
                    )
                    visualize_landscape_F(
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
                        errors_prediction,
                        # output_chain_Train,
                        log_probs_tensor,
                        #### VCA SPECIFIC STUFF ####
                        Errors,
                        new_samples_placeholder,
                        new_log_probs_tensor,
                        #### END VCA SPECIFIC STUFF ####
                        #### END NEW STUFF ####
                        T,
                        numsamples,
                        max_change,
                        N_POINTS,
                        deltas,
                        etas,
                        fname_F,
                    )
                    print("making landscape, B = ", T)
                    # visualize_landscape_cost(
                        # sess,
                        # #### OLD STUFF HERE ####
                        # cost,
                        # errors_placeholder,
                        # # output_chain_Train,
                        # freeEnergy_placeholder,
                        # freeEnergy,
                        # #### NOW NEW STUFF ####
                        # getFreeEnergy,
                        # Nqubits,
                        # errors_prediction, ## NOTE: GET RID OF LATER
                        # # output_chain_Train,
                        # log_probs_tensor,
                        # #### END NEW STUFF ####
                        # T,
                        # numsamples,
                        # max_change,
                        # N_POINTS,
                        # deltas,
                        # etas,
                        # fname_cost,
                    # )

                if step == Nwarm or step == Nwarm + Nequilibrium * Nsteps:
                    samples = []
                    for sample in output_chain_Train:
                        samples.append(sample)
                    fname = "{0}/{1}_samples.txt".format(savelocation, names_dict[step])
                    np.savetxt(fname, np.array(samples))
                elif math.isclose(T, 0.1):
                    samples = []
                    for sample in output_chain_Train:
                        samples.append(sample)
                    fname = "{0}/low_B_samples.txt".format(savelocation)
                    np.savetxt(fname, np.array(samples))


                #####################################################
                ############# VISUALIZATION STUFF ENDS ##############
                #####################################################

                modelCost.append(TotalCost)
                free_energy.append(np.mean(freeEnergy))
                energy.append(np.mean(freeEnergy))
                temperatures.append(T)
                if step % 5 == 0:
                    print(
                        "Annealing Step = ",
                        step,
                        " at T = ",
                        T,
                        " Cost = ",
                        TotalCost,
                        " Average Energy = ",
                        energy[-1],
                    )

                HomologyTrain = WilsonLoopH(output_chain_Train, Nqubits, numsamples)
                # wilsonLoop.append(np.mean(HomologyTrain))

                # Optimize in minibatches

                np.random.shuffle(permut)
                # print("Starting optimization")
                # optime = time.time()
                chains_shuffled = output_chain_Train[permut]
                freeEnergy_shuffled = freeEnergy[permut]
                for b in range(0, numsamples, minibatch_size):
                    sess.run(
                        optstep,
                        feed_dict={
                            errors_placeholder: chains_shuffled[
                                b : b + minibatch_size, :
                            ],
                            freeEnergy_placeholder: freeEnergy_shuffled[
                                b : b + minibatch_size
                            ],
                            learningrate_placeholder: lr,
                        },
                    )

                # Save the last step
                if step + 1 == Nwarm + Nsteps:
                    print("Saving the model")
                    saving = time.time()
                    saver.save(sess, fileNameModel, global_step=step + 1)
                    end = time.time()
                    print("Done saving the model in {0} s".format(end - saving))
                    # np.save(
                    # "{0}/{1}".format(savelocation, "wilsonLoopsAnnealing.npy"),
                    # wilsonLoop,
                    # )
                    np.save(
                        "{0}/{1}".format(savelocation, "energyAnnealing.npy"), energy
                    )

            end = time.time()

            print("Training runtime = %s s" % (end - start))

    print("Final sampling")
    with tf.variable_scope(Errors.scope, reuse=tf.AUTO_REUSE):
        with Errors.graph.as_default():

            # samples = 10 ** 6
            # final_chain = np.zeros((samples, Nqubits ** 2), np.int32)
            # # Final samples
            # if samples > maxbatchsize:
            # final_samples = Errors.sample(numsamples=maxbatchsize, inputdim=2)

            BoundaryListX, BulkListX, FullListX = ImportantXSites(Nqubits)
            # finalSyndromes = SyndromeX(
            # final_chain, Nqubits, BoundaryListX, BulkListX, FullListX
            # )
            # finalEnergy = np.sum(finalSyndromes, axis=1)
            # meanEnergy = np.mean(finalEnergy)
            # varianceEnergy = np.var(finalEnergy)
            # np.save("{0}/{1}".format(savelocation, "meanEnergy.npy"), meanEnergy)
            # np.save("{0}/{1}".format(savelocation, "varEnergy.npy"), varianceEnergy)

            # print(
            # "Proportion of Samples with Wilson Loop = 1: ",
            # np.count_nonzero(homologyFinal == 1) / len(homologyFinal),
            # )
            # print(
            # "Proportion of Samples with Wilson Loop = 0: ",
            # np.count_nonzero(homologyFinal == 0) / len(homologyFinal),
            # )

            save_data_array[:, 0] = np.arange(Nwarm + Nequilibrium * Nsteps + 1)
            save_data_array[:, 1] = np.array(temperatures)
            save_data_array[:, 2] = np.array(energy)
            save_data_array[:, 3] = np.array(free_energy)
            save_data_array[:, 4] = np.array(modelCost)
            np.savetxt("{0}/training_info.txt".format(savelocation), save_data_array)
            # np.save(savelocation + "outputWilsonLoops.npy", homologyFinal)
            # np.save(savelocation + "outputChains.npy", final_chain)
            # np.save("{0}/{1}".format(savelocation, "outputEnergy.npy"), finalEnergy)
