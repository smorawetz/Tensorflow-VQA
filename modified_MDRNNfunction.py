#%tensorflow_version 1.x
import tensorflow as tf
import numpy as np
import random
from modified_HelperFunctionsandDataGeneration import ImportantXSites, SnakeImportantXSites
from modified_HelperFunctionsandDataGeneration import ParityCheck, SnakeParityCheck

class RNNfunction(object):
    def __init__(self, systemsize, activation = tf.nn.tanh, num_units = 10, scope = "RNNfunction", seed = 111):
        """
            systemsize:  int, size of the lattice
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
        """

        self.graph = tf.Graph()
        self.scope = scope #Label of the RNN function
        self.N = systemsize #Number of sites on each side
        self.activation = activation #The RNN cell activation function
        self.num_units = num_units
        num_inputs = 2

        #Seeding to get reproducible results
        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        with self.graph.as_default():
            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                tf.set_random_seed(seed)  # tensorflow pseudo-random generator

                # The weight matrix for the hidden state

                # The weight matrix for the recovery chain
                self.Ul = tf.get_variable("Ul", shape = [num_inputs+num_units, num_units], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
                self.Uu = tf.get_variable("Uu", shape = [num_inputs+num_units, num_units], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

                # The bias
                self.b = tf.get_variable("b", shape = [num_units], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
                #self.bg = tf.get_variable("bg", shape = [num_units], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

                #self.Un = tf.get_variable("Un", shape = [num_units, 2*num_units], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

                self.dense_prob = tf.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense', dtype = tf.float64) #Define the Fully-Connected layer followed by a Softmax

    def MDRNNCell(self, hiddenstate, inputs):
        return self.activation( tf.matmul(tf.concat([tf.cast(inputs[0], tf.float64),hiddenstate[0]],1), self.Ul) +  tf.matmul(tf.concat([tf.cast(inputs[1], tf.float64),hiddenstate[1]],1), self.Uu) + self.b)

    def MDGRUCell(self, hiddenstate, inputs):
        h_tilda = self.activation( tf.matmul(tf.concat([tf.cast(inputs[0], tf.float64),hiddenstate[0]],1), self.Ul) +  tf.matmul(tf.concat([tf.cast(inputs[1], tf.float64),hiddenstate[1]],1), self.Uu) + self.b)
        gate = tf.nn.sigmoid( tf.matmul(tf.concat([tf.cast(inputs[0], tf.float64),hiddenstate[0]],1), self.Ulg) +  tf.matmul(tf.concat([tf.cast(inputs[1], tf.float64),hiddenstate[1]],1), self.Ugu) + self.bg)

        h_new = gate*h_tilda + (1. - gate) * tf.matmul(tf.concat(hiddenstate,1), self.Un)

        return h_new

    def get_neighbours(self, inputs, zeroinput, index):
      if index == 0:
        return (zeroinput, zeroinput)
      elif index < self.N:
        return (inputs[index-1], zeroinput)
      else:
        return (inputs[index-1], inputs[index-self.N])

    def get_neighboursSnake(self, inputs, zeroinput, index, yCoordinate):
        """
            - Purpose: To find the neighbours of a given qubit in a snake pattern (as opposed to raster scan).
            - Arguments:
                - inputs (numpy array): The current state of the RNN at the index.
                - zeroinput (numpy array): A bunch of zeros to pad the array given the index.
                - index (integer): Site on the lattice (labeling the sites from left to right).
                - yCoordinate (integer): Which row of the lattice are we on? This tells us which direction to go.
            - Output:
                - Gives the states that influence the computation at the given index.
                - For the snake, this depends on which row you're on, but it's always of the form (left/right state, above state).
                - The left/right distinction depends on if the scan is going from right to left or left to right.
        """
        if index == 0:
            return (zeroinput, zeroinput)
        elif index < self.N:
            return (inputs[index-1], zeroinput)
        elif yCoordinate % 2 == 0:
            if index % self.N == 0: # Check if you're at the first site of a row going left to right.
                return (zeroinput, inputs[index-self.N])
            else:
                return (inputs[index-1], inputs[index-self.N]) # The inputs to the left and from above.
        else:
            if (index+1) % self.N == 0: # Check if you're at the first site of a row going right to left.
                return(zeroinput, inputs[index-self.N])
            else:
                return (inputs[index+1], inputs[index-self.N]) # The inputs to the right and from above.


    def sample(self, numsamples, inputdim):
            """
                Purpose:
                    - Generate samples from a probability distribution parametrized by a recurrent network.

                Arguments:
                    - numsamples (integer): Number of samples to be produced
                    - inputdim (integer): Hilbert space dimension of one error (so, 2).
                Returns:
                    - samples (tf.Tensor of shape (numsamples,systemsize) ): The samples in integer encoding.
            """
            with self.graph.as_default(): #Call the default graph, used if willing to create multiple graphs.
                #samples = []
                #onehotsamples = []
                samples = [0]*self.N**2
                onehotsamples = [0]*self.N**2
                with tf.variable_scope(self.scope, reuse = tf.AUTO_REUSE):
                    zeroinput = tf.zeros([numsamples, inputdim], dtype = tf.float64)
                    # This is like my configuration of current spins, right?

                    self.inputdim = inputdim
                    self.outputdim = self.inputdim
                    self.numsamples = numsamples

                    # Initialize the hidden state
                    zerohiddenstate = tf.zeros([numsamples, self.num_units], dtype = tf.float64)
                    hiddenstates = [0]*self.N**2
                    # Note: The zero state returns a zero-filled tensor with shape = (self.numsamples, num_units)

                    """
                    # Perform the multiplication of the syndrome and the weight matrix
                    static_syndrome = tf.matmul(tf.cast(syndromes,tf.float64), self.V)
                    """

                    """
                    # Get the important lattice sites
                    BoundaryListX, BulkListX, FullListX = ImportantXSites(self.N)
                    """

                    # For the Snake scanning
                    BoundaryListX, BulkListX, FullListX = ImportantXSites(self.N)

                    # Now we do the loop over the whole RNN
                    for Ny in range(self.N):
                      for Nx in range(self.N):
                        site = self.N*Ny + Nx # This makes the scan go from left to right.


                        # if Ny % 2 == 0:
                            # site = self.N*Ny + Nx # This makes the scan go from left to right.
                        # else:
                            # site = self.N*(Ny + 1) - (1 + Nx) # This makes the scan go from right to left.

                        current_states = self.get_neighbours(hiddenstates, zerohiddenstate, site)
                        current_inputs = self.get_neighbours(onehotsamples, zeroinput, site)
                        #print("Nx = ", Nx, "Ny = ", Ny, "site = ", site)
                        # current_states = self.get_neighboursSnake(hiddenstates, zerohiddenstate, site, Ny)
                        # current_inputs = self.get_neighboursSnake(onehotsamples, zeroinput, site, Ny)

                        hiddenstate = self.MDRNNCell(current_states,current_inputs)

                        #hiddenstates.append(hiddenstate)
                        hiddenstates[site] = hiddenstate

                        #Apply the Softmax layer
                        output_prob = self.dense_prob(hiddenstate) + 1e-8

                        activations_up = tf.ones(self.numsamples, dtype = tf.float64)
                        activations_down = tf.ones(self.numsamples, dtype = tf.float64)

                        output_prob = output_prob * tf.cast(tf.stack([activations_down, activations_up], axis = 1), tf.float64)
                        output_prob = tf.nn.l2_normalize(tf.sqrt(output_prob), axis = 1, epsilon = 1e-12)**2
                        #---------------------

                        sample_temp = tf.reshape(tf.random.categorical(tf.log(output_prob), num_samples = 1), [-1, ]) # Sample from the probability
                        samples[site] = sample_temp
                        inputs = tf.one_hot(sample_temp, depth = self.outputdim)
                        onehotsamples[site] = inputs

            self.samples = tf.stack(values = samples, axis = 1) # (self.N, num_samples) to (num_samples, self.N): Generate self.numsamples vectors of size self.N errors containing 0 or 1

            return self.samples


    def log_probability(self, errors, inputdim):
          """
          Purpose:
              - Calculate the log-probabilities of the samples that are generated.

          Parameters:
              - errors (array of shape (numsamples, N**2) ): The error configurations.
              - inputdim (integer): Dimension of the input space (so, 2).

          Output:
              - log-probs (tf.Tensor of shape (numsamples,) ): Log-probability of each sample.
          """

          with self.graph.as_default():

              self.inputdim = inputdim
              self.outputdim = self.inputdim

              self.numsamples = tf.shape(errors)[0]
              zeroinput = tf.zeros([self.numsamples, inputdim], dtype = tf.float64)

              with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                  #probs=[]
                  probs = [0]*self.N**2
                  #onehotsamples = []
                  onehotsamples = [0]*self.N**2

                  """
                  # Get the important lattice sites
                  BoundaryListX, BulkListX, FullListX = ImportantXSites(self.N)
                  """

                  # For the Snake scanning
                  BoundaryListX, BulkListX, FullListX = ImportantXSites(self.N)

                  # Initialize the hidden state
                  zerohiddenstate = tf.zeros([self.numsamples, self.num_units], dtype = tf.float64)
                  #hiddenstates = []
                  hiddenstates = [0]*self.N**2
                  # Note: The zero state returns a zero-filled tensor with shape = (self.numsamples, num_units)

                  errors_temp = tf.transpose(tf.reshape( tf.cast(errors, tf.int64), [self.numsamples, self.N**2] )) #Reshape the errors to deal with them.

                  for Ny in range(self.N):
                    for Nx in range(self.N):
                      site = self.N*Ny + Nx
                      # if Ny % 2 == 0:
                          # site = self.N*Ny + Nx
                      # else:
                          # site = self.N*(Ny + 1) - (1 + Nx)

                      current_states = self.get_neighbours(hiddenstates, zerohiddenstate, site)
                      current_inputs = self.get_neighbours(onehotsamples, zeroinput, site)

                      # current_states = self.get_neighboursSnake(hiddenstates, zerohiddenstate, site, Ny)
                      # current_inputs = self.get_neighboursSnake(onehotsamples, zeroinput, site, Ny)

                      hiddenstate = self.MDRNNCell(current_states,current_inputs)

                      #hiddenstates.append(hiddenstate)
                      hiddenstates[site] = hiddenstate
                      # Apply the Softmax layer
                      output_prob = self.dense_prob(hiddenstate) + 1e-8

                      activations_up = tf.ones(self.numsamples, dtype = tf.float64)
                      activations_down = tf.ones(self.numsamples, dtype = tf.float64)

                      output_prob = output_prob * tf.cast(tf.stack([activations_down, activations_up], axis = 1), tf.float64)
                      output_prob = tf.nn.l2_normalize(tf.sqrt(output_prob), axis = 1, epsilon = 1e-12)**2
                      #-------------------

                      #probs.append(output_prob)
                      probs[site] = output_prob
                      inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(errors,begin=[np.int32(0),np.int32(site)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim, dtype = tf.float64),shape=[self.numsamples,self.inputdim])
                      #onehotsamples.append(inputs)
                      onehotsamples[site] = inputs

              probs = tf.cast(tf.transpose(tf.stack(values = probs, axis = 2), perm = [0,2,1]), tf.float64)
              one_hot_errors = tf.one_hot(errors, depth = self.inputdim, dtype = tf.float64)

              self.log_probs = tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(probs, one_hot_errors), axis = 2)), axis = 1)

              return self.log_probs
