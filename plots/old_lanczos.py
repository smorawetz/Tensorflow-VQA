import time
import resource

import numpy as np

import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

# define matrices
I = sparse.csr_matrix([[1, 0], [0, 1]])
Z = sparse.csr_matrix([[1, 0], [0, -1]])
X = sparse.csr_matrix([[0, 1], [1, 0]])

J = 1

num_intervals = 101

def make_data(L, B_start):

    gs_energies = np.zeros(num_intervals)

    B_vals = np.linspace(0, B_start, num_intervals)
    for B_index in range(len(B_vals)):
        start_time = time.time()
        B = B_vals[B_index]
        N = L ** 2
        matrices = [ I ] * N

        J_mats_dict = {}

        for i in range(N):
            temp_matrices = matrices.copy()
            temp_matrices[i] = Z
            temp_mat = temp_matrices[0]
            for j in range(1, N):
                temp_mat = sparse.kron(temp_mat, temp_matrices[j], format="csr")
            J_mats_dict[i] = temp_mat.copy()
            print(i)
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(usage)

        B_mats_dict = {}

        for i in range(N):
            temp_matrices = matrices.copy()
            temp_matrices[i] = X
            temp_mat = temp_matrices[0]
            for j in range(1, N):
                temp_mat = sparse.kron(temp_mat, temp_matrices[j], format="csr")
            B_mats_dict[i] = temp_mat.copy()

        # Add NM terms
        for first_index in range(N):
            # Determine indices for summing using rhombic lattice (see repo image)
            second_index = (first_index + 1) % L + L * (first_index // L)
            third_index = (first_index + L) % L ** 2
            if first_index == 0:
                H = J / 2 * J_mats_dict[first_index].multiply(J_mats_dict[second_index]).multiply(J_mats_dict[third_index])
            else:
                H += J / 2 * J_mats_dict[first_index].multiply(J_mats_dict[second_index]).multiply(J_mats_dict[third_index])

        # add transverse field terms
        for i in range(N):
            H -= B * B_mats_dict[i]

        eigvals, eigvecs = linalg.eigsh(H, k=1)
        gs_energies[B_index] = eigvals[0]

        end_time = time.time()
        print("done B = ", B, " in ", end_time - start_time)

    np.savetxt("exact_data/lanczos_B_data_L{0}.txt".format(L), B_vals)
    np.savetxt("exact_data/lanczos_gs_energies_L{0}.txt".format(L), gs_energies)


make_data(5, 1)
