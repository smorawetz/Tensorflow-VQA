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

    # NOTE: Idea to is track the Hamiltonian on the fly, so as to not store all sparse matrices in dicts

    gs_energies = np.zeros(num_intervals)
    fes_energies = np.zeros(num_intervals)
    ses_energies = np.zeros(num_intervals)

    B_vals = np.linspace(0, B_start, num_intervals)
    for B_index in range(len(B_vals)):
        start_time = time.time()
        B = B_vals[B_index]
        N = L ** 2
        matrices = [ I ] * N

        for i in range(N):
            j = (i + 1) % L + L * (i // L)
            k = (i + L) % L ** 2
            temp_matrices = matrices.copy()
            temp_matrices[i] = Z
            temp_mat1 = temp_matrices[0]
            for ind in range(1, N):
                temp_mat1 = sparse.kron(temp_mat1, temp_matrices[ind], format="csr")

            temp_matrices = matrices.copy()
            temp_matrices[j] = Z
            temp_mat2 = temp_matrices[0]
            for ind in range(1, N):
                temp_mat2 = sparse.kron(temp_mat2, temp_matrices[ind], format="csr")

            temp_matrices = matrices.copy()
            temp_matrices[k] = Z
            temp_mat3 = temp_matrices[0]
            for ind in range(1, N):
                temp_mat3 = sparse.kron(temp_mat3, temp_matrices[ind], format="csr")

            if i == 0:
                H = J / 2 * temp_mat1.multiply(temp_mat2).multiply(temp_mat3)
            else:
                H += J / 2 * temp_mat1.multiply(temp_mat2).multiply(temp_mat3)

            print(i)
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(usage)

        for i in range(N):
            temp_matrices = matrices.copy()
            temp_matrices[i] = X
            temp_mat = temp_matrices[0]
            for j in range(1, N):
                temp_mat = sparse.kron(temp_mat, temp_matrices[j], format="csr")
            H -= B * temp_mat

            print(i)
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(usage)

        # Add NM terms

        eigvals, eigvecs = linalg.eigsh(H, k=3, which="SA")
        gs_energies[B_index] = eigvals[0]
        fes_energies[B_index] = eigvals[1]
        ses_energies[B_index] = eigvals[2]

        end_time = time.time()
        print("done B = ", B, " in ", end_time - start_time)

    np.savetxt("exact_data/lanczos_B_data_L{0}.txt".format(L), B_vals)
    np.savetxt("exact_data/lanczos_gs_energies_L{0}.txt".format(L), gs_energies)
    np.savetxt("exact_data/lanczos_fes_energies_L{0}.txt".format(L), fes_energies)
    np.savetxt("exact_data/lanczos_ses_energies_L{0}.txt".format(L), ses_energies)


make_data(4, 1)
