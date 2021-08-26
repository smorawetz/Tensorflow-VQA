import numpy as np

# define matrices
I = np.array([[1, 0], [0, 1]])
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])

J = 1

num_intervals = 101

def make_data(L, B_start):

    energy_levels = np.zeros((num_intervals, 2 ** (L ** 2)))

    B_vals = np.linspace(0, B_start, num_intervals)
    for B_index in range(len(B_vals)):
        B = B_vals[B_index]
        N = L ** 2
        matrices = [ I ] * N

        J_mats_dict = {}

        for i in range(N):
            temp_matrices = matrices.copy()
            temp_matrices[i] = Z
            temp_mat = temp_matrices[0]
            for j in range(1, N):
                temp_mat = np.kron(temp_mat, temp_matrices[j])
            J_mats_dict[i] = temp_mat.copy()

        B_mats_dict = {}

        for i in range(N):
            temp_matrices = matrices.copy()
            temp_matrices[i] = X
            temp_mat = temp_matrices[0]
            for j in range(1, N):
                temp_mat = np.kron(temp_mat, temp_matrices[j])
            B_mats_dict[i] = temp_mat.copy()

        H = np.zeros((2 ** N, 2 ** N))

        # Add NM terms
        for first_index in range(N):
            # Determine indices for summing using rhombic lattice (see repo image)
            second_index = (first_index + 1) % L + L * (first_index // L)
            third_index = (first_index + L) % L ** 2
            H += J / 2 * np.matmul(J_mats_dict[first_index], np.matmul(J_mats_dict[second_index], J_mats_dict[third_index]))

        # add transverse field terms
        for i in range(N):
            H -= B * B_mats_dict[i]

        eigvals = np.sort(np.real_if_close(np.linalg.eigvals(H)))

        energy_levels[B_index, :] = eigvals

        print("done B = ", B)

    np.savetxt("exact_data/B_data_L{0}.txt".format(L), B_vals)
    np.savetxt("exact_data/energy_levels_L{0}.txt".format(L), energy_levels)


make_data(3, 1)
