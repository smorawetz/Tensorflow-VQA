import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate as integrate

# plot style
params = {
    "text.usetex": True,
    "font.family": "serif",
    "legend.fontsize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1,
    "patch.edgecolor": "black",
    "pgf.rcfonts": False,
}
plt.rcParams.update(params)
plt.style.use("seaborn-deep")


# Define parameters of the study to make the visualization

Ls = [16]  # Side length of lattice
NH = 40  # Number of hidden units
LR = 0.001  # Learning rate
NS = 50  # Number of samples to compute F
NW = 1000  # Number of warmup steps
NT = 5  # Number of training steps
NA = 1000  # Number of annealing steps
SEEDs = [0, 1, 2, 3]  # Random seed
# MAX_CHANGES = [0.1, 1]  # Possible values the landscape can range over
MAX_CHANGES = [1]  # Possible values the landscape can range over
# LANDSCAPE_TYPES = ["F", "cost"]
LANDSCAPE_TYPES = ["F"]

NUM_MID_VIZ = 3  # Number of visualizations made during annealing
LANDSCAPE_TIMES = ["init", "warmup", "completion"]  # Landscape before or after warmup

for mid_viz_num in range(1, NUM_MID_VIZ + 1):
    LANDSCAPE_TIMES.append("mid_viz{0}_of{1}".format(mid_viz_num, NUM_MID_VIZ))

N_POINTS = 201  # Number of points between -alpha, +alpha

Ts = [10]  # Temperature

model_type = "NM"
# model_type = "Ising"

title_names = {"F": "Free energy", "cost": "Loss function"}

# Define integral to compute Onsager solution
# def integrand(theta1, theta2, beta):
    # return - 1 / beta * ( np.log(2) + 1 / 8 / np.pi / np.pi * np.log(np.cosh(2 * beta) * np.cosh(2 * beta) - np.sinh(2 * beta) * (np.cos(theta1) + np.cos(theta2))))

for L in Ls:
    for T in Ts:
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$T$")
        ax.set_ylabel(r"$F / L^2$")
        ax.set_title("Free energy density at different temperatures")
        T0 = T
        for SEED in SEEDs:
            folder = "../results/{0}_L{1}_results/L{1}_T{2}_nh{3}_lr{4}_Ns{5}_Nw{6}_Nt{7}_Na{8}_seed{9}".format(
                model_type, L, T0, NH, LR, NS, NW, NT, NA, SEED
            )
            fname = "{0}/training_info.txt".format(folder)

            data = np.loadtxt(fname)

            T = data[:, 1]
            F = data[:, 3]

            ax.plot(T, F / L ** 2, "-", label="Neural Annealing, Seed {0}".format(SEED))

        if model_type == "NM" and np.log2(L) % 1 == 0:
            analytical = - T * np.log(1 + np.exp(- 1 / (T + 1e-12))) - 0.5  # NOTE: -1 to correct for Ising -> binary spins
            ax.plot(T, analytical, "r-", label="Analytical Result")

        plot_name = (
            "{0}_L{1}_T{2}_nh{3}_lr{4}_Ns{5}_Nw{6}_Nt{7}_Na{8}_compare_seeds_F_vs_T.png".format(
                model_type, L, T0, NH, LR, NS, NW, NT, NA,
            )
        )
        fig.legend(loc=[0.1, 0.1], frameon=False)

        plt.savefig(plot_name, bbox_inches="tight")
