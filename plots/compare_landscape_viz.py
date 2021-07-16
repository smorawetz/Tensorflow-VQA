import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

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

Ls = [13, 16]  # Side length of lattice
NH = 40  # Number of hidden units
LR = 0.001  # Learning rate
NS = 100  # Number of samples to compute F
NW = 1000  # Number of warmup steps
NT = 5  # Number of training steps
NA = 10000  # Number of annealing steps
# SEEDs = [0, 1, 2, 3, 4]  # Random seed
SEEDs = [0]  # Random seed
# MAX_CHANGES = [0.25, 1.0, 3.0, 5.0]  # Possible values the landscape can range over
MAX_CHANGES = [1.0]  # Possible values the landscape can range over
LANDSCAPE_TYPES = ["F"]

NUM_MID_VIZ = 5  # Number of visualizations made during annealing
LANDSCAPE_TIMES = ["init", "warmup", "completion"]  # Landscape before or after warmup

for mid_viz_num in range(1, NUM_MID_VIZ + 1):
    LANDSCAPE_TIMES.append("mid_viz{0}_of{1}".format(mid_viz_num, NUM_MID_VIZ))

N_POINTS = 201  # Number of points between -alpha, +alpha

T = 1.0  # Temperature

nice_names = {
    "init": "before warmup",
    "warmup": "after warmup",
    "completion": "after annealing",
}
for mid_viz_num in range(1, NUM_MID_VIZ + 1):
    nice_names["mid_viz{0}_of{1}".format(mid_viz_num, NUM_MID_VIZ)] = (
        "during annealing T = {0}".format(
            round(T * (1 - mid_viz_num / (NUM_MID_VIZ + 1)), 2)
        )
    )

model_type = "NM"
# model_type = "Ising"

title_names = {"F": "Free energy", "cost": "Loss function"}

for SEED in SEEDs:
    for LAND_TYPE in LANDSCAPE_TYPES:
        for LAND_TIME in LANDSCAPE_TIMES:
            for MAX_CHANGE in MAX_CHANGES:
                fig, axs = plt.subplots(nrows=1, ncols=len(Ls), figsize=(10, 4))
                for i in range(len(Ls)):
                    L = Ls[i]
                    ax = axs[i]

                    folder = "../results/{0}_L{1}_results/L{1}_T{2}_nh{3}_lr{4}_Ns{5}_Nw{6}_Nt{7}_Na{8}_seed{9}".format(
                        model_type, L, T, NH, LR, NS, NW, NT, NA, SEED + i
                    )
                    fname = "{0}/{1}_landscape_viz_{2}_max{3}_Npoints{4}.txt".format(
                        folder, LAND_TYPE, LAND_TIME, MAX_CHANGE, N_POINTS
                    )

                    data = np.loadtxt(fname)

                    norm = colors.Normalize(vmin=data.min(), vmax=data.max())
                    contour = ax.imshow(
                        data,
                        cmap="YlGn",
                        norm=norm,
                        extent=[-MAX_CHANGE, MAX_CHANGE, -MAX_CHANGE, MAX_CHANGE,],
                    )
                    ax.set_xlim(-MAX_CHANGE, MAX_CHANGE)
                    ax.set_ylim(-MAX_CHANGE, MAX_CHANGE)
                    fig.colorbar(contour, ax=ax)

                    ax.set_title(r"$L = {0}$".format(L,))

                    ax.set_xlabel(r"$\alpha$")
                    ax.set_ylabel(r"$\beta$")

                fig.suptitle("Free energy landscape during annealing")

                plot_name = (
                    "{0}_{1}_{2}_landscape_max{3}_npoints{4}_L{5}_T{6}_seed{7}.png"
                    .format(
                        model_type,
                        LAND_TYPE,
                        LAND_TIME,
                        MAX_CHANGE,
                        N_POINTS,
                        L,
                        T,
                        SEED,
                    )
                )

                plt.savefig(plot_name)
