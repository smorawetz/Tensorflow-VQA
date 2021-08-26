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

Ls = [3]  # Side length of lattice
NH = 40  # Number of hidden units
LR = 0.001  # Learning rate
NS = 100  # Number of samples to compute F
NW = 100  # Number of warmup steps
NT = 5  # Number of training steps
NA = 100  # Number of annealing steps
SEEDs = [0]  # Random seed
MAX_CHANGES = [1.0]  # Possible values the landscape can range over
# MAX_CHANGES = [0.25, 1.0, 3.0]  # Possible values the landscape can range over
LANDSCAPE_TYPES = ["cost"]

NUM_MID_VIZ = 5  # Number of visualizations made during annealing
LANDSCAPE_TIMES = ["init", "warmup", "completion"]  # Landscape before or after warmup

for mid_viz_num in range(1, NUM_MID_VIZ + 1):
    LANDSCAPE_TIMES.append("mid_viz{0}_of{1}".format(mid_viz_num, NUM_MID_VIZ))

N_POINTS = 21  # Number of points between -alpha, +alpha

Ts = [1]  # Temperature

nice_names = {
    "init": "before warmup",
    "warmup": "after warmup",
    "completion": "after annealing",
}

model_type = "NM"
# model_type = "Ising"

title_names = {"F": "Free energy", "cost": "Loss function"}

for L in Ls:
    for T in Ts:
        for mid_viz_num in range(1, NUM_MID_VIZ + 1):
            nice_names["mid_viz{0}_of{1}".format(mid_viz_num, NUM_MID_VIZ)] = (
                "during annealing T = {0}".format(round(T * (1 - mid_viz_num / (NUM_MID_VIZ + 1)), 2))
            )
        for SEED in SEEDs:
            for LAND_TYPE in LANDSCAPE_TYPES:
                for LAND_TIME in LANDSCAPE_TIMES:
                    for MAX_CHANGE in MAX_CHANGES:
                        for THREE_D in [False, True]:
                            folder = "../results/{0}_L{1}_results/L{1}_T{2}_nh{3}_lr{4}_Ns{5}_Nw{6}_Nt{7}_Na{8}_seed{9}".format(
                                model_type, L, T, NH, LR, NS, NW, NT, NA, SEED
                            )
                            fname = (
                                "{0}/{1}_landscape_viz_{2}_max{3}_Npoints{4}.txt".format(
                                    folder, LAND_TYPE, LAND_TIME, MAX_CHANGE, N_POINTS
                                )
                            )

                            data = np.loadtxt(fname)

                            # print(fname, min(data[100,:]))

                            if THREE_D:
                                fig = plt.figure()
                                ax = fig.gca(projection="3d")

                                vals = np.linspace(-MAX_CHANGE, MAX_CHANGE, N_POINTS)
                                grid_alphas, grid_betas = np.meshgrid(vals, vals)
                                ax.plot_surface(grid_alphas, grid_betas, data)

                                ax.set_xlabel(r"$\alpha$")
                                ax.set_ylabel(r"$\beta$")
                                ax.set_zlabel(r"Loss function")

                                ax.set_title(
                                    r"{0} {1}: $T = {2}$".format(
                                        title_names[LAND_TYPE],
                                        nice_names[LAND_TIME],
                                        T,
                                    )
                                )

                                plot_name = "plots_landscape/{0}_3D_{1}_{2}_landscape_max{3}_npoints{4}_L{5}_T{6}_seed{7}.png".format(
                                    model_type,
                                    LAND_TYPE,
                                    LAND_TIME,
                                    MAX_CHANGE,
                                    N_POINTS,
                                    L,
                                    T,
                                    SEED,
                                )

                                plt.savefig(plot_name)

                            else:
                                fig, ax = plt.subplots()

                                norm = colors.Normalize(vmin=data.min(), vmax=data.max())
                                contour = ax.imshow(
                                    data,
                                    cmap="YlGn",
                                    norm=norm,
                                    extent=[
                                        -MAX_CHANGE,
                                        MAX_CHANGE,
                                        -MAX_CHANGE,
                                        MAX_CHANGE,
                                    ],
                                )
                                ax.set_xlim(-MAX_CHANGE, MAX_CHANGE)
                                ax.set_ylim(-MAX_CHANGE, MAX_CHANGE)
                                fig.colorbar(contour, ax=ax)

                                ax.set_title(
                                    r"{0} {1}: $T = {2}$".format(
                                        title_names[LAND_TYPE],
                                        nice_names[LAND_TIME],
                                        T,
                                    )
                                )

                                ax.set_xlabel(r"$\alpha$")
                                ax.set_ylabel(r"$\beta$")

                                plot_name = "plots_landscape/{0}_{1}_{2}_landscape_max{3}_npoints{4}_L{5}_T{6}_seed{7}.png".format(
                                    model_type,
                                    LAND_TYPE,
                                    LAND_TIME,
                                    MAX_CHANGE,
                                    N_POINTS,
                                    L,
                                    T,
                                    SEED,
                                )

                                plt.savefig(plot_name)
