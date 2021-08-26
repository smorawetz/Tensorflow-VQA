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

Ls = [3, 4, 5]  # Side length of lattice
# Ls = [13, 16]  # Side length of lattice
NH = 40  # Number of hidden units
LR = 0.001  # Learning rate
NS = 100  # Number of samples to compute F
NW = 1000  # Number of warmup steps
NT = 5  # Number of training steps
NA = 10000  # Number of annealing steps
SEEDs = [0, 1, 2, 3, 4, 5]  # Random seed
MAX_CHANGES = [0.25, 1.0, 3.0]  # Possible values the landscape can range over
LANDSCAPE_TYPES = ["F"]

NUM_MID_VIZ = 5  # Number of visualizations made during annealing
LANDSCAPE_TIMES = ["init", "warmup", "completion"]  # Landscape before or after warmup

for mid_viz_num in range(1, NUM_MID_VIZ + 1):
    LANDSCAPE_TIMES.append("mid_viz{0}_of{1}".format(mid_viz_num, NUM_MID_VIZ))

# N_POINTS = 201  # Number of points between -alpha, +alpha
N_POINTS = 201  # Number of points between -alpha, +alpha

Ts = [1, 10]  # Temperature

nice_names = {
    "init": "Before warmup: B = 1",
    "warmup": "After warmup: B = 1",
    "completion": "After annealing: B = 0",
}

model_type = "NM"
# model_type = "Ising"

title_names = {"F": "Free energy", "cost": "Loss function"}

LANDSCAPE_TIMES = ["warmup", "mid_viz3_of5", "completion"]

for L in Ls:
    for T in Ts:
        for mid_viz_num in range(1, NUM_MID_VIZ + 1):
            nice_names["mid_viz{0}_of{1}".format(mid_viz_num, NUM_MID_VIZ)] = (
                "During annealing: B = {0}".format(round(T * (1 - mid_viz_num / (NUM_MID_VIZ + 1)), 2))
            )
        for SEED in SEEDs:
            for LAND_TYPE in LANDSCAPE_TYPES:
                for MAX_CHANGE in MAX_CHANGES:
                    try:
                        folder = "../plots_results/{0}_L{1}_results/L{1}_T{2}_nh{3}_lr{4}_Ns{5}_Nw{6}_Nt{7}_Na{8}_seed{9}".format(
                            model_type, L, T, NH, LR, NS, NW, NT, NA, SEED
                        )

                        if LAND_TYPE == "cost":
                            norm = 1
                        else:
                            norm = L ** 2

                        datas = []

                        for LAND_TIME in LANDSCAPE_TIMES:
                            fname = (
                                "{0}/{1}_landscape_viz_{2}_max{3}_Npoints{4}.txt".format(
                                    folder, LAND_TYPE, LAND_TIME, MAX_CHANGE, N_POINTS
                                )
                            )
                            datas.append(np.loadtxt(fname) / norm)

                        fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

                        # datamin = 1
                        # datamax = -1

                        # for data in datas:
                            # if data.min() <= datamin:
                                # datamin = data.min()
                            # if data.max() >= datamax:
                                # datamax = data.max()

                        norm = colors.Normalize(-1, 1)
                        for i in range(len(datas)):
                            data = datas[i]
                            LAND_TIME = LANDSCAPE_TIMES[i]
                            ax = axs[i]
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
                            ax.set_xlabel(r"$\alpha$")
                            ax.set_ylabel(r"$\beta$")
                            ax.set_xlim(-MAX_CHANGE, MAX_CHANGE)
                            ax.set_ylim(-MAX_CHANGE, MAX_CHANGE)


                            ax.set_title(
                                r"{0}".format(
                                    nice_names[LAND_TIME],
                                )
                            )

                            # fig.colorbar(contour, ax=ax)


                        fig.colorbar(contour, ax=axs.ravel().tolist())

                        plot_name = "plots_landscape/{0}_{1}_compare_landscape_max{2}_npoints{3}_L{4}_T{5}_seed{6}.png".format(
                            model_type,
                            LAND_TYPE,
                            MAX_CHANGE,
                            N_POINTS,
                            L,
                            T,
                            SEED,
                        )

                        plt.savefig(plot_name, bbox_inches="tight")
                    except:
                        pass
