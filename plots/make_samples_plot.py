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

Ls = [13]  # Side length of lattice
Ls = [16]  # Side length of lattice
NH = 40  # Number of hidden units
LR = 0.001  # Learning rate
NS = 100  # Number of samples to compute F
NW = 1000  # Number of warmup steps
NT = 5  # Number of training steps
NA = 10000  # Number of annealing steps
SEEDs = [0, 1, 2, 3]  # Random seed

Ts = [10]  # Temperature

model_type = "NM"
# model_type = "Ising"

SAMPLES_TYPES = ["warmup", "completion"]

for L in Ls:
    for T in Ts:
        T0 = T
        for SEED in SEEDs:
            for SAMPLES_TYPE in SAMPLES_TYPES:
                folder = "../results/{0}_L{1}_results/L{1}_T{2}_nh{3}_lr{4}_Ns{5}_Nw{6}_Nt{7}_Na{8}_seed{9}".format(
                    model_type, L, T0, NH, LR, NS, NW, NT, NA, SEED
                )
                fname = "{0}/{1}_samples.txt".format(folder, SAMPLES_TYPE)

                data = np.loadtxt(fname)
                
                # TESTING
                counts_dict = {}
                for sample_num in range(NS):
                    sample = data[sample_num, :]
                    nice_sample = "".join(["{0}".format(int(val)) for val in sample])
                    if nice_sample in counts_dict.keys():
                        counts_dict[nice_sample] += 1
                    else:
                        counts_dict[nice_sample] = 1

                sample_names = list(counts_dict.keys())
                sample_counts = list(counts_dict.values())
                all_zeros_str = "0" * L ** 2
                colours = []
                tick_labels = []
                for sample_name in sample_names:
                    if sample_name == all_zeros_str:
                        colours.append('r')
                        tick_labels.append("All 0s")
                    else:
                        colours.append('b')
                        tick_labels.append("Not all 0s")
                ax.bar(sample_names, sample_counts, color=colours, tick_label=tick_labels)
                ax.set_title("Ground state sampling counts for L = {0} at T = 0".format(L))
                # ax.set_xticklabels([])
                # fig.legend()

                plot_name = (
                    "{0}_L{1}_T{2}_nh{3}_lr{4}_Ns{5}_Nw{6}_Nt{7}_Na{8}_seed{9}_{10}_sample_counts.png".format(
                        model_type, L, T0, NH, LR, NS, NW, NT, NA, SEED, SAMPLES_TYPE
                    )
                )

                plt.savefig(plot_name, bbox_inches="tight")
