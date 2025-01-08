import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import string
from run_exp import get_timestamp


def get_ticks(data_min, data_max, n_spaces):
    delta = data_max - data_min
    return [data_min + i * delta / n_spaces for i in range(n_spaces + 1)]


palette = ["#808080", "#ee82ee", "#FFA500", "#7741f7", "#805380"]
sampling = ["random", "greedy", "epig", "greedyskill", "epigskill"]
dataset_name_lookup = {"thiol": "TR", "redox": "RR", "fluc": "FI", "nluc": "NI"}
sampling_name_look_up = {
    "random": "Random",
    "greedy": "Greedy",
    "epig": "EPIG",
    "greedyskill": "GreedySkill",
    "epigskill": "EPIGSkill",
}

COLOR_DICT = {samp: col for samp, col in zip(sampling, palette)}


def main():
    time = "2024-10-25_09:06:37"
    teacher = pd.read_csv(f"result_csv/comparision_teacher_{time}.csv")
    seeds = pd.read_csv(f"result_csv/comparision_seeds_{time}.csv")
    fig, axs = plt.subplots(
        teacher.dataset.nunique(),
        len(sampling),
        figsize=(20, 14),
    )
    offset_scale = 0.2
    n_spaces_y = 4
    iterations = 10
    n_spaces_x = iterations - 1
    formatter = ticker.StrMethodFormatter("{x:.2f}")
    fontsize = 20
    for i, dataset in enumerate(teacher.dataset.unique()):
        all_cka = (
            teacher.query(f"dataset == '{dataset}'").cka_rf_to_teacher.to_list()
            + seeds.query(f"dataset == '{dataset}'").inter_replica_cka_rf.to_list()
        )

        cka_min = min(all_cka)
        cka_max = max(all_cka)
        y_ticks = get_ticks(cka_min, cka_max, n_spaces_y)
        print(dataset, cka_min, cka_max, y_ticks)
        delta_cka = cka_max - cka_min

        iter_min = 1
        iter_max = iterations
        x_ticks = get_ticks(iter_min, iter_max, n_spaces_x)
        delta_iter = iter_max - iter_min

        for j, acquisition_f in enumerate(sampling):
            ax = axs[i, j]
            # ax2 = ax.twinx()
            sns.lineplot(
                teacher.query(
                    f"dataset == '{dataset}' and acquisition_func == '{acquisition_f}'"
                ),
                x="iteration",
                y="cka_rf_to_teacher",
                ax=ax,
                color="black",
            )
            sns.lineplot(
                seeds.query(
                    f"dataset == '{dataset}' and acquisition_func == '{acquisition_f}'"
                ),
                x="iteration",
                y="inter_replica_cka_rf",
                ax=ax,
                linestyle="dashed",
                color="black",
            )

            if j == 0:
                ax.set_ylabel(f"{dataset_name_lookup[dataset]}", fontsize=fontsize)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            if i == 0:
                ax.set_title(
                    f"{sampling_name_look_up[acquisition_f]}",
                    fontsize=fontsize,
                    color=COLOR_DICT[acquisition_f],
                )

            ax.set_ylim(
                cka_min - offset_scale * delta_cka / n_spaces_y,
                cka_max + offset_scale * delta_cka / n_spaces_y,
            )
            ax.set_yticks(y_ticks)

            ax.set_xlim(
                iter_min - offset_scale * delta_iter / n_spaces_x,
                iter_max + offset_scale * delta_iter / n_spaces_x,
            )
            ax.set_xticks(x_ticks)
    for n, ax in enumerate(axs.flat):
        ax.text(
            -0.20,  # -0.15
            1.03,  # 1.03
            "(" + string.ascii_lowercase[n] + ")",
            transform=ax.transAxes,  #
            size=fontsize - 5,
            weight="bold",
        )
        ax.yaxis.set_major_formatter(formatter)

    legend = plt.legend([None, None, None])
    handles = legend.legend_handles
    plt.legend(
        handles[0:1] + handles[2:3],
        [r"student-teacher CKA$_{\text{rf}}$", r"inter-student CKA$_{\text{rf}}$"],
        loc="upper center",
        bbox_to_anchor=(-2.3, -0.2),
        fancybox=False,
        shadow=False,
        ncol=2,
        fontsize=fontsize,
    )
    for i in range(16, 21, 1):
        ax = axs.flat[i - 1]
        ax.set_xlabel("Iteration")
    plt.tight_layout()
    # [r'student-teacher CKA$_{\text{rf}}$', None , r'inter-student CKA$_{\text{rf}}$']
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    plt.savefig(f"similarities_{time}.pdf")
    plt.show()


if __name__ == "__main__":
    main()
