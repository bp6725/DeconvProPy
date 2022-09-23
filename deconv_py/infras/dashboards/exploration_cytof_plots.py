import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pandas as pd
import numpy as np

def plot_mass_to_cytof_scatter(res_as_cytof, sample_over_cytof_count, mass_results_not_none_df):
    res_as_cytof_for_corr = res_as_cytof.copy(deep=True)
    sample_over_cytof_count = sample_over_cytof_count

    try:
        res_as_cytof_for_corr = res_as_cytof.copy(deep=True).drop(["Unknown"])  # 24_v1
        sample_over_cytof_count = sample_over_cytof_count.rename(
            columns={col: f"{col.split('-')[1]}_v{col.split('V')[1]}" for col in sample_over_cytof_count.columns})
    except:
        pass

    mass_results_not_none_over_sample = mass_results_not_none_df[sample_over_cytof_count.columns]

    mutual_mixtures = res_as_cytof_for_corr.columns.intersection(sample_over_cytof_count.columns).tolist()

    for mixture in mutual_mixtures:
        lab = res_as_cytof_for_corr[mixture].index.tolist()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        for i in range(len(res_as_cytof_for_corr[mixture])):
            x_cytof = res_as_cytof_for_corr[mixture].iloc[i]
            x_not_none_in_cytof = mass_results_not_none_over_sample[mixture].iloc[i]
            y = sample_over_cytof_count[mixture].iloc[i]


            ax1.plot(x_cytof, y, 'bo')
            ax1.text(x_cytof, y, lab[i], fontsize=12)

            ax2.plot(x_not_none_in_cytof, y, 'bo')
            ax2.text(x_not_none_in_cytof, y, lab[i], fontsize=12)

        corr = res_as_cytof_for_corr[mixture].corr(sample_over_cytof_count[mixture])
        ax1.set_title(f"with labeling , mixture - {mixture} , corr - {round(corr,2)}")
        ax1.set_ylabel("cytof")
        ax1.set_xlabel("Deconv")

        corr2 = mass_results_not_none_over_sample[mixture].corr(sample_over_cytof_count[mixture])
        ax2.set_title(f"no labeling , mixture - {mixture} , corr - {round(corr2,2)}")
        ax2.set_ylabel("cytof")
        ax2.set_xlabel("Deconv")

        plt.subplots_adjust(wspace=0.4)
        plt.show()

def plot_mass_to_cytof_scatter_all_on_one(res_as_cytof, sample_over_cytof_count, mass_results_not_none_df):
    res_as_cytof_for_corr = res_as_cytof.copy(deep=True)
    sample_over_cytof_count = sample_over_cytof_count

    try:
        res_as_cytof_for_corr = res_as_cytof.copy(deep=True).drop(["Unknown"])  # 24_v1
        sample_over_cytof_count = sample_over_cytof_count.rename(
            columns={col: f"{col.split('-')[1]}_v{col.split('V')[1]}" for col in sample_over_cytof_count.columns})
    except:
        pass

    mass_results_not_none_over_sample = mass_results_not_none_df[sample_over_cytof_count.columns]
    mutual_mixtures = res_as_cytof_for_corr.columns.intersection(sample_over_cytof_count.columns).tolist()

    propgated_scatter_results = []
    not_propgated_scatter_results = []
    for mixture in mutual_mixtures:
        for cell, x_cytof in res_as_cytof_for_corr[mixture].iteritems():
            x_not_none_in_cytof = mass_results_not_none_over_sample[mixture].loc[cell]
            y = sample_over_cytof_count[mixture].loc[cell]
            propgated_scatter_results.append((x_cytof, y, cell))
            not_propgated_scatter_results.append((x_not_none_in_cytof, y, cell))

    cell_to_color_map = {cell: color for cell, color in zip(res_as_cytof.index, mcolors.BASE_COLORS.keys())}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    for i in range(len(propgated_scatter_results)):
        color = f"{cell_to_color_map[propgated_scatter_results[i][2]]}o"
        lab = str(propgated_scatter_results[i][2])
        ax1.plot(propgated_scatter_results[i][0], propgated_scatter_results[i][1], color, label=lab)
        #         ax1.text(propgated_scatter_results[i][0], propgated_scatter_results[i][1], fontsize=12)

        ax2.plot(not_propgated_scatter_results[i][0], not_propgated_scatter_results[i][1], color, label=lab)
    #         ax2.text(not_propgated_scatter_results[i][0], not_propgated_scatter_results[i][1], fontsize=12)

    handles, labels = ax1.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax1.legend(*zip(*unique), loc=(0, -0.24))
    add_identity(ax1, color='r', ls='--')

    handles, labels = ax2.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax2.legend(*zip(*unique), loc=(0, -0.24))
    add_identity(ax2, color='r', ls='--')

    ax1.set_title(f"results vs cytof - with labeling")
    ax1.set_ylabel("cytof")
    ax1.set_xlabel("Deconv")

    ax2.set_title(f"results vs cytof - no labeling")
    ax2.set_ylabel("cytof")
    ax2.set_xlabel("Deconv")

    ax1.set_xlim(0,1)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    #     plt.subplots_adjust(wspace=0.4)
    plt.show()

def plot_mass_to_cytof_scatter_per_cell(res_as_cytof, sample_over_cytof_count, mass_results_not_none_df):
    res_as_cytof_for_corr = res_as_cytof.copy(deep=True)
    sample_over_cytof_count = sample_over_cytof_count

    try:
        res_as_cytof_for_corr = res_as_cytof.copy(deep=True).drop(["Unknown"])  # 24_v1
        sample_over_cytof_count = sample_over_cytof_count.rename(
            columns={col: f"{col.split('-')[1]}_v{col.split('V')[1]}" for col in sample_over_cytof_count.columns})
    except:
        pass

    mass_results_not_none_over_sample = mass_results_not_none_df[sample_over_cytof_count.columns]
    mutual_mixtures = res_as_cytof_for_corr.columns.intersection(sample_over_cytof_count.columns).tolist()

    propgated_scatter_results = {cell: pd.DataFrame(index=mutual_mixtures, columns=["deconv", "cytof"], data=0.0) for
                                 cell in res_as_cytof.index}
    not_propgated_scatter_results = {cell: pd.DataFrame(index=mutual_mixtures, columns=["deconv", "cytof"], data=0.0)
                                     for cell in res_as_cytof.index}
    for mixture in mutual_mixtures:
        for cell, x_cytof in res_as_cytof_for_corr[mixture].iteritems():
            x_not_none_in_cytof = mass_results_not_none_over_sample[mixture].loc[cell]
            y = sample_over_cytof_count[mixture].loc[cell]

            propgated_scatter_results[cell]["deconv"].loc[mixture] = x_cytof
            propgated_scatter_results[cell]["cytof"].loc[mixture] = y

            not_propgated_scatter_results[cell]["deconv"].loc[mixture] = x_not_none_in_cytof
            not_propgated_scatter_results[cell]["cytof"].loc[mixture] = y

    fig, axs = plt.subplots(9, 2, figsize=(20, 15))
    for i, _cell in enumerate(propgated_scatter_results.keys()):
        axs[i, 0].scatter(propgated_scatter_results[_cell]["deconv"], propgated_scatter_results[_cell]["cytof"],
                          label=_cell)
        axs[i, 1].scatter(not_propgated_scatter_results[_cell]["deconv"], not_propgated_scatter_results[_cell]["cytof"],
                          label=_cell)

        axs[i, 0].set_xlim(left=0, right=1)
        axs[i, 1].set_xlim(left=0, right=1)

        axs[i, 0].set_ylim(bottom=0, top=1)
        axs[i, 1].set_ylim(bottom=0, top=1)

        axs[i, 0].set_ylabel("cytof")
        axs[i, 1].set_ylabel("cytof")

        if i == 8:
            axs[i, 0].set_xlabel("Deconv")
            axs[i, 1].set_xlabel("Deconv")

        axs[i, 0].legend()
        axs[i, 1].legend()

        prop_corr = np.round(propgated_scatter_results[_cell]["deconv"].corr(propgated_scatter_results[_cell]["cytof"]),
                             2)
        not_prop_corr = np.round(
            not_propgated_scatter_results[_cell]["deconv"].corr(not_propgated_scatter_results[_cell]["cytof"]), 2)

        if i == 0:
            axs[i, 0].set_title(f"with labeling,corr -{prop_corr}")
            axs[i, 1].set_title(f"no labeling,corr -{not_prop_corr} ")
        else:
            axs[i, 0].set_title(f"with labeling,corr -{prop_corr}")
            axs[i, 1].set_title(f"no labeling,corr -{not_prop_corr} ")

    plt.subplots_adjust(hspace=1)
    plt.show()

def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes