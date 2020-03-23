import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np


def plot_categoric_attribute(data_frame, column, **kwargs):
    count = kwargs.get('count', 10)

    plt.xticks(rotation=90)
    tmp_data = data_frame[column].value_counts(normalize=True).rename('percentage').reset_index().sort_values(
        by='percentage', ascending=False)[:count]
    ax = sns.barplot(x="index", y="percentage", data=tmp_data)
    ax.set_title(f'Top {count} values in {column}')


def plot_discrete_attribute(data_frame, column):
    sns.boxplot(x=column, data=data_frame)


def heat_map(data_frame, **kwargs):
    figsize = kwargs.get('figsize', (10, 10))

    corr = data_frame.corr()
    fig, ax = plt.subplots(figsize=figsize)

    # Generate Color Map
    drop_self = np.zeros_like(corr)
    drop_self[np.triu_indices_from(drop_self)] = True  # Generate Color Map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    # Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=drop_self)

    # Apply xticks
    plt.xticks(range(len(corr.columns)), corr.columns);

    # Apply yticks
    plt.yticks(range(len(corr.columns)), corr.columns)

    # show plot
    plt.show()

    return fig