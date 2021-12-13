import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Set default style
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
    # https://maketintsandshades.com/#fce5cd
    '#fce5cd',
    '#e3ceb9',
    '#cab7a4',
    '#b0a090',
    '#97897b',
])
mpl.rcParams.update({
    'font.weight': 'bold',
    'font.size': 17
})


def time_visualize(*runs, arch=None, ax=None):
    """
    Plots time for one architecture based on nodes separated into
    forward/backward/misc phases

    runs: array
        Assumed to have entries of form
        ['name', forward time, backward time, total time]
    """
    if ax is None:
        _, ax = plt.subplots()

    width = .4

    names = [run[0] for run in runs]
    forwards = np.array([run[1] for run in runs]).squeeze().astype(np.float32)
    backwards = np.array([run[2] for run in runs]).squeeze().astype(np.float32)
    totals = np.array([run[3] for run in runs]).squeeze().astype(np.float32)
    totals -= forwards + backwards

    ax.bar(names, forwards, width, label='Forward')
    bottom = forwards
    ax.bar(names, backwards, width, label='Backward', bottom=bottom)
    bottom += backwards
    ax.bar(names, totals, width, label='Total', bottom=bottom)

    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Nodes')
    if arch is not None:
        ax.set_title(arch)
    else:
        ax.set_title('Time by # of Nodes and Phase')
    ax.legend()


def per_architecture_backward_visualize(*runs, ax=None):
    """
    Plots time not accounted for by forward or backward

    runs: array
        Assumed to have entries of form
        ['architecture name', 'setting name', forward time, backward time, total time]
    """
    if ax is None:
        _, ax = plt.subplots()

    width = 5

    architectures = np.array([run[0] for run in runs]).squeeze()
    settings = np.array([run[1] for run in runs]).squeeze()
    forwards = np.array([run[2] for run in runs]).squeeze().astype(np.float32)
    backwards = np.array([run[3] for run in runs]).squeeze().astype(np.float32)
    totals = np.array([run[4] for run in runs]).squeeze().astype(np.float32)
    totals = backwards / totals

    for arch in np.unique(architectures):
        idx = architectures==arch
        set = settings[np.argwhere(idx)].squeeze()
        tot = totals[np.argwhere(idx)].squeeze()
        ax.plot(set, tot, 'd-', linewidth=width, markersize=3*width, label=arch)

    ax.set_ylabel('Fraction Time')
    ax.set_xlabel('Nodes')
    ax.set_title('Backward Time by # of Nodes')
    ax.legend()


def per_architecture_time_visualize(*runs, ax=None, legend=True):
    """
    Plots total time per architecture

    runs: array
        Assumed to have entries of form
        ['architecture name', 'setting name', total time]
    """
    if ax is None:
        _, ax = plt.subplots()

    width = 5

    architectures = np.array([run[0] for run in runs]).squeeze()
    settings = np.array([run[1] for run in runs]).squeeze()
    totals = np.array([run[2] for run in runs]).squeeze().astype(np.float32)

    for arch in np.unique(architectures):
        idx = architectures==arch
        set = settings[np.argwhere(idx)].squeeze()
        tot = totals[np.argwhere(idx)].squeeze()
        ax.plot(set, tot, 'd-', linewidth=width, markersize=3*width, label=arch)

    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Nodes')
    ax.set_title('Time by # of Nodes and Architecture')
    if legend:
        ax.legend()
