"""Helper functions."""


def add_twin(ax, direction="in", **kwargs):
    twin = ax.twinx()
    twin.yaxis.tick_left()
    twin.tick_params(axis="y", direction=direction, **kwargs)
    for tick in twin.get_yticklabels():
        tick.set_horizontalalignment("right")
    return twin


def add_twin_right(ax, direction="in", **kwargs):
    twin = ax.twinx()
    twin.yaxis.tick_right()
    twin.tick_params(axis="y", direction=direction, **kwargs)
    for tick in twin.get_yticklabels():
        tick.set_horizontalalignment("right")
    return twin


def add_twin_bottom(ax, direction="in", **kwargs):
    twin = ax.twiny()
    twin.xaxis.tick_bottom()
    twin.tick_params(axis="x", direction=direction, **kwargs)
    for tick in twin.get_xticklabels():
        tick.set_horizontalalignment("right")
    return twin
