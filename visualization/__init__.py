"""
Public API for the visualisation package.

Exposes the two functions used by the main simulation runner:
    - update_plot : live animation frame update (called every ANIMATION_STEP steps)
    - plot_final  : static final plot shown/saved at the end of the simulation

All matplotlib figure management is handled internally in plotting.py.
"""

from .plotting import update_plot, plot_final

__all__ = ["update_plot", "plot_final"]