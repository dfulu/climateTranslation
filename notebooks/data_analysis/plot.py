import matplotlib.patches as patches
import matplotlib.pyplot as plt


def transform_root_pr_axis_ticks(ax, pr_on_x=True, linear_tick_values=None):
    """Function assumes you have already ploted on the axis `ax` with
    4th root of precip on either the x or y axis. Changes the 4th root labels
    to their linear space values
    
    Parameters
    ----------
    pr_on_x: bool
        4th root of precip plotted on x-axis, else assumes it is on y-axis
    linear_tick_values: array_like
        The values of precip (not 4th rooted values) you would like to appear on
        the axis.
    """
    if linear_tick_values is None:
        if pr_on_x:
            ticks = ax.get_xticks()
        else:
            ticks = ax.get_yticks()
        linear_tick_values = np.array(ticks)**4
    else:
        ticks = np.array(linear_tick_values)**.25
    if pr_on_x:
        ax.set_xticks(ticks)
        ax.set_xticklabels(linear_tick_values)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(linear_tick_values)


def hot_pressure_plots(t_data, p_data, figsize=(6,8), mask_x_below_zero=False, mask_x_below_zero=False):
    
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True, sharey=True)
    
    axes[0, -1].axis('off')
    axes_used = axes.ravel()[[0,2,3,4,5]]
    dataset_labels = ['ERA5', 'HadGEM3', 'UNIT', 'UNIT+QM', 'QM']
    
    xlim = (15, 36)
    ylim = (5640, 5920)

    for t, p, ax, label in zip(t_data, p_data, axes_used, dataset_labels):
        t = t.values.ravel() - 273.15
        p = p.values.ravel()

        ax.annotate(label, xy=(0.05, 0.9), xycoords='axes fraction')
        
        ax.plot(t, p, linestyle='', marker='.', alpha=0.2, markersize=2, zorder=-1);
        sns.kdeplot(t, p, ax=ax, color='k', linewidths=1, alpha=.7)
        
        if mask_x_below_zero and  ylim[0]<0:
            # mask x values below zero
            rect1 = patches.Rectangle((xlim[0]-1,ylim[0]-1), 1-xlim[0], 2+ylim[1]-ylim[0], color='white', zorder=2)
            ax.add_patch(rect1)
        if mask_y_below_zero and ylim[0]<0:
            # mask y values below zero
            rect1 = patches.Rectangle((xlim[0]-1,ylim[0]-1), 2+xlim[1]-xlim[0], 1-ylim[0], color='white', zorder=2)
            ax.add_patch(rect2)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    for ax in axes[:,0]:
        ax.set_ylabel('500hPa geopotential height (m)')
    for ax in axes[-1,:]:
        ax.set_xlabel('Grid point temperature (°C)')

t_data = [t_era, t_era, t_had, t_trans, t_combo, t_qm]
p_data = [p_era, p_era, p_had, p_trans, p_combo, p_qm]
hot_pressure_plots(t_data, p_data)
plt.savefig("india_hotdays_joint_TP.png", dpi=300)