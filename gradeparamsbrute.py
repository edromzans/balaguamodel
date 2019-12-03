def plot_results_brute(result, best_vals=True, varlabels=None,
                       output=None):
    """Visualize the result of the brute force grid search.

    The output file will display the chi-square value per parameter and contour
    plots for all combination of two parameters.

    Inspired by the `corner` package (https://github.com/dfm/corner.py).

    Parameters
    ----------
    result : :class:`~lmfit.minimizer.MinimizerResult`
        Contains the results from the :meth:`brute` method.

    best_vals : bool, optional
        Whether to show the best values from the grid search (default is True).

    varlabels : list, optional
        If None (default), use `result.var_names` as axis labels, otherwise
        use the names specified in `varlabels`.

    output : str, optional
        Name of the output PDF file (default is 'None')
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatter, LogLocator

    npars = len(result.var_names)
    fig, axes = plt.subplots(npars, npars)

    if not varlabels:
        varlabels = result.var_names
    if best_vals and isinstance(best_vals, bool):
        best_vals = result.params

    for i, par1 in enumerate(result.var_names):
        for j, par2 in enumerate(result.var_names):

            # parameter vs chi2 in case of only one parameter
            if npars == 1:
                axes.plot(result.brute_grid, result.brute_Jout, '+', ms=3)
                axes.set_ylabel(r'$\chi^{2}$')
                axes.set_xlabel(varlabels[i])
                if best_vals:
                    axes.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parameter vs chi2 profile on top
            elif i == j and j < npars-1:
                if i == 0:
                    axes[0, 0].axis('off')
                ax = axes[i, j+1]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(np.unique(result.brute_grid[i]),
                        np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        '+', ms=3)
                ax.set_ylabel(r'$\chi^{2}$')
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position('right')
                ax.set_xticks([])
                # Evandro
                # plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 6))
                #  ------------------------------------
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parameter vs chi2 profile on the left
            elif j == 0 and i > 0:
                ax = axes[i, j]
                red_axis = tuple([a for a in range(npars) if a != i])
                print(red_axis, ' <<<<<<<<<<<<<<<<<<')
                ax.plot(np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        np.unique(result.brute_grid[i]), '+', ms=3)
                ax.invert_xaxis()
                ax.set_ylabel(varlabels[i])
                if i != npars-1:
                    ax.set_xticks([])
                elif i == npars-1:
                    ax.set_xlabel(r'$\chi^{2}$')
                if best_vals:
                    ax.axhline(best_vals[par1].value, ls='dashed', color='r')

            # contour plots for all combinations of two parameters
            elif j > i:
                ax = axes[j, i+1]
                red_axis = tuple([a for a in range(npars) if a != i and a != j])
                X, Y = np.meshgrid(np.unique(result.brute_grid[i]),
                                   np.unique(result.brute_grid[j]))
                # lvls1 = np.linspace(result.brute_Jout.min(),
                #                     np.median(result.brute_Jout)/2.0, 30, dtype='int')
                # lvls2 = np.linspace(np.median(result.brute_Jout)/2.0,
                #                     np.median(result.brute_Jout), 5, dtype='int')
                # lvls = np.unique(np.concatenate((lvls1, lvls2)))
                
                # lvls = np.linspace(result.brute_Jout.min(),
                #                   result.brute_Jout.max(), 255, dtype='int')
                lvls = np.linspace(result.brute_Jout.min(),
                                   np.median(result.brute_Jout), 150, dtype='int')
                print('------> red_axis ----->', red_axis)
                cplot = ax.contourf(
                    X.T, Y.T, np.minimum.reduce(
                        result.brute_Jout, axis=red_axis),lvls, norm=LogNorm(),
                    cmap='viridis', extend='max')  # viridis 'YlGnBu_r'
                #cplot.cmap.set_over('cyan')
                # cplot = ax.contourf(
                #     X.T, Y.T, np.minimum.reduce(
                #         result.brute_Jout, axis=red_axis), lvls,
                #     cmap='viridis')  # viridis
                
                ax.set_yticks([])
                # print('-------------lvls---> ', lvls)
                # fig.colorbar(cplot, ax=ax, extend='max')
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='r')
                    ax.axhline(best_vals[par2].value, ls='dashed', color='r')
                    ax.plot(best_vals[par1].value, best_vals[par2].value, 'rs', ms=3)
                if j != npars-1:
                    ax.set_xticks([])
                elif j == npars-1:
                    ax.set_xlabel(varlabels[i])
                if j - i >= 2:
                    axes[i, j].axis('off')
                    
    # flog = LogFormatter(10, labelOnlyBase=False)
    # sbticks = np.arange(40, dtype=float)*100+300 
    cbar = fig.colorbar(cplot, ax=axes, orientation='horizontal',
                        fraction=0.03)  # format='%.0e' ticks=lvls
    # cbar.formatter = LogFormatter(base=10, minor_thresholds=(2, 1)) #LogLocator(subs='all')
    # cbar.formatter = LogLocator(subs='all')
    cbar.set_ticks(LogLocator(subs='all'))
    cbar.update_ticks()
    cbar.set_label(r'$\chi^{2}$')
    # cbar.set_ticks(LogLocator(subs='all'))
    # cbar.set_ticks(LogLocator(subs='all'))  # (base=10.,subs=(500.,1000.,1500.,2000.,2500.)))
    if output is not None:
        plt.savefig(output, format='png', dpi=300)
    plt.show(output)
