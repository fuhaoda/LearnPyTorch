
from matplotlib import pyplot as plt
import numpy as np

def plot_squares(points, directions, n_rows=2, n_cols=5):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axs = axs.flatten()
    
    for e, ax in enumerate(axs):
        pred_corners = points[e]
        clockwise = directions[e]
        for i in range(4):
            color = 'k'
            ax.scatter(*pred_corners.T, c=color, s=400)
            if i == 3:
                start = -1
            else:
                start = i
            ax.plot(*pred_corners[[start, start+1]].T, c='k', lw=2, alpha=.5, linestyle='-')
            ax.text(*(pred_corners[i] - np.array([.04, 0.04])), str(i+1), c='w', fontsize=12)
            if directions is not None:
                ax.set_title(f'{"Counter-" if not clockwise else ""}Clockwise (y={clockwise})', fontsize=14)

        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$x_1$", rotation=0)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])

    fig.tight_layout()
    return fig

def counter_vs_clock(basic_corners=None, basic_colors=None, basic_letters=None, draw_arrows=True, binary=True):
    transparent_alpha = 0.2
    if basic_corners is None:
        basic_corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
        clock_arrows = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])
    else:
        clock_arrows = np.array([[0, basic_corners[0][1]], [basic_corners[1][0], 0], 
                                 [0, basic_corners[2][1]], [basic_corners[3][0], 0]])
        
    if basic_colors is None:
        basic_colors = ['gray', 'g', 'b', 'r']
    if basic_letters is None:
        basic_letters = ['A', 'B', 'C', 'D']        
    
    fig, axs = plt.subplots(1 + draw_arrows, 1, figsize=(3, 3+ 3 * draw_arrows))
    if not draw_arrows:
        axs = [axs]

    corners = basic_corners[:]
    factor = (corners.max(axis=0) - corners.min(axis=0)).max() / 2

    for is_clock in range(1 + draw_arrows):
        if draw_arrows:
            if binary:
                if is_clock:
                    axs[is_clock].text(-.5, 0, 'Clockwise')
                    axs[is_clock].text(-.2, -.25, 'y=1')
                else:
                    axs[is_clock].text(-.5, .0, ' Counter-\nClockwise')
                    axs[is_clock].text(-.2, -.25, 'y=0')

        for i in range(4):
            coords = corners[i]
            color = basic_colors[i]
            letter = basic_letters[i]
            if not binary:
                targets = [2, 3] if is_clock else [1, 2]
            else:
                targets = []
            
            alpha = transparent_alpha  if i in targets else 1.0
            axs[is_clock].scatter(*coords, c=color, s=400, alpha=alpha)

            start = i
            if is_clock:
                end = i + 1 if i < 3 else 0
                arrow_coords = np.stack([corners[start] - clock_arrows[start]*0.15,
                                      corners[end] + clock_arrows[start]*0.15])
            else:
                end = i - 1 if i > 0 else -1
                arrow_coords = np.stack([corners[start] + clock_arrows[end]*0.15,
                                      corners[end] - clock_arrows[end]*0.15])
            alpha = 1.0
            if draw_arrows:
                alpha = transparent_alpha if ((start in targets) or (end in targets)) else 1.0
            line = axs[is_clock].plot(*arrow_coords.T, c=color, lw=0 if draw_arrows else 2, 
                                      alpha=alpha, linestyle='--' if (alpha < 1) and (not draw_arrows) else '-')[0]
            if draw_arrows:
                add_arrow(line, lw=3, alpha=alpha)

            axs[is_clock].text(*(coords - factor*np.array([.05, 0.05])), letter, c='k' if i in targets else 'w', fontsize=12, alpha=transparent_alpha  if i in targets else 1.0)

            axs[is_clock].grid(False)
            limits = np.stack([corners.min(axis=0), corners.max(axis=0)])
            limits = limits.mean(axis=0).reshape(2, 1) + 1.2*np.array([[-factor, factor]])
            axs[is_clock].set_xlim(limits[0])
            axs[is_clock].set_ylim(limits[1])
            
            axs[is_clock].set_xlabel(r'$x_0$')
            axs[is_clock].set_ylabel(r'$x_1$', rotation=0)

    fig.tight_layout()
    
    return fig

def sequence_pred(trainer_obj, X, directions=None, n_rows=2, n_cols=5):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axs = axs.flatten()

    for e, ax in enumerate(axs):
        first_corners = X[e, :2, :]
        trainer_obj.model.eval()
        next_corners = trainer_obj.model(X[e:e+1, :2].to(trainer_obj.device)).squeeze().detach().cpu().numpy()
        pred_corners = np.concatenate([first_corners, next_corners], axis=0)

        for j, corners in enumerate([X[e], pred_corners]):
            for i in range(4):
                coords = corners[i]
                color = 'k'
                ax.scatter(*coords, c=color, s=400)
                if i == 3:
                    start = -1
                else:
                    start = i
                if (not j) or (j and i):
                    ax.plot(*corners[[start, start+1]].T, c='k', lw=2, alpha=.5, linestyle='--' if j else '-')
                ax.text(*(coords - np.array([.04, 0.04])), str(i+1), c='w', fontsize=12)
                if directions is not None:
                    ax.set_title(f'{"Counter-" if not directions[e] else ""}Clockwise')

        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$x_1$", rotation=0)
        ax.set_xlim([-1.7, 1.7])
        ax.set_ylim([-1.7, 1.7])        

    fig.tight_layout()
    return fig


def add_arrow(line, position=None, direction='right', size=15, color=None, lw=2, alpha=1.0, text=None, text_offset=(0 , 0)):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle='--' if alpha < 1 else '-', alpha=alpha),
        size=size,
    )
    if text is not None:
        line.axes.annotate(text, color=color,
            xytext=(xdata[end_ind] + text_offset[0], ydata[end_ind] + text_offset[1]),
            xy=(xdata[end_ind], ydata[end_ind]),
            size=size,
        )



# plot the clock and counter-clock sequence with the unknowns gray out
counter_vs_clock(binary=False)
