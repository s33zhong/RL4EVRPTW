import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm, colormaps

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td, actions=None, ax=None, arr_col=None):
    if actions[0] != 0:
        actions = torch.cat((torch.tensor([0]), actions))
    num_routine = (actions == 0).sum().item() + 2
    base = colormaps["nipy_spectral"]
    color_list = base(np.linspace(0, 1, num_routine))
    cmap_name = base.name + str(num_routine)
    out = base.from_list(cmap_name, color_list, num_routine)

    if ax is None:
        # Create a plot of the nodes
        _, ax = plt.subplots()

    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)

    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]

    locs = td["locs"].clone()
    scale_demand = td["capacity"][0]
    demands = td["demand"] * scale_demand

    # add the depot at the first action and the end action
    actions = torch.cat([torch.tensor([0]), actions, torch.tensor([0])])

    # gather locs in order of action if available
    if actions is None:
        log.warning("No action in TensorDict, rendering unsorted locs")
    else:
        locs = locs

    min_vals = locs.min(dim=1, keepdim=True)[0]  # Keep dimension for broadcasting
    max_vals = locs.max(dim=1, keepdim=True)[0]

    locs_scaled = (locs - min_vals) / (max_vals - min_vals)

    # Cat the first node to the end to complete the tour
    x, y = locs[:, 0], locs[:, 1]
    num_station = td["stations"].size(0)
    
    # plot depot
    ax.scatter(
        locs[0, 0],
        locs[0, 1],
        edgecolors=cm.Set2(2),
        facecolors="none",
        s=500,
        linewidths=2,
        marker="s",
        alpha=1,
    )

    # plot visited nodes
    ax.scatter(
        x[1:-num_station],
        y[1:-num_station],
        edgecolors=cm.Set2(0),
        facecolors="none",
        s=500,
        linewidths=2,
        marker="o",
        alpha=1,
    )
    
    # plot station nodes
    ax.scatter(
    x[-num_station:],
    y[-num_station:],
    edgecolors=cm.Set2(1),
    facecolors="none",
    s=500,
    linewidths=2,
    marker="^",
    alpha=1,
)

    bbox = ax.get_window_extent()
    for i, (x, y) in enumerate(td['locs']):
        # Convert the data coordinates to pixel coordinates relative to the axes
        x_pixel, y_pixel = ax.transData.transform((x, y))
        
        # Normalize the pixel coordinates into the range [0, 1]
        x_norm = (x_pixel-bbox.x0) / bbox.width
        y_norm = (y_pixel-bbox.y0) / bbox.height
        
        # Place text at the normalized position (relative to the axes' size)
        if i <= td['demand'].shape[-1]:
            ax.text(x_norm, y_norm, i,
                    transform=ax.transAxes, fontsize=8, color='black', ha='center', va='center')
        else:
            ax.text(x_norm, y_norm, i,
                    transform=ax.transAxes, fontsize=8, color='black', ha='center', va='top')
    # plot demand bars
    # for node_idx in range(1, len(locs)-num_station):
    #     ax.add_patch(
    #         plt.Rectangle(
    #             (locs[node_idx, 0] - 0.005, locs[node_idx, 1] + 0.015),
    #             0.01,
    #             demands[node_idx - 1] / (scale_demand * 10),
    #             edgecolor=cm.Set2(0),
    #             facecolor=cm.Set2(0),
    #             fill=True,
    #         )
    #     )

    # # text demand
    # for node_idx in range(1, len(locs)-num_station):
    #     ax.text(
    #         locs[node_idx, 0],
    #         locs[node_idx, 1] - 0.025,
    #         f"{demands[node_idx-1].item():.2f}",
    #         horizontalalignment="center",
    #         verticalalignment="top",
    #         fontsize=10,
    #         color=cm.Set2(0),
    #     )

    # # text depot
    # ax.text(
    #     locs[0, 0],
    #     locs[0, 1] - 0.025,
    #     "Depot",
    #     horizontalalignment="center",
    #     verticalalignment="top",
    #     fontsize=10,
    #     color=cm.Set2(2),
    # )

    # plot actions arrow
    color_idx = 0
    for action_idx in range(1, len(actions) - 1):
        if actions[action_idx] == 0:
            color_idx += 1
        from_loc = locs[actions[action_idx]]
        to_loc = locs[actions[action_idx + 1]]
        vector = to_loc - from_loc

        # trim arrows
        if np.linalg.norm(vector)>0.075:    
            norm_vector = vector / np.linalg.norm(vector)
            from_loc = from_loc + norm_vector * 0.03
            to_loc = to_loc - norm_vector * 0.03
            
        # ax.plot(
        #     [from_loc[0], to_loc[0]],
        #     [from_loc[1], to_loc[1]],
        #     color=out(color_idx),
        #     lw=1,
        # )
        ax.annotate(
            "",
            xy=(to_loc[0], to_loc[1]),
            xytext=(from_loc[0], from_loc[1]),
            arrowprops=dict(arrowstyle="-|>", color=out(color_idx)),
            size=10,
            annotation_clip=False,
        )

    flattened_locs = td["locs"].view(-1, 2)

    min_x = flattened_locs[:, 0].min().item()
    max_x = flattened_locs[:, 0].max().item()
    min_y = flattened_locs[:, 1].min().item()
    max_y = flattened_locs[:, 1].max().item()

    plt.axis('off')
    
    # ax.set_xlim(min_x-0.05, max_x+0.05)
    # ax.set_ylim(min_y-0.05, max_y+0.05)
