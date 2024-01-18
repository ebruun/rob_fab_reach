import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy import interpolate


def interpolate_data(X, Y, Z1, X_finer, Y_finer, method="linear"):
    # Create a meshgrid for the original and finer X and Y
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    X_finer_mesh, Y_finer_mesh = np.meshgrid(X_finer, Y_finer)

    # Flatten the meshes for interpolation
    X_mesh_flat = X_mesh.flatten()
    Y_mesh_flat = Y_mesh.flatten()
    Z1_flat = Z1.flatten()

    # Interpolate Z1 data onto the finer grid
    Z_finer = interpolate.griddata(
        (X_mesh_flat, Y_mesh_flat), Z1_flat, (X_finer_mesh, Y_finer_mesh), method=method
    )

    return Z_finer


X = np.arange(2, 7.1, 1)
Y = np.arange(0.125, 1.01, 0.125)

Z1 = np.array(
    [
        [133, 149, 161, 165, 164, 149],
        [153, 167, 171, 171, 166, 143],
        [162, 170, 172, 168, 153, 105],
        [165, 171, 168, 149, 112, 57],
        [167, 169, 151, 114, 81, 39],
        [168, 162, 125, 93, 64, 32],
        [167, 147, 104, 80, 54, 28],
        [166, 128, 91, 71, 47, 25],
    ]
)

Z2 = np.array(
    [
        [99, 114, 118, 106, 90, 78],
        [110, 123, 126, 114, 93, 76],
        [114, 126, 134, 106, 72, 52],
        [116, 126, 127, 75, 37, 28],
        [115, 121, 104, 34, 21, 16],
        [113, 110, 73, 17, 13, 11],
        [108, 94, 52, 12, 10, 8],
        [102, 75, 42, 9, 7, 6],
    ]
)

# Define the finer X and Y grids
X_finer = np.arange(2, 7.000001, 1 / 100)
Y_finer = np.arange(0.125, 1.000001, 0.125 / 100)

# Interpolate the data
Z1_finer = interpolate_data(X, Y, Z1, X_finer, Y_finer)
Z2_finer = interpolate_data(X, Y, Z2, X_finer, Y_finer)

X, Y = np.meshgrid(X_finer, Y_finer)


# Create plots
fig, axs = plt.subplots(
    1,
    2,
)
fig.set_size_inches(20.5, 9.5)

# Surface
surf1 = axs[0].contourf(X, Y, Z1_finer, np.linspace(0, 200, 81), vmin=0, vmax=200, cmap=cm.coolwarm)
surf2 = axs[1].contourf(X, Y, Z2_finer, np.linspace(0, 200, 81), vmin=0, vmax=200, cmap=cm.coolwarm)

# Contours
contour1 = axs[0].contour(X, Y, Z1_finer, levels=[25, 50, 75, 100, 125, 150], colors="k")
contour2 = axs[1].contour(X, Y, Z2_finer, levels=[25, 50, 75, 100, 125], colors="k")

# Add points with labels to the surface
points_x = [4, 4]  # X-coordinates of points
points_y = [0.375, 0.375]  # Y-coordinates of points
point_labels = ["max:172", "max:134"]  # Labels for the points

point1 = axs[0].scatter(points_x[0], points_y[0], color="black", marker="o", label="Points")
point2 = axs[1].scatter(points_x[1], points_y[1], color="black", marker="o", label="Points")

for ax in axs:
    ax.set_xlabel("$Span (m)$", fontsize=12)
    ax.set_ylabel("$Rise-to-Span$", fontsize=12)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.125))
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")


# Add labels to the contour lines within subplots
axs[0].clabel(contour1, inline=1, fontsize=10, fmt="%1.0f")
axs[1].clabel(contour2, inline=1, fontsize=10, fmt="%1.0f")

# Add labels to the cmax points
axs[0].annotate(
    point_labels[0], (points_x[0], points_y[0] - 0.02), fontsize=9, color="black", ha="center"
)
axs[1].annotate(
    point_labels[1], (points_x[1], points_y[1] - 0.02), fontsize=9, color="black", ha="center"
)

axs[0].set_title("Medium-Payload Robot on Track Setup", fontsize=13, y=-0.05)
axs[1].set_title("High-Payload Fixed Robot Setup", fontsize=13, y=-0.05)


# Add a color bar which maps values to colors.
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(
    surf1,
    cax=cbar_ax,
    ticks=range(0, 201, 25),
    label="Fab. Score",
)


label = cbar.ax.yaxis.label
# Set the label position on top of the colorbar
label.set_rotation(0)
label.set_verticalalignment("bottom")  # Align the label to the bottom of the colorbar
label.set_horizontalalignment("center")  # Align the label to the center of the colorbar
label.set_position((0.0, 1.02))  # Set the position relative to the colorbar

# Adjust the labelpad to control the distance between the label and the colorbar
plt.show()
fig.savefig("reachability_contours", dpi=300)
