import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax.nn import relu


def plot_surf(R_pnts, R_scores, pnts_save, L_max, num_pnts):

    rng_max = R_pnts.max(axis=0)
    rng_min = R_pnts.min(axis=0)

    # Make data.
    X = jnp.arange(rng_min[0] - L_max, rng_max[0] + L_max, 0.25)
    Y = jnp.arange(rng_min[1] - L_max, rng_max[1] + L_max, 0.25)
    X, Y = jnp.meshgrid(X, Y)

    l_val = np.zeros(X.shape)
    for p, lam in zip(R_pnts, R_scores):
        l_val += lam * relu(1 - jnp.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2) / L_max)

    Z = l_val

    # Plot the surface.
    fig1, ax1 = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "3d"}, num="Error")
    surf1 = ax1.plot_surface(
        X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, zorder=2, alpha=0.8
    )

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax1.zaxis.set_major_locator(LinearLocator(10))

    # A StrMethodFormatter is used automatically
    ax1.zaxis.set_major_formatter("{x:.02f}")
    ax1.xaxis.set_label_text("X")
    ax1.yaxis.set_label_text("Y")

    # Add a color bar which maps values to colors.
    fig1.colorbar(surf1, shrink=0.5, aspect=5)

    fig2, ax2 = plt.subplots(figsize=(9, 9), num="2D Error")

    surf2 = ax2.contourf(
        X,
        Y,
        Z,
        70,
        cmap=cm.coolwarm,
    )

    bb = np.reshape(pnts_save, (-1, num_pnts, 3))
    colors = ["red", "green"]

    for i in range(len(bb[0])):
        ax2.plot(bb[:, i, 0], bb[:, i, 1], c=colors[i], marker="o", zorder=1)

    ax2.axis("equal")

    ax2.scatter(R_pnts[:, 0], R_pnts[:, 1], c="black", marker="o", zorder=1)

    fig2.colorbar(surf2, shrink=0.5, aspect=5)

    fig1.savefig("./output/gradient_3d")
    fig2.savefig("./output/gradient_2d")
    plt.show()


#######################################


def func_reach_score_plus(pnts, R_pnts, R_scores, L_max, L_orig=False):

    X = pnts[:, 0]
    Y = pnts[:, 1]

    l_1 = 0
    for p, r in zip(R_pnts, R_scores):
        l_1 += r * relu(1 - jnp.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2) / L_max)

    l_2 = -jnp.abs(
        (L_orig - jnp.sqrt((pnts[0, 0] - pnts[1, 0]) ** 2 + (pnts[0, 1] - pnts[1, 1]) ** 2))
        / L_orig
    )

    return 1.0 * sum(l_1) + 0.0 * l_2


def func_reach_score(pnts, R_pnts, R_scores, L_max, L_orig=False):

    l_1 = 0

    X = pnts[:, 0]
    Y = pnts[:, 1]

    for p, r in zip(R_pnts, R_scores):
        l_1 += r * relu(1 - jnp.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2) / L_max)

    return sum(l_1)


def func_reach_score2(pnts, R_pnts, R_scores, L_max):

    l_1 = 0

    X = pnts[:, 0]
    Y = pnts[:, 1]

    for p, r in zip(R_pnts, R_scores):
        l_1 += r * relu(1 - jnp.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2) / L_max)

    return l_1


def backtrack(R_pnts, R_scores, pnts, L_max, grad, flag=True):

    reach_scores = func_reach_score2(pnts, R_pnts, R_scores, L_max)
    reach_scores_new = func_reach_score2(pnts + grad, R_pnts, R_scores, L_max)

    print(
        "--btrack: alpha = (0.5)^0, OLD: {:.3f} - NEW: {:.3f}".format(
            sum(reach_scores), sum(reach_scores_new)
        )
    )

    alphas = np.ones(pnts.shape[0])
    cnts = np.zeros(pnts.shape[0])

    if flag:
        while np.any(reach_scores_new < reach_scores) or np.any(
            reach_scores_new == 0
        ):  # if 0, then it can go outside function range

            cnts = np.array(
                [cnt + bool for bool, cnt in zip(reach_scores_new < reach_scores, cnts)]
            )

            alphas = (1 / 2) ** cnts

            reach_scores_new = func_reach_score2(
                pnts + (grad.T * alphas).T, R_pnts, R_scores, L_max
            )

            print(
                "--btrack: alpha = (0.5)^{}, OLD: {:.3f} - NEW: {:.3f}".format(
                    cnts, sum(reach_scores), sum(reach_scores_new)
                )
            )

    return alphas


def backtrack2(R_pnts, R_scores, pnts, L_max, grad, L_orig, flag=True):

    reach_scores = func_reach_score_plus(pnts, R_pnts, R_scores, L_max, L_orig)
    reach_scores_new = func_reach_score_plus(pnts + grad, R_pnts, R_scores, L_max, L_orig)

    print(
        "--btrack: alpha = (0.5)^0, OLD: {:.3f} - NEW: {:.3f}".format(
            reach_scores, reach_scores_new
        )
    )

    alphas = np.ones(pnts.shape[0])
    cnts = np.zeros(pnts.shape[0])

    if flag:
        while (
            reach_scores_new < reach_scores or reach_scores_new == 0
        ):  # if 0, then it can go outside function range

            cnts += 1

            alphas = (1 / 2) ** cnts

            reach_scores_new = func_reach_score_plus(
                pnts + (grad.T * alphas).T, R_pnts, R_scores, L_max, L_orig
            )

            print(
                "--btrack: alpha = (0.5)^{}, OLD: {:.3f} - NEW: {:.3f}".format(
                    cnts, reach_scores, reach_scores_new
                )
            )

    return alphas


def main(R_pnts, R_scores, pnts, L_max):

    num_pnts = pnts.shape[0]
    pnts_save = np.hstack(
        (pnts, func_reach_score2(pnts, R_pnts, R_scores, L_max).reshape(num_pnts, 1))
    )

    reach_score_save = 0

    L_orig = jnp.linalg.norm(pnts[1] - pnts[0])

    for i in range(0, 50):

        print("\n\n ITER #{}".format(i + 1))
        print("pnt_start: \n{}".format(pnts))
        print("length: {:.3f}/{:.3f}".format(np.linalg.norm(pnts[1] - pnts[0]), L_orig))

        # 1. EVAL FUNC
        reach_score, grad = value_and_grad(func_reach_score_plus, (0))(
            pnts, R_pnts, R_scores, L_max, L_orig
        )
        print("lam(x,y) = {:.3f}\n grad:\n {}".format(reach_score, grad._value))

        # reach_score, grad = value_and_grad(func_reach_score,(0))(pnts,R_pnts,R_scores,L_max)
        # print("lam(x,y) = {:.3f}\n grad:\n {}".format(reach_score, grad._value))

        # 2. BACKTRACK
        # alphas = backtrack(R_pnts, R_scores, pnts, L_max, grad, flag=FALSE)
        alphas = backtrack2(R_pnts, R_scores, pnts, L_max, grad, L_orig, flag=True)
        pnts = pnts + (grad.T * alphas).T
        reach_score = func_reach_score2(pnts, R_pnts, R_scores, L_max)

        # 3. SAVE
        print("new points: \n{}".format(pnts))
        print("new length: {:.3f}/{:.3f}".format(np.linalg.norm(pnts[1] - pnts[0]), L_orig))

        pnt_save = np.hstack((pnts, reach_score.reshape(num_pnts, 1)))
        pnts_save = np.vstack((pnts_save, pnt_save))

        # 4. TERMINATION
        a = func_reach_score_plus(pnts, R_pnts, R_scores, L_max, L_orig)
        print("{:.3f} --> {:.3f}".format(a, reach_score_save))
        if abs(a - reach_score_save) < 0.0001:
            break
        else:
            reach_score_save = a

    # 5. PLOTS
    plot_surf(R_pnts, R_scores, pnts_save, L_max, num_pnts)


if __name__ == "__main__":

    # reach_points = jnp.array([
    #     [3.0,1.0], #0
    #     [5.0,1.0], #1
    #     [7.0,1.0], #2
    #     [3.0,3.0], #3
    #     [5.0,3.0], #4
    #     [7.0,3.0], #5
    #     [3.0,5.0], #6
    #     [5.0,5.0], #7
    #     [7.0,5.0], #8
    # ])

    # reach_scores = jnp.array([
    #     0.2,
    #     0.3,
    #     1.0,
    #     -0.2,
    #     0.7,
    #     0.9,
    #     0.5,
    #     0.7,
    #     0.0
    # ])

    # L_max = 2.5

    # structure_pnt = jnp.array([
    #     [4.7,1.8],
    #     [2.7,0.3],
    # ])

    reach_points = jnp.array(
        [
            [3.0, 1.0],  # 0
            [5.0, 1.0],  # 1
            [5.0, 3.0],  # 2
            [3.0, 3.0],  # 3
        ]
    )

    reach_scores = jnp.array(
        [
            0.5,
            -0.2,
            1.0,
            -0.2,
        ]
    )

    L_max = 2.5

    structure_pnt = jnp.array(
        [
            [4.7, 1.8],
        ]
    )

    main(reach_points, reach_scores, structure_pnt, L_max)
