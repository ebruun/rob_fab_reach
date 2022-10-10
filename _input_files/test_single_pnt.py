import jax.numpy as jnp

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
        0.5,  # 0
        -0.2,  # 1
        1.0,  # 2
        -0.2,  # 3
    ]
)

L_max = 2.5

structure_pnt = jnp.array(
    [
        [4.7, 0.7],
    ]
)
