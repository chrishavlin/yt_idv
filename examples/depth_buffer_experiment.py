import numpy as np
import yt
from scipy.ndimage import gaussian_filter

import yt_idv


def some_spheres():
    dim = np.linspace(0, 1, 100)
    x, y, z = np.meshgrid(dim, dim, dim, indexing="ij")

    centers = [(0.5, 0.25, 0.25), (0.5, 0.75, 0.75)]
    decay = [0.1, 0.25]
    amplitudes = [1.0, 1.0]

    density = np.zeros(x.shape)

    for c, d, amp in zip(centers, decay, amplitudes):
        dist = (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2
        density += amp * np.exp(-dist / (d ** 2))

    data = {"density": (density, "")}
    ds = yt.load_uniform_grid(data, x.shape, geometry=("cartesian", "xyz"))
    dd = ds.all_data()
    dd.max_level = 1
    return ds


def some_cubes(smo_iters: int = 0):
    dim = np.linspace(0, 1, 100)
    x, y, z = np.meshgrid(dim, dim, dim, indexing="ij")

    centers = [
        (0.5, 0.25, 0.25),
        (0.5, 0.25, 0.25),
        (0.5, 0.75, 0.75),
        (0.5, 0.75, 0.75),
        (0.5, 0.75, 0.75),
    ]
    wids = [0.075, 0.05, 0.15, 0.1, 0.05]
    amplitudes = [1.0, 2.0, 1.0, 3.0, 10.0]

    density = np.full(x.shape, np.min(amplitudes) / 10.0)

    for c, wid, amp in zip(centers, wids, amplitudes):
        xmask = (x >= c[0] - wid) & (x <= c[0] + wid)
        ymask = (y >= c[1] - wid) & (y <= c[1] + wid)
        zmask = (z >= c[2] - wid) & (z <= c[2] + wid)
        full_mask = xmask * ymask * zmask
        outside = ~full_mask
        density = full_mask * amp + outside * density

    # optional smoothing iters
    for _ in range(smo_iters):
        density = gaussian_filter(density, 1.0)
    data = {"density": (density, "")}
    ds = yt.load_uniform_grid(data, x.shape, geometry=("cartesian", "xyz"))
    dd = ds.all_data()
    dd.max_level = 1
    return ds


# ds = some_spheres()
ds = some_cubes(smo_iters=1)
rc = yt_idv.render_context(height=800, width=800, gui=True)
sg = rc.add_scene(ds, ("stream", "density"), no_ghost=True)
rc.run()
