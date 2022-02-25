import yt
import numpy as np
import yt_idv


def some_spheres():
    dim = np.linspace(0, 1, 100)
    x, y, z = np.meshgrid(dim, dim, dim, indexing="ij")

    centers = [(0.5, 0.25, 0.25),
               (0.5, 0.75, 0.75)]
    decay = [0.1, 0.25]
    amplitudes = [1., 1.]

    density = np.zeros(x.shape)

    for c, d, amp in zip(centers, decay, amplitudes):
        dist = (x - c[0])**2 + (y - c[1])**2 + (z - c[2])**2
        density += amp * np.exp(-dist / (d**2))

    data = {"density": (density, "")}
    return yt.load_uniform_grid(data, x.shape, geometry=("cartesian", "xyz"), nprocs=4096)


def some_cubes():
    dim = np.linspace(0, 1, 100)
    x, y, z = np.meshgrid(dim, dim, dim, indexing="ij")

    centers = [(0.5, 0.25, 0.25),
               (0.5, 0.25, 0.25),
               (0.5, 0.75, 0.75),
               (0.5, 0.75, 0.75),
               (0.5, 0.75, 0.75)]
    wids = [0.075, 0.05, 0.15, 0.1, 0.05]
    amplitudes = [1., 2., 1., 3., 10.]

    density = np.full(x.shape, np.min(amplitudes)/10.)

    for c, wid, amp in zip(centers, wids, amplitudes):
        xmask = (x >= c[0] - wid) & (x <= c[0] + wid)
        ymask = (y >= c[1] - wid) & (y <= c[1] + wid)
        zmask = (z >= c[2] - wid) & (z <= c[2] + wid)
        full_mask = xmask * ymask * zmask
        outside = ~full_mask
        density = full_mask * amp + outside * density

    data = {"density": (density, "")}
    return yt.load_uniform_grid(data, x.shape, geometry=("cartesian", "xyz"), nprocs=4096,)

# ds = some_spheres()
ds = some_cubes()


#
slc = yt.SlicePlot(ds, "x", ("stream", "density"), origin="native")
slc.set_log(("stream", "density"), False)
slc.save("/tmp/whatever.png")



rc = yt_idv.render_context(height=800, width=800, gui=True)
sg = rc.add_scene(ds, ("stream", "density"), no_ghost=True)
rc.run()
