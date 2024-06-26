import sys

import numpy as np
import yt

import yt_idv

# yt reminder: phi is the polar angle (0 to 2pi)
# theta is the angle from north (0 to pi)


# coord ordering here will be r, phi, theta

bbox_options = {
    "partial": np.array([[0.5, 1.0], [0.0, np.pi / 4], [np.pi / 4, np.pi / 2]]),
    "whole": np.array([[0.1, 1.0], [0.0, 2 * np.pi], [0, np.pi]]),
    "north_hemi": np.array([[0.1, 1.0], [0.0, 2 * np.pi], [0, 0.5 * np.pi]]),
    "south_hemi": np.array([[0.1, 1.0], [0.0, 2 * np.pi], [0.5 * np.pi, np.pi]]),
    "ew_hemi": np.array([[0.1, 1.0], [0.0, np.pi], [0.0, np.pi]]),
}

bbox = bbox_options['whole']
sz = (256, 256, 256)
fake_data = {"density": np.random.random(sz)}

def _neato(field, data):
    r = data['index', 'r'].d
    theta = data['index', 'theta'].d
    phi = data['index', 'phi'].d
    phi_c = 0.25 * np.pi
    theta_c = 0.5 * np.pi

    # decay away from phi_c, theta_c
    fac = np.exp(-((phi_c - phi) / 0.5) ** 2) * np.exp(-((theta_c - theta) / 0.5) ** 2)
    # cos^2 variation in r with slight increase towards rmin
    rfac = np.cos((r - 0.1)/0.9 * 3 * np.pi)**2 * (1 - 0.25 * (r - 0.1)/0.9)
    field = fac * rfac + 0.1 * np.random.random(r.shape)

    # field = field * (theta <= 2.0) * (phi < 1.25)
    return field

yt.add_field(
    name=("stream", "neat"),
    function=_neato,
    sampling_type="local",
    units="",
)


ds = yt.load_uniform_grid(
    fake_data,
    sz,
    bbox=bbox,
    nprocs=4096,
    geometry=("spherical", ("r", "phi", "theta")),
    length_unit="m",
)

rc = yt_idv.render_context(height=800, width=800, gui=True)
dd = ds.all_data()
dd.max_level = 1
sg = rc.add_scene(ds, ("stream", "neat"), no_ghost=True)
# sg = rc.add_scene(ds, ("index", "theta"), no_ghost=True)
# sg = rc.add_scene(ds, ("index", "phi"), no_ghost=True)
sg.camera.focus = [0.0, 0.0, 0.0]
rc.run()
