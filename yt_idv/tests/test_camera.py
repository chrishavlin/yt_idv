import numpy as np
import pytest

from yt_idv.cameras.trackball_camera import TrackballCamera


@pytest.fixture()
def camera():
    cam = TrackballCamera(
        position=np.array([0.5, 0.5, 2.5]), focus=np.array([0.5, 0.5, 0.5])
    )
    cam._update_matrices()
    return cam


@pytest.mark.parametrize(
    "trait, value",
    [("fov", 20.0), ("near_plane", 0.5), ("far_plane", 5.0), ("aspect_ratio", 2.0)],
)
def test_projection_traits_rebuild_projection(camera, trait, value):
    p0 = camera.projection_matrix.copy()
    v0 = camera.view_matrix.copy()
    o0 = camera.orientation.copy()

    setattr(camera, trait, value)

    assert not np.allclose(p0, camera.projection_matrix)
    assert np.array_equal(v0, camera.view_matrix)
    assert np.array_equal(o0, camera.orientation)


def test_trait_set_after_drag_keeps_trackball_rotation(camera):
    camera.update_orientation(0.0, 0.0, 0.3, 0.2)
    v0 = camera.view_matrix.copy()
    o0 = camera.orientation.copy()
    pos0 = camera.position.copy()

    camera.fov = 30.0
    camera.near_plane = 0.1

    assert np.array_equal(v0, camera.view_matrix)
    assert np.array_equal(o0, camera.orientation)
    assert np.array_equal(pos0, camera.position)


def test_position_moves_leave_orientation_alone(camera):
    camera.update_orientation(0.0, 0.0, 0.3, 0.2)
    o0 = camera.orientation.copy()

    camera.move_forward(0.1)
    assert np.array_equal(o0, camera.orientation)

    camera.offset_position(np.array([0.05, 0.0, 0.0]))
    assert np.array_equal(o0, camera.orientation)


def test_update_rebuilds_projection_once(camera):
    calls = []
    original = camera._compute_matrices

    def counting():
        calls.append(1)
        original()

    camera._compute_matrices = counting
    p0 = camera.projection_matrix.copy()

    camera.update(fov=30.0, near_plane=0.1, far_plane=10.0)

    assert len(calls) == 1
    assert not np.allclose(p0, camera.projection_matrix)
    assert camera.fov == 30.0 and camera.near_plane == 0.1 and camera.far_plane == 10.0
    assert not camera.held
