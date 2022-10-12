import numpy as np
import scipy.sparse
from scipy.signal import savgol_filter


def smooth_1D_data(data, kernel_size=51, polynomial_order=3):
    return savgol_filter(data, kernel_size, polynomial_order)


def laplacian_smoothing(data, iters=100):
    n = len(data)
    L = scipy.sparse.diags(
        [0.25, 0.5, 0.25], [-1, 0, 1], shape=(n, n), format="lil")
    L[[0, -1], :] = 0
    L[[0, -1], [0, -1]] = 1
    L = L.tocsc()
    for i in range(iters):
        data = L @ data
    return data


def compute_scale_and_shift_to_fit_mesh(V):
    if V.shape[0] == 0:
        return 1.0, np.zeros(3)
    min_point = V.min(axis=0)
    max_point = V.max(axis=0)
    centroid = (min_point + max_point) / 2
    shift = -centroid
    zoom = 2 / abs(max_point - min_point).max()
    return zoom, shift


def plot_zooms_and_shifts(zooms, shifts):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(ncols=2)

    x = np.arange(zooms.size)
    ax[0].plot(x, zooms)
    ax[0].plot(x, smooth_1D_data(zooms))
    ax[0].set(ylabel='zoom', title='Zoom')
    ax[0].grid()

    def relative_shift(shift):
        return np.linalg.norm(shift[1:] - shift[:-1], axis=1)
    rel_shifts = relative_shift(shifts)
    ax[1].plot(x[1:], rel_shifts)
    ax[1].plot(x[1:], smooth_1D_data(rel_shifts))
    ax[1].plot(x[1:], relative_shift(laplacian_smoothing(shifts)))
    ax[1].set(ylabel='relative shift', title='Relative Shift')
    ax[1].grid()

    fig.savefig("zoom_and_shift.png")


def compute_dynamic_scale_and_shift_to_fit_meshes(meshes):
    zooms_and_shifts = [
        # compute_scale_and_shift_to_fit_mesh(mesh.points[~mesh.point_data["is_obstacle"]]) for mesh in meshes
        compute_scale_and_shift_to_fit_mesh(mesh.points) for mesh in meshes
    ]

    zooms_and_shifts = (
        np.vstack([np.hstack(zs) for zs in zooms_and_shifts]))

    plot_zooms_and_shifts(
        zooms_and_shifts[:, 0], zooms_and_shifts[:, 1:])

    zooms_and_shifts[:, 0] = smooth_1D_data(zooms_and_shifts[:, 0])
    zooms_and_shifts[:, 1:] = laplacian_smoothing(zooms_and_shifts[:, 1:])

    return [[zoom, shift] for zoom, *shift in zooms_and_shifts]


def compute_zooms_and_shifts(meshes, camera_mode):
    if camera_mode == "static":
        return [
            compute_scale_and_shift_to_fit_mesh(meshes[0].points)
        ] * len(meshes)
    elif camera_mode == "dynamic":
        return compute_dynamic_scale_and_shift_to_fit_meshes(
            meshes)
    else:
        assert(camera_mode == "global")
        global_V = np.vstack([mesh.points for mesh in meshes])
        return [
            compute_scale_and_shift_to_fit_mesh(global_V)
        ] * len(meshes)
