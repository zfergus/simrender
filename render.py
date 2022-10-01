# Top of main python script
import os
if 'PYOPENGL_PLATFORM' not in os.environ:  # noqa
    os.environ['PYOPENGL_PLATFORM'] = 'egl'  # noqa
print("Using PYOPENGL_PLATFORM={}".format(os.environ['PYOPENGL_PLATFORM']))  # noqa

import argparse
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.sparse
from scipy.signal import savgol_filter

from matplotlib import cm
from tqdm import tqdm
import imageio.v3 as iio
from natsort import natsorted

import trimesh
import meshio
import igl

import pyrender
from pyrender import RenderFlags as rf


def homogeneous(a):
    assert(len(a) == 3)
    return np.array([*a, 1.0], dtype=float)


def scale(s):
    if isinstance(s, (int, float)):
        return np.diag([s, s, s, 1.0])
    assert(len(s) == 3)
    return np.diag([*s, 1.0])


def translate(t):
    T = np.eye(4, dtype=float)
    T[:3, 3] = t
    return T


def align_vectors(a, b):
    """Create a rotation matrix that aligns vector a to vector b."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.align_vectors(
        np.array(b).reshape(1, -1),
        np.array(a).reshape(1, -1)
    )[0].as_matrix()
    return T


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


def create_default_scene(aspectRatio, bg_color=None, camera_height=0.0):
    scene = pyrender.Scene(
        nodes=None,
        bg_color=[0.3, 0.3, 0.5, 1.0] if bg_color is None else bg_color,
        ambient_light=np.full((3,), 0.3)
    )

    key_intensity = 60.0
    lights = [{
        "position": np.array([-3.0, 3.0, 3.0]),
        "target": np.array([0.0, 0.5, 0.0]),
        "intensity": key_intensity,
        "innerConeAngle": np.pi / 8.0,
        "outerConeAngle": np.pi / 4.0,
    }, {
        "position": np.array([3.0, 3.0, 3.0]),
        "target": np.array([-0.1, 0.5, 0.0]),
        "intensity": 0.6 * key_intensity,
        "innerConeAngle": np.pi / 6.0,
        "outerConeAngle": np.pi / 3.0,
    }, {
        "position": np.array([-1.5, 3.5, -3.0]),
        "target": np.array([0.0, 0.0, 0.0]),
        "intensity": 0.3 * key_intensity,
        "innerConeAngle": np.pi / 4.0,
        "outerConeAngle": np.pi / 2.0,
    }]

    for light in lights:
        spot_light = pyrender.SpotLight(
            color=np.ones(3),
            intensity=light["intensity"],
            innerConeAngle=light["innerConeAngle"],
            outerConeAngle=light["outerConeAngle"],
        )
        scene.add(
            spot_light,
            pose=(translate(light["position"])
                  @ align_vectors([0, 0, -1], light["target"] - light["position"]))
        )

    camera_position = np.array([0.0, camera_height, 3.0])
    # camera_position = np.array([0.0, 1.5, 2.65])
    camera_target = np.array([0.0, camera_height, 0.0])
    scene.add(
        pyrender.PerspectiveCamera(
            yfov=np.deg2rad(45),
            znear=0.1,
            zfar=100,
            aspectRatio=aspectRatio
        ),
        pose=(translate(camera_position) @
              align_vectors([0, 0, -1], camera_target - camera_position))
    )

    return scene


def render(mesh, width=800, height=600, bg_color=None, wireframe=True, smooth=False,
           camera_height=0.0, base_zoom=1.0, tensor_field=None, zoom=None, shift=None):
    scene = create_default_scene(
        aspectRatio=width/height, bg_color=bg_color, camera_height=camera_height)

    if zoom is None or shift is None:
        zoom, shift = compute_scale_and_shift_to_fit_mesh(mesh.points)
    mesh_pose = scale(base_zoom * zoom) @ translate(shift)

    # scene.add(pyrender.Mesh.from_trimesh(
    #     trimesh.creation.axis(), smooth=False), pose=mesh_pose)

    if tensor_field is not None:
        arrow = trimesh.creation.revolve(np.array([
            [0.0, 0.0],
            [0.05, 0.0],
            [0.05, 0.7],
            [0.1,  0.7],
            [0.0, 1.0],
        ]))
        scene.add(
            pyrender.Mesh.from_trimesh(arrow, smooth=True, poses=[
                mesh_pose
                @ translate(v)
                @ align_vectors([0, 0, 1], mesh.point_data[tensor_field][i])
                @ scale([1, 1, np.linalg.norm(mesh.point_data[tensor_field][i])])
                @ scale(0.25)
                for i, v in enumerate(mesh.points)
                if not mesh.point_data["is_obstacle"][i]
            ]))

    for cells in mesh.cells:
        if cells.type == "triangle":
            NV, NF, IM, JM = igl.remove_unreferenced(mesh.points, cells.data)
            tmesh = trimesh.Trimesh(NV, NF)
            vertex_colors = mesh.point_data["colors"][JM]
            assert(tmesh.vertices.shape[0] == vertex_colors.shape[0])
            tmesh.visual.vertex_colors = vertex_colors

            mesh_args = dict(
                is_visible=True,
                poses=mesh_pose,
                smooth=smooth
            )

            scene.add(pyrender.Mesh.from_trimesh(
                tmesh,
                # material=pyrender.Material(
                #     doubleSided=True,
                # ),
                **mesh_args))

            if wireframe:
                VN = trimesh.geometry.weighted_vertex_normals(
                    tmesh.vertices.shape[0], tmesh.faces, tmesh.face_normals, tmesh.face_angles)
                tmesh.vertices += 1e-3 * VN
                scene.add(pyrender.Mesh.from_trimesh(
                    tmesh,
                    material=pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=[0.0, 0.0, 0.0, 1.0],
                    ),
                    wireframe=True,
                    **mesh_args
                ))
        elif cells.type == "line":
            capsule = trimesh.creation.capsule(
                height=1.0, radius=5e-3, count=[4, 4])
            capsule.visual.vertex_colors = [0.0, 0.0, 0.0]
            poses = []
            for edge in cells.data:
                v1, v2 = mesh.points[edge]
                direction = v2 - v1
                length = np.linalg.norm(direction)
                poses.append(
                    mesh_pose
                    @ translate(v1)
                    # cylinder is along z-axis
                    @ align_vectors([0.0, 0.0, 1.0], direction)
                    @ scale([1, 1, length])
                )
            scene.add(pyrender.Mesh.from_trimesh(capsule, poses=poses))
        elif cells.type == "vertex":
            sphere = trimesh.creation.uv_sphere(radius=2.5e-2, count=(4, 4))
            sphere.visual.vertex_colors = [0.0, 0.0, 0.0]
            poses = [mesh_pose @ translate(p) for p in mesh.points[cells.data]]
            scene.add(pyrender.Mesh.from_trimesh(sphere, poses=poses))
        else:
            raise NotImplementedError(f"Unsupported cell type: {cells.type}")

    r = pyrender.OffscreenRenderer(
        viewport_width=width, viewport_height=height, point_size=1.0)
    color, depth = r.render(
        scene, flags=rf.RGBA
        | rf.FACE_NORMALS
        | rf.SHADOWS_DIRECTIONAL
        # | rf.SHADOWS_POINT
        # | rf.FLAT
        | rf.SHADOWS_SPOT
    )
    r.delete()
    return color, depth


def fix_normals(V, F):
    mesh = trimesh.Trimesh(V, F, process=False, validate=False)
    trimesh.repair.fix_normals(mesh)
    return mesh.faces


def load_mesh(path):
    mesh = meshio.read(path)

    V, I, J, _ = igl.remove_duplicate_vertices(
        mesh.points, np.array([], dtype=int), 1e-7)

    CV = []  # codim vertices
    E = []  # edges
    F = []  # triangles
    for cells in mesh.cells:
        if cells.type == "triangle":
            F.append(J[cells.data])
        elif cells.type == "tetra":
            F.append(fix_normals(V, igl.boundary_facets(J[cells.data])))
        elif cells.type == "line":
            E.append(J[cells.data])
        elif cells.type == "vertex":
            CV.append(J[cells.data])
        else:
            raise Exception("Unsupported cell type: {}".format(cells.type))

    cells = []
    if F:
        cells.append(("triangle", np.vstack(F)))
    if E:
        cells.append(("line", np.vstack(E)))
    if CV:
        cells.append(("vertex", np.vstack(CV)))

    if "solution" in mesh.point_data:
        V += mesh.point_data["solution"][I]

    point_data = dict((k, v[I]) for k, v in mesh.point_data.items())
    if "E" in point_data:
        point_data["is_obstacle"] = (point_data["E"] == 0).flatten()
    else:
        point_data["is_obstacle"] = np.zeros((V.shape[0],), dtype=bool)

    # V[point_data["is_obstacle"]] *= [[1e3, 0, 1e3]]

    mesh = meshio.Mesh(points=V, cells=cells, point_data=point_data)

    return mesh


def compute_vertex_colors(
        meshes, scalar_field="scalar_value_avg", scalar_field_min=None,
        scalar_field_max=None, scalar_field_log=False, obstacle_alpha=1.0,
        cmap=cm.viridis):
    if scalar_field not in meshes[0].point_data:
        print("Scalar field for color {} not found".format(scalar_field))
        print("Possible fields: {}".format(
            ", ".join(meshes[0].point_data.keys())))
        raise Exception(
            "Scalar field for color {} not found".format(scalar_field))

    def get_scalar_field(mesh):
        scalars = mesh.point_data[scalar_field]
        if(len(scalars.shape) > 1 and scalars.shape[1] > 1):
            scalars = np.linalg.norm(scalars, axis=1)
        return scalars

    min_scalar = np.inf if scalar_field_min is None else scalar_field_min
    max_scalar = -np.inf if scalar_field_max is None else scalar_field_max
    for mesh in meshes:
        scalars = get_scalar_field(mesh)[~mesh.point_data["is_obstacle"]]
        if scalar_field_min is None:
            min_scalar = min(min_scalar, scalars.min())
        if scalar_field_max is None:
            max_scalar = max(max_scalar, scalars.max())
    print("Scalar range: [{}, {}]".format(min_scalar, max_scalar))

    if scalar_field_log:
        assert(min_scalar > 0)
        real_min = min_scalar
        min_scalar = np.log10(min_scalar)
        max_scalar = np.log10(max_scalar)
        print("Scalar range: [{}, {}]".format(min_scalar, max_scalar))

    scalar_range = max_scalar - min_scalar if max_scalar != min_scalar else np.inf

    for mesh in meshes:
        scalars = get_scalar_field(mesh)
        if scalar_field_log:
            scalars[scalars < real_min] = real_min
            scalars = np.log10(scalars)
        scalars_normalized = (scalars - min_scalar) / scalar_range
        vertex_colors = cmap(scalars_normalized).reshape(-1, 4)
        vertex_colors[mesh.point_data["is_obstacle"], :] = [
            0.4, 0.4, 0.4, obstacle_alpha]
        mesh.point_data["colors"] = vertex_colors


def resolve_path(path, root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def parse_vtm(path):
    tree = ET.parse(path)
    root = tree.getroot()
    blocks = root.find("vtkMultiBlockDataSet").findall("Block")
    # assert(len(blocks) == 1)
    for block in blocks:
        if block.get("name") == "Volume":
            break
    dataset = block.find("DataSet")
    return resolve_path(dataset.attrib["file"], path.parent)


def parse_pvd(path, fps=None):
    seq = {
        "meshes": [],
        "output": path.parent.name + ".mp4",
    }

    tree = ET.parse(path)
    root = tree.getroot()
    frames = root.find("Collection").findall("DataSet")
    assert(len(frames) > 0)

    if fps is None:
        assert(len(frames) > 1)
        fps = 1 / (float(frames[1].attrib["timestep"]) -
                   float(frames[0].attrib["timestep"]))
    fps = int(round(max(1, fps)))
    seq["fps"] = fps

    for f in map(lambda f: f.attrib["file"], frames):
        f = resolve_path(f, path.parent)
        if f.suffix == ".vtm":
            f = parse_vtm(f)
        seq["meshes"].append(resolve_path(f, path.parent))

    return seq


def parse_args():
    parser = argparse.ArgumentParser("Render a mesh sequence as a video.")
    parser.add_argument("--input", "-i", nargs="+", required=True, type=pathlib.Path,
                        help="Meshes to render.")
    parser.add_argument("--output", "-o", nargs="+", default=None, type=pathlib.Path,
                        help="Output video file.")
    parser.add_argument("--width", type=int, default=1920,
                        help="Output video width.")
    parser.add_argument("--height", type=int, default=1080,
                        help="Output video height.")
    parser.add_argument("--fps", "-f", type=int, default=None,
                        help="Output video frames per second.")
    parser.add_argument("--drop-frames", type=int, default=0,
                        help="Number of frames to drop.")
    parser.add_argument("--bg-color", "-b", type=float, nargs="+",
                        default=[0.3, 0.3, 0.5, 1.0], help="Background color.")
    parser.add_argument("--dynamic-camera", action="store_true", default=False,
                        help="Zoom and shift the camera dynamically.")
    parser.add_argument("--no-wireframe", action="store_false", dest="wireframe",
                        default=True, help="Disable the wireframe rendering.")
    parser.add_argument("--smooth", action="store_true", default=False,
                        help="Disable the wireframe rendering.")
    parser.add_argument("--camera-height", type=float, default=0,
                        help="Height of the camera with the world normalized.")
    parser.add_argument("--base-zoom", type=float, default=1.0,
                        help="Camera base zoom level.")
    parser.add_argument("--scalar-field", type=str, default="E",
                        help="Scalar field to use as colors.")
    parser.add_argument("--scalar-field-min", type=float, default=None,
                        help="Scalar field max value to use as colors.")
    parser.add_argument("--scalar-field-max", type=float, default=None,
                        help="Scalar field min value to use as colors.")
    parser.add_argument("--scalar-field-log", action="store_true", default=False,
                        help="Use scalar field as colors in log scale.")
    parser.add_argument("--tensor-field", type=str, default=None,
                        help="Scalar field to visualize as a vector field of arrow.")
    parser.add_argument("--obstacle-alpha", type=float, default=1.0,
                        help="Transparency of obstacles.")
    return parser.parse_args()


def resolve_output(mesh_sequences, outputs):
    if outputs is None:
        return
    assert(len(outputs) == 1 or len(outputs) == len(mesh_sequences))
    for i, seq in enumerate(mesh_sequences):
        if len(outputs) == 1:
            i = 0
        if outputs[i].suffix == "":
            seq["output"] = outputs[i] / seq["output"]
        else:
            seq["output"] = outputs[i]


def main(args=None):
    if args is None:
        args = parse_args()

    assert(3 <= len(args.bg_color) <= 4)
    args.bg_color = np.array(args.bg_color, dtype=float)
    if args.bg_color.max() > 1.0:
        args.bg_color /= 255.0
    if(len(args.bg_color) == 3):
        args.bg_color = np.array([*args.bg_color, 1.0])

    args.input = natsorted(args.input)

    mesh_sequences = [{
        "meshes": [],
        "fps": 30,
        "output": "anim.mp4",
    }]
    for f in args.input:
        if f.suffix == ".pvd":
            mesh_sequences.append(parse_pvd(f))
        else:
            mesh_sequences[0]["meshes"].append(f)

    resolve_output(mesh_sequences, args.output)

    for seq in mesh_sequences:
        if not seq["meshes"]:
            continue

        mesh_paths = seq["meshes"][::args.drop_frames + 1]
        meshes = [load_mesh(mesh_path) for mesh_path in tqdm(mesh_paths)]

        compute_vertex_colors(
            meshes,
            scalar_field=args.scalar_field,
            scalar_field_min=args.scalar_field_min,
            scalar_field_max=args.scalar_field_max,
            scalar_field_log=args.scalar_field_log,
            obstacle_alpha=args.obstacle_alpha,
        )

        zooms_and_shifts = [
            # compute_scale_and_shift_to_fit_mesh(mesh.points[~mesh.point_data["is_obstacle"]]) for mesh in meshes
            compute_scale_and_shift_to_fit_mesh(mesh.points) for mesh in meshes
        ]
        zooms_and_shifts = (
            np.vstack([np.hstack(zs) for zs in zooms_and_shifts]))
        if args.dynamic_camera:
            plot_zooms_and_shifts(
                zooms_and_shifts[:, 0], zooms_and_shifts[:, 1:])
            zooms_and_shifts[:, 0] = smooth_1D_data(zooms_and_shifts[:, 0])
            zooms_and_shifts[:, 1:] = laplacian_smoothing(
                zooms_and_shifts[:, 1:])

        frames = []
        for i, mesh in enumerate(pbar := tqdm(meshes)):
            pbar.set_description(str(seq["output"]))
            zoom, *shift = zooms_and_shifts[i if args.dynamic_camera else 0]
            color, _ = render(
                mesh, width=args.width, height=args.height,
                bg_color=args.bg_color, wireframe=args.wireframe, smooth=args.smooth,
                camera_height=args.camera_height, base_zoom=args.base_zoom,
                tensor_field=args.tensor_field, zoom=zoom, shift=shift)
            frames.append(color)

        if args.fps is not None:
            seq["fps"] = args.fps
        elif args.drop_frames > 0:
            seq["fps"] //= (args.drop_frames + 1)

        writer_args = dict()
        if seq["output"].suffix == ".mp4":
            macro_block_size = 1
            while (args.width % macro_block_size == 0
                   and args.height % macro_block_size == 0
                   and macro_block_size < 16):
                macro_block_size *= 2
            macro_block_size = max(1, macro_block_size // 2)
            writer_args |= dict(
                fps=seq["fps"],
                quality=5,
                macro_block_size=macro_block_size,
                codec="libx264",
                pixelformat="yuv420p",
            )
        elif seq["output"].suffix == ".gif":
            writer_args |= dict(
                plugin="pillow",
                mode="RGBA",
                duration=len(frames) / seq["fps"],
                loop=0,
                transparency=0,
                disposal=2
            )

        seq["output"].parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(seq["output"], frames, **writer_args)


if __name__ == "__main__":
    main()
