# Top of main python script
import sys
import os
if 'PYOPENGL_PLATFORM' not in os.environ:  # noqa
    os.environ['PYOPENGL_PLATFORM'] = 'egl'  # noqa
print("Using PYOPENGL_PLATFORM={}".format(os.environ['PYOPENGL_PLATFORM']))  # noqa

import argparse
import pathlib

import numpy as np

from tqdm import tqdm
import imageio.v3 as iio
from natsort import natsorted

from simrender import *


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


def parse_args(cli_args=sys.argv[1:]):
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
    parser.add_argument("--camera", "-c", default="static", choices=["static", "dynamic", "global"],
                        help="How to zoom and shift the camera over time.")
    parser.add_argument("--camera-mode", default="perspective", choices=["perspective", "orthogonal"],
                        help="Camera mode (perspective or orthogonal).")
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
    return parser.parse_args(cli_args)


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

        zooms_and_shifts = compute_zooms_and_shifts(meshes, args.camera)

        frames = []
        for i, mesh in enumerate(pbar := tqdm(meshes)):
            pbar.set_description(str(seq["output"]))
            zoom, shift = zooms_and_shifts[i]
            color, _ = render(
                mesh, width=args.width, height=args.height,
                bg_color=args.bg_color, wireframe=args.wireframe, smooth=args.smooth,
                camera_height=args.camera_height, base_zoom=args.base_zoom,
                tensor_field=args.tensor_field, zoom=zoom, shift=shift,
                camera_mode=args.camera_mode)
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
                quality=10,
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
