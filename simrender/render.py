import numpy as np

import trimesh
import igl

import pyrender
from pyrender import RenderFlags as rf

from .transform import *
from .camera import *


def create_default_scene(aspectRatio, bg_color=None, camera_height=0.0, camera_mode="perspective"):
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

    if camera_mode == "perspective":
        camera = pyrender.PerspectiveCamera(
            yfov=np.deg2rad(45),
            znear=0.1,
            zfar=100,
            aspectRatio=aspectRatio
        )
    else:
        camera = pyrender.OrthographicCamera(
            xmag=1.0,
            ymag=1.0,
            znear=0.1,
            zfar=100
        )
    camera_position = np.array([0.0, camera_height, 3.0])
    # camera_position = np.array([0.0, 1.5, 2.65])
    camera_target = np.array([0.0, camera_height, 0.0])
    scene.add(
        camera,
        pose=(translate(camera_position) @
              align_vectors([0, 0, -1], camera_target - camera_position))
    )

    return scene


def render(mesh, width=800, height=600, bg_color=None, wireframe=True, smooth=False,
           camera_height=0.0, base_zoom=1.0, tensor_field=None, zoom=None, shift=None,
           camera_mode="perspective"):
    scene = create_default_scene(
        aspectRatio=width/height, bg_color=bg_color, camera_height=camera_height,
        camera_mode=camera_mode)

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
