import numpy as np
from matplotlib import cm
import meshio
import trimesh
import igl


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
