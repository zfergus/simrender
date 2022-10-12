import pathlib
import xml.etree.ElementTree as ET


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
