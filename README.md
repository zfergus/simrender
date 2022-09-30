# simrender
Render a simulation mesh sequence using Pyrender.

## Usage

```
$ python render.py --help
usage: Render a mesh sequence as a video. [-h] --input INPUT [INPUT ...] [--output OUTPUT [OUTPUT ...]] [--width WIDTH] [--height HEIGHT] [--fps FPS] [--drop-frames DROP_FRAMES] [--bg-color BG_COLOR [BG_COLOR ...]] [--dynamic-camera]
                                          [--no-wireframe] [--camera-height CAMERA_HEIGHT] [--base-zoom BASE_ZOOM] [--scalar-field SCALAR_FIELD] [--tensor-field TENSOR_FIELD]

options:
  -h, --help            show this help message and exit
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Meshes to render.
  --output OUTPUT [OUTPUT ...], -o OUTPUT [OUTPUT ...]
                        Output video file.
  --width WIDTH         Output video width.
  --height HEIGHT       Output video height.
  --fps FPS, -f FPS     Output video frames per second.
  --drop-frames DROP_FRAMES
                        Number of frames to drop.
  --bg-color BG_COLOR [BG_COLOR ...], -b BG_COLOR [BG_COLOR ...]
                        Background color.
  --dynamic-camera      Zoom and shift the camera dynamically.
  --no-wireframe        Disable the wireframe rendering.
  --camera-height CAMERA_HEIGHT
                        Height of the camera with the world normalized.
  --base-zoom BASE_ZOOM
                        Camera base zoom level.
  --scalar-field SCALAR_FIELD
                        Scalar field to use as colors.
  --tensor-field TENSOR_FIELD
                        Scalar field to visualize as a vector field of arrow.
```

## Dependencies

`simrender` depends on a number of other Python libraries for reading and manipulating meshs (`meshio` and `trimesh`), rendering results (`pyrender`), writing the output images and videos (`imageio`), and other utilities (e.g., `numpy`, `scipy`, `matplotlib`, `tqdm`, `natsort`). 

All of these dependencies can be installed using the command:
```
pip install -r requirements.txt
```

### Build and Install `igl`

The only compilcated dependency is the Python bindings for libigl (used for `remove_unreferenced`, `remove_duplicate_vertices`, and `boundary_facets`). To install `libigl` using you can do the following:

```
git clone https://github.com/libigl/libigl-python-bindings.git
cd libigl-python-bindings
python setup.py install
```
