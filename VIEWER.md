# PartField Viewers

PartField includes two interactive 3D viewers for visualizing segmentation results.

## Open3D Viewer (PLY/GLB/OBJ)

**Files:** `viewer.py` + `run_viewer.sh`

The Open3D-based viewer displays triangulated mesh files (PLY, GLB, OBJ) with part colorization. Best for viewing mesh-based outputs from variants 1-4.

### Usage

```bash
# Direct Python
python viewer.py model.glb
python viewer.py model.glb --labels labels.npy
python viewer.py --browse data/objaverse_samples/

# With WSL2/software rendering
./run_viewer.sh model.glb
./run_viewer.sh --browse exp_results/clustering/objaverse/ply/
```

### Requirements

- `open3d` - 3D visualization and mesh loading
- `trimesh` - Mesh format conversion
- `matplotlib` - Color palettes
- `Pillow` - Screenshot support

### Setup (WSL2)

The `run_viewer.sh` script auto-configures:
- Software rendering via Mesa llvmpipe
- X11 display detection (WSLg or external X server)
- OpenGL 3.3 compatibility override

---

## BREP Viewer (STEP/CAD)

**Files:** `brep_viewer.py` + `run_brep_viewer.sh`

The Qt+pythonOCC viewer displays STEP CAD files with per-solid segment coloring. Best for viewing BREP outputs from variants 5-8 and `brep_generator.py`.

### Usage

```bash
# View a STEP file
python brep_viewer.py result.step
./run_brep_viewer.sh result.step

# On-the-fly: generate BREP from mesh + labels and view
python brep_viewer.py --mesh model.glb --labels labels.npy

# With global alignment
python brep_viewer.py --mesh model.glb --labels labels.npy --alignment global

# Browse STEP files in directory
./run_brep_viewer.sh --browse exp_results/brep/variant5_auto_self_brep/
```

### Requirements

- `pythonocc-core` - BREP geometry and STEP I/O (`conda install -c conda-forge pythonocc-core`)
- `PyQt5` or `PySide2` - Qt GUI framework
- `matplotlib` - Color palettes

### Setup (WSL2)

The `run_brep_viewer.sh` script auto-configures:
- `QT_QPA_PLATFORM=xcb` for X11 mode
- Software rendering via Mesa llvmpipe
- Display detection for WSLg or external X server
- Dependency verification (pythonocc-core, PyQt5)

---

## Keyboard Controls

Both viewers share similar keyboard controls:

| Key | Open3D Viewer | BREP Viewer |
|-----|---------------|-------------|
| `T` / `TAB` | Cycle: Original, Segmented, BBoxes, PCA | Cycle: Colored, Wireframe, Original |
| `C` | Next clustering (more parts) | Next clustering |
| `V` | Previous clustering (fewer parts) | Previous clustering |
| `A` / `LEFT` | Previous file | Previous file |
| `D` / `RIGHT` | Next file | Next file |
| `S` | Save screenshot | Save screenshot |
| `R` | Reset camera | Reset/Fit All |
| `H` | Toggle help | - |
| `ESC` / `Q` | Exit | Exit |

---

## Command Line Options

### Open3D Viewer (`viewer.py`)

| Option | Short | Description |
|--------|-------|-------------|
| `file` | | Path to mesh file (GLB, PLY, OBJ, STL, OFF) |
| `--labels` | `-l` | Path to clustering labels NPY file |
| `--browse` | `-b` | Browse all files in directory |

### BREP Viewer (`brep_viewer.py`)

| Option | Short | Description |
|--------|-------|-------------|
| `file` | | Path to STEP file |
| `--mesh` | `-m` | Mesh file for on-the-fly generation |
| `--labels` | `-l` | Labels file for on-the-fly generation |
| `--alignment` | `-a` | BBox alignment: `self` or `global` |
| `--mode` | | Generation mode: `bbox` or `primitive` |
| `--browse` | `-b` | Browse STEP files in directory |

---

## Feature Comparison

| Feature | Open3D Viewer | BREP Viewer |
|---------|---------------|-------------|
| File formats | PLY, GLB, OBJ, STL, OFF | STEP, STP |
| Geometry type | Triangulated mesh | Parametric BREP solid |
| Per-segment coloring | Vertex colors | Per-solid XCAF colors |
| On-the-fly generation | No | Yes (mesh + labels -> STEP) |
| PCA feature view | Yes | No |
| Bounding box view | Yes (mesh overlay) | Yes (native solids) |
| CAD tool compatible | No | Yes (FreeCAD, SolidWorks, etc.) |
| Software rendering | Yes (Mesa llvmpipe) | Yes (Mesa llvmpipe) |
| WSL2 support | Yes | Yes |
