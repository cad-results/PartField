#!/usr/bin/env python3
"""
BREP Part Viewer - Qt + pythonOCC interactive viewer for PartField STEP files.

Modeled after BrepMFR/brepformer/visualize_seg.py (SegViewer class).
Supports loading STEP files, coloring segments, on-the-fly generation from
mesh + labels, and directory browsing.

Keyboard Controls:
    T/TAB       Cycle views: Colored Segments / Wireframe / Original
    C           Next clustering result (more parts)
    V           Previous clustering result (fewer parts)
    A/LEFT      Previous file (browse mode)
    D/RIGHT     Next file (browse mode)
    S           Screenshot
    R           Reset view / Fit All
    ESC/Q       Quit

Usage:
    # View a STEP file
    python brep_viewer.py result.step

    # On-the-fly: generate BREP from mesh + labels and view
    python brep_viewer.py --mesh model.glb --labels labels.npy

    # On-the-fly with global alignment
    python brep_viewer.py --mesh model.glb --labels labels.npy --alignment global

    # Browse STEP files in directory
    python brep_viewer.py --browse exp_results/brep/

    # Visualize original + generated STEP files (compare side by side)
    python brep_viewer.py --visualize original.step generated.step
"""
from __future__ import annotations

import sys
import os
import argparse
import glob
import math
from pathlib import Path
from typing import List, Optional, Tuple

# Qt binding selection
try:
    from PyQt5 import QtCore, QtWidgets, QtGui
    _qt_backend = "pyqt5"
except ImportError:
    from PySide2 import QtCore, QtWidgets, QtGui
    _qt_backend = "pyside2"

from OCC.Display.backend import load_backend
load_backend(_qt_backend)

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_SHELL, TopAbs_COMPOUND
from OCC.Core.TopoDS import topods
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Display.qtDisplay import qtViewer3d

try:
    from OCC.Core.Graphic3d import Graphic3d_NOM_MATTE
except Exception:
    Graphic3d_NOM_MATTE = None

try:
    from OCC.Core.Aspect import Aspect_TOL_SOLID
    from OCC.Core.Prs3d import Prs3d_LineAspect
except Exception:
    Aspect_TOL_SOLID = None
    Prs3d_LineAspect = None

import matplotlib.pyplot as plt
import numpy as np


def get_tab20_color(index: int, total: int) -> Tuple[float, float, float]:
    """Get a color from matplotlib's tab20 colormap."""
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(total, 1))
    rgba = cmap(index % 20)
    return (rgba[0], rgba[1], rgba[2])


def rgb_to_quantity(r: float, g: float, b: float) -> Quantity_Color:
    return Quantity_Color(r, g, b, Quantity_TOC_RGB)


class BRepPartViewer(QtWidgets.QMainWindow):
    """Interactive 3D viewer for PartField BREP STEP files."""

    VIEW_COLORED = 0
    VIEW_WIREFRAME = 1
    VIEW_ORIGINAL = 2
    VIEW_NAMES = ["Colored Segments", "Wireframe", "Original"]

    def __init__(
        self,
        step_path: Optional[str] = None,
        mesh_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        alignment: str = 'self',
        mode: str = 'bbox',
        browse_dir: Optional[str] = None,
    ):
        super().__init__()
        self.setWindowTitle("PartField BREP Viewer")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)
        self._pending_fit_all = False

        # Display state — initialized after widget is shown so the OCC view
        # gets the real window dimensions (fixes tiny-viewport-in-corner bug).
        self.display = None
        self._display_initialized = False
        self._pending_load_path: Optional[str] = None
        self._pending_generate: Optional[tuple] = None

        # State
        self.current_step_path: Optional[str] = None
        self.solid_ais_list: list = []
        self.solid_colors: list = []
        self.n_segments = 0
        self.view_mode = self.VIEW_COLORED

        # Browse mode
        self.step_files: List[str] = []
        self.file_index = 0

        # On-the-fly generation params
        self.mesh_path = mesh_path
        self.labels_path = labels_path
        self.alignment = alignment
        self.gen_mode = mode

        # Clustering files for cycling
        self.clustering_files: List[str] = []
        self.clustering_index = 0

        self._build_ui()

        # Queue loading for after display init (happens in showEvent)
        if browse_dir:
            browse_path = Path(browse_dir)
            self.step_files = sorted(
                [str(f) for f in browse_path.rglob("*.step")] +
                [str(f) for f in browse_path.rglob("*.stp")]
            )
            if self.step_files:
                self._pending_load_path = self.step_files[0]

        elif step_path:
            self._pending_load_path = step_path

        elif mesh_path and labels_path:
            self._pending_generate = (mesh_path, labels_path)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # 3D viewer — do NOT call InitDriver() here; the widget has no real
        # size yet, so the OCC view would get a tiny/zero viewport.  Init is
        # deferred to showEvent → _init_display_and_load().
        self.viewer = qtViewer3d(central)
        root_layout.addWidget(self.viewer, 1)

        # Side panel
        panel = QtWidgets.QWidget(central)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)
        panel.setMaximumWidth(280)
        root_layout.addWidget(panel, 0)

        # Title
        title_label = QtWidgets.QLabel("PartField BREP Viewer")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        panel_layout.addWidget(title_label)

        # Buttons
        self.btn_open = QtWidgets.QPushButton("Open STEP...")
        self.btn_open.clicked.connect(self.on_open_step)
        panel_layout.addWidget(self.btn_open)

        # View mode
        self.btn_view_mode = QtWidgets.QPushButton("View: Colored Segments")
        self.btn_view_mode.clicked.connect(self.on_toggle_view)
        panel_layout.addWidget(self.btn_view_mode)

        # Screenshot
        self.btn_screenshot = QtWidgets.QPushButton("Screenshot (S)")
        self.btn_screenshot.clicked.connect(self.on_screenshot)
        panel_layout.addWidget(self.btn_screenshot)

        # Reset
        self.btn_reset = QtWidgets.QPushButton("Reset View (R)")
        self.btn_reset.clicked.connect(self.on_reset_view)
        panel_layout.addWidget(self.btn_reset)

        panel_layout.addSpacing(8)

        # Navigation
        nav_label = QtWidgets.QLabel("Navigation")
        nav_label.setStyleSheet("font-weight: bold;")
        panel_layout.addWidget(nav_label)

        nav_layout = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("<< Prev (A)")
        self.btn_prev.clicked.connect(self.on_prev_file)
        self.btn_next = QtWidgets.QPushButton("Next (D) >>")
        self.btn_next.clicked.connect(self.on_next_file)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        panel_layout.addLayout(nav_layout)

        # Clustering cycling
        clust_layout = QtWidgets.QHBoxLayout()
        self.btn_prev_clust = QtWidgets.QPushButton("<< Less (V)")
        self.btn_prev_clust.clicked.connect(self.on_prev_clustering)
        self.btn_next_clust = QtWidgets.QPushButton("More (C) >>")
        self.btn_next_clust.clicked.connect(self.on_next_clustering)
        clust_layout.addWidget(self.btn_prev_clust)
        clust_layout.addWidget(self.btn_next_clust)
        panel_layout.addLayout(clust_layout)

        panel_layout.addSpacing(8)

        # Status
        self.status_label = QtWidgets.QLabel("No STEP loaded")
        self.status_label.setWordWrap(True)
        panel_layout.addWidget(self.status_label)

        panel_layout.addSpacing(8)

        # Legend
        legend_label = QtWidgets.QLabel("Segment Legend")
        legend_label.setStyleSheet("font-weight: bold;")
        panel_layout.addWidget(legend_label)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(400)
        self.legend_widget = QtWidgets.QWidget()
        self.legend_layout = QtWidgets.QVBoxLayout(self.legend_widget)
        self.legend_layout.setSpacing(2)
        scroll.setWidget(self.legend_widget)
        panel_layout.addWidget(scroll)

        panel_layout.addStretch(1)

        # Keyboard shortcuts help
        help_label = QtWidgets.QLabel(
            "Keys: T=view, C/V=cluster, A/D=file, S=screenshot, R=reset, Q=quit"
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("font-size: 10px; color: gray;")
        panel_layout.addWidget(help_label)

        self.setCentralWidget(central)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._display_initialized:
            # Wait for Qt to finish layout so the viewer widget has its
            # real pixel dimensions before we create the OCC view.
            QtCore.QTimer.singleShot(100, self._init_display_and_load)

    def _init_display_and_load(self) -> None:
        """Initialize OCC driver now that the widget is shown and laid out."""
        if self._display_initialized:
            return
        # Flush any pending resize / layout events first
        QtWidgets.QApplication.processEvents()

        self._display_initialized = True
        self.viewer.InitDriver()
        self.display = self.viewer._display

        # The view now has the correct window dimensions — tell it explicitly
        try:
            self.display.View.MustBeResized()
        except Exception:
            pass

        # Load whatever was queued in __init__
        if self._pending_load_path:
            path = self._pending_load_path
            self._pending_load_path = None
            self._load_step_file(path)
        elif self._pending_generate:
            mesh, labels = self._pending_generate
            self._pending_generate = None
            self._generate_and_load(mesh, labels)

    def _ensure_display(self) -> None:
        """Eagerly init the display if not done yet (fallback for pre-show calls)."""
        if self._display_initialized:
            return
        self._display_initialized = True
        self.viewer.InitDriver()
        self.display = self.viewer._display

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._pending_fit_all:
            QtCore.QTimer.singleShot(50, self._deferred_fit_all)

    def _deferred_fit_all(self) -> None:
        """Fit the view after the OpenGL widget has its final size."""
        self._pending_fit_all = False
        if self.display is None:
            return
        try:
            self.display.View.MustBeResized()
            self.display.FitAll()
            self.display.Repaint()
        except Exception:
            pass

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (QtCore.Qt.Key_T, QtCore.Qt.Key_Tab):
            self.on_toggle_view()
        elif key == QtCore.Qt.Key_C:
            self.on_next_clustering()
        elif key == QtCore.Qt.Key_V:
            self.on_prev_clustering()
        elif key in (QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
            self.on_next_file()
        elif key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_A):
            self.on_prev_file()
        elif key == QtCore.Qt.Key_S:
            self.on_screenshot()
        elif key == QtCore.Qt.Key_R:
            self.on_reset_view()
        elif key in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q):
            self.close()
        else:
            super().keyPressEvent(event)

    def on_open_step(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open STEP", "", "STEP Files (*.stp *.step)"
        )
        if path:
            self._load_step_file(path)

    def on_toggle_view(self) -> None:
        self.view_mode = (self.view_mode + 1) % 3
        self.btn_view_mode.setText(f"View: {self.VIEW_NAMES[self.view_mode]}")
        self._apply_view_mode()

    def on_screenshot(self) -> None:
        if self.display is None:
            return
        name = Path(self.current_step_path).stem if self.current_step_path else "brep"
        filename = f"screenshot_{name}_{self.view_mode}.png"
        self.display.View.Dump(filename)
        print(f"Screenshot saved: {filename}")

    def on_reset_view(self) -> None:
        if self.display is None:
            return
        self.display.View.MustBeResized()
        self.display.FitAll()

    def on_prev_file(self) -> None:
        if self.step_files and self.file_index > 0:
            self.file_index -= 1
            self._load_step_file(self.step_files[self.file_index])

    def on_next_file(self) -> None:
        if self.step_files and self.file_index < len(self.step_files) - 1:
            self.file_index += 1
            self._load_step_file(self.step_files[self.file_index])

    def on_prev_clustering(self) -> None:
        if not self.clustering_files or not self.mesh_path:
            return
        if self.clustering_index > 0:
            self.clustering_index -= 1
            self._generate_and_load(self.mesh_path, self.clustering_files[self.clustering_index])

    def on_next_clustering(self) -> None:
        if not self.clustering_files or not self.mesh_path:
            return
        if self.clustering_index < len(self.clustering_files) - 1:
            self.clustering_index += 1
            self._generate_and_load(self.mesh_path, self.clustering_files[self.clustering_index])

    def _load_step_file(self, step_path: str) -> None:
        """Load and display a STEP file with per-solid coloring."""
        if self.display is None:
            # Display not ready yet — queue for after init
            self._pending_load_path = step_path
            return
        reader = STEPControl_Reader()
        status = reader.ReadFile(step_path)
        if status != IFSelect_RetDone:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to read: {step_path}")
            return

        reader.TransferRoots()
        shape = reader.OneShape()
        self.current_step_path = step_path

        # Clear previous
        self.display.EraseAll()
        self.solid_ais_list.clear()
        self.solid_colors.clear()

        # Iterate solids (each solid = one segment)
        solids = []
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            solids.append(topods.Solid(explorer.Current()))
            explorer.Next()

        # If no solids, try shells
        if not solids:
            explorer = TopExp_Explorer(shape, TopAbs_SHELL)
            while explorer.More():
                solids.append(explorer.Current())
                explorer.Next()

        # If still nothing, display the whole shape
        if not solids:
            solids = [shape]

        self.n_segments = len(solids)

        for i, solid in enumerate(solids):
            color = get_tab20_color(i, self.n_segments)
            q_color = rgb_to_quantity(*color)

            ais = self.display.DisplayShape(solid, update=False, color=q_color)
            if isinstance(ais, list):
                ais = ais[0]

            # Apply matte material
            if Graphic3d_NOM_MATTE is not None:
                try:
                    ais.SetMaterial(Graphic3d_NOM_MATTE)
                except Exception:
                    pass

            self.solid_ais_list.append(ais)
            self.solid_colors.append(color)

        # Defer FitAll so the OpenGL viewport has its final size
        self._pending_fit_all = True
        QtCore.QTimer.singleShot(100, self._deferred_fit_all)
        self._update_legend()
        self._update_status()

    def _generate_and_load(self, mesh_path: str, labels_path: str) -> None:
        """Generate BREP on-the-fly from mesh + labels, then display."""
        import tempfile
        from brep_generator import BRepFromSegmentation

        pipeline = BRepFromSegmentation()

        # Create temp STEP file
        tmp = tempfile.NamedTemporaryFile(suffix='.step', delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            if self.gen_mode == 'primitive':
                success = pipeline.process_primitives(mesh_path, labels_path, tmp_path)
            else:
                success = pipeline.process_bboxes(
                    mesh_path, labels_path, tmp_path,
                    alignment=self.alignment,
                )

            if success:
                self._load_step_file(tmp_path)
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to generate BREP")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Generation failed: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _apply_view_mode(self) -> None:
        """Apply the current view mode to all displayed solids."""
        if self.display is None:
            return
        for i, ais in enumerate(self.solid_ais_list):
            if self.view_mode == self.VIEW_COLORED:
                color = self.solid_colors[i] if i < len(self.solid_colors) else (0.7, 0.7, 0.7)
                q_color = rgb_to_quantity(*color)
                try:
                    ais.SetColor(q_color)
                except Exception:
                    self.display.Context.SetColor(ais, q_color, False)
                try:
                    self.display.Context.SetDisplayMode(ais, 1, False)  # Shaded
                except Exception:
                    pass

            elif self.view_mode == self.VIEW_WIREFRAME:
                try:
                    self.display.Context.SetDisplayMode(ais, 0, False)  # Wireframe
                except Exception:
                    pass

            elif self.view_mode == self.VIEW_ORIGINAL:
                gray = rgb_to_quantity(0.7, 0.7, 0.7)
                try:
                    ais.SetColor(gray)
                except Exception:
                    self.display.Context.SetColor(ais, gray, False)
                try:
                    self.display.Context.SetDisplayMode(ais, 1, False)
                except Exception:
                    pass

            self.display.Context.Redisplay(ais, False)

        self.display.Repaint()

    def _update_legend(self) -> None:
        """Update the segment legend in the side panel."""
        # Clear existing legend items
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i in range(self.n_segments):
            color = self.solid_colors[i] if i < len(self.solid_colors) else (0.7, 0.7, 0.7)
            r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            # Choose text color for contrast
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            fg = "#000000" if luminance > 0.5 else "#ffffff"

            label = QtWidgets.QLabel(f"  Segment {i}  ")
            label.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); color: {fg}; "
                f"padding: 2px 6px; border-radius: 3px; font-size: 11px;"
            )
            self.legend_layout.addWidget(label)

    def _update_status(self) -> None:
        parts = []
        if self.current_step_path:
            parts.append(f"File: {os.path.basename(self.current_step_path)}")
        parts.append(f"Segments: {self.n_segments}")
        parts.append(f"View: {self.VIEW_NAMES[self.view_mode]}")
        if self.step_files:
            parts.append(f"File {self.file_index + 1}/{len(self.step_files)}")
        if self.clustering_files:
            parts.append(f"Clustering {self.clustering_index + 1}/{len(self.clustering_files)}")

        self.status_label.setText("\n".join(parts))
        self.setWindowTitle(f"PartField BREP Viewer - {os.path.basename(self.current_step_path or '')}")


def _setup_opengl_surface_format() -> None:
    """Set a QSurfaceFormat with depth and stencil buffers.

    Must be called BEFORE QApplication is created so the OpenGL context
    gets proper buffers.  Fixes the TKOpenGl warning:
      "window Visual is incomplete: no depth buffer, no stencil buffer"
    and the related symptom of geometry rendering as a tiny patch in the
    bottom-left corner.
    """
    try:
        from PyQt5.QtGui import QSurfaceFormat
    except ImportError:
        from PySide2.QtGui import QSurfaceFormat

    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(4)               # multi-sample anti-aliasing
    fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(fmt)


def main() -> int:
    # Request depth + stencil BEFORE QApplication is created
    _setup_opengl_surface_format()

    # Ensure proper high-DPI scaling so the GL viewport fills the widget
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    if hasattr(QtWidgets.QApplication, 'setAttribute'):
        try:
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        except AttributeError:
            pass  # Qt6+ handles this automatically

    parser = argparse.ArgumentParser(
        description="PartField BREP Viewer - Interactive 3D viewer for STEP files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View a STEP file
  python brep_viewer.py result.step

  # On-the-fly: generate BREP from mesh + labels
  python brep_viewer.py --mesh model.glb --labels labels.npy

  # On-the-fly with global alignment
  python brep_viewer.py --mesh model.glb --labels labels.npy --alignment global

  # Browse STEP files in a directory
  python brep_viewer.py --browse exp_results/brep/

  # Visualize original + generated STEP files
  python brep_viewer.py --visualize original.step generated_brep.step

Keyboard Controls:
  T/TAB       Cycle views: Colored / Wireframe / Original
  C/V         Next/Previous clustering
  A/D/Arrows  Previous/Next file
  S           Screenshot
  R           Reset view
  ESC/Q       Quit
        """
    )

    parser.add_argument("file", nargs="?", help="STEP file to view")
    parser.add_argument("--mesh", "-m", help="Mesh file for on-the-fly BREP generation")
    parser.add_argument("--labels", "-l", help="Labels file (.npy) for on-the-fly generation")
    parser.add_argument("--alignment", "-a", choices=['self', 'global'], default='self',
                        help="BBox alignment for on-the-fly generation (default: self)")
    parser.add_argument("--mode", choices=['bbox', 'primitive'], default='bbox',
                        help="Generation mode for on-the-fly (default: bbox)")
    parser.add_argument("--browse", "-b", metavar="DIR",
                        help="Browse STEP files in directory")
    parser.add_argument("--visualize", nargs='+', metavar="STEP_FILE",
                        help="Visualize one or more STEP files (e.g. original + generated)")

    args = parser.parse_args()

    if not args.file and not args.mesh and not args.browse and not args.visualize:
        parser.print_help()
        print("\nError: Specify a STEP file, --mesh + --labels, --browse DIR, or --visualize FILE(s)")
        return 1

    # --visualize mode: load multiple STEP files as a browseable list
    if args.visualize:
        step_files = []
        for f in args.visualize:
            if os.path.isfile(f):
                step_files.append(os.path.abspath(f))
            else:
                print(f"WARNING: STEP file not found, skipping: {f}")
        if not step_files:
            print("Error: No valid STEP files provided to --visualize")
            return 1

        app = QtWidgets.QApplication(sys.argv)
        window = BRepPartViewer()
        window.step_files = step_files
        window.file_index = 0
        window._pending_load_path = step_files[0]
        window.setWindowTitle(
            f"PartField BREP Viewer - Comparing {len(step_files)} file(s)"
        )
        window.show()
        return app.exec_()

    app = QtWidgets.QApplication(sys.argv)
    window = BRepPartViewer(
        step_path=args.file,
        mesh_path=args.mesh,
        labels_path=args.labels,
        alignment=args.alignment,
        mode=args.mode,
        browse_dir=args.browse,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
