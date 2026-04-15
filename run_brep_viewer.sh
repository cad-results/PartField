#!/bin/bash
# Wrapper script for PartField BREP Viewer with WSL2/Qt display support
# Requires: pythonocc-core, PyQt5 (or PySide2)

# Function to check if we're running in WSL
is_wsl() {
    [ -f /proc/version ] && grep -qi microsoft /proc/version
}

# Function to check if X server is running
check_x_server() {
    xset -q &>/dev/null
    return $?
}

# Function to find working display
find_working_display() {
    local displays=(":0" ":0.0" "localhost:0" "localhost:0.0")

    if command -v ip &>/dev/null; then
        local host_ip=$(ip route show | grep -i default | awk '{ print $3 }')
        if [ -n "$host_ip" ]; then
            displays+=("${host_ip}:0" "${host_ip}:0.0")
        fi
    fi

    if [ -f /etc/resolv.conf ]; then
        local nameserver=$(grep nameserver /etc/resolv.conf | awk '{print $2}' | head -1)
        if [ -n "$nameserver" ]; then
            displays+=("${nameserver}:0" "${nameserver}:0.0")
        fi
    fi

    for display in "${displays[@]}"; do
        if DISPLAY=$display timeout 1 xset -q &>/dev/null; then
            echo "$display"
            return 0
        fi
    done

    return 1
}

echo "PartField BREP Viewer - Initializing display environment..."

# Detect WSL
if is_wsl; then
    echo "Running in WSL2 environment"

    if [ -d "/mnt/wslg" ]; then
        echo "WSLg detected - using native WSL GUI support"
        echo "Forcing X11 mode for Qt/pythonOCC compatibility"
        export DISPLAY=:0

        unset WAYLAND_DISPLAY
        export WAYLAND_DISPLAY=
        export GDK_BACKEND=x11
        export QT_QPA_PLATFORM=xcb
        export SDL_VIDEODRIVER=x11
        export PYOPENGL_PLATFORM=x11
    else
        if [ -z "$DISPLAY" ]; then
            echo "DISPLAY not set, searching for X server..."
            working_display=$(find_working_display)
            if [ -n "$working_display" ]; then
                export DISPLAY="$working_display"
                echo "Found working display: $DISPLAY"
            else
                echo ""
                echo "ERROR: No X server detected!"
                echo ""
                echo "To run GUI applications in WSL2, you need one of these options:"
                echo ""
                echo "Option 1: Windows 11 with WSLg (Recommended)"
                echo "  - Update to Windows 11 (build 22000 or later)"
                echo "  - Update WSL: wsl --update"
                echo ""
                echo "Option 2: Install VcXsrv on Windows"
                echo "  1. Download VcXsrv from: https://sourceforge.net/projects/vcxsrv/"
                echo "  2. Run XLaunch with 'Disable access control'"
                echo "  3. export DISPLAY=\$(cat /etc/resolv.conf | grep nameserver | awk '{print \$2}'):0"
                echo ""
                exit 1
            fi
        else
            echo "Using DISPLAY=$DISPLAY"
        fi
    fi
else
    # Native Linux
    if [ -z "$DISPLAY" ]; then
        if [ -n "$WAYLAND_DISPLAY" ]; then
            echo "Wayland session detected"
            export DISPLAY=:0
        else
            export DISPLAY=:0
        fi
    fi
fi

# Qt/OpenGL configuration
echo "Configuring display settings..."

# Force software rendering for compatibility
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330
export GALLIUM_DRIVER=llvmpipe

# Request depth and stencil buffers from Mesa
# NOTE: Do NOT set EGL_PLATFORM=surfaceless — it prevents creation of
# a proper window visual with depth/stencil buffers, causing TKOpenGl
# to render incorrectly (tiny patch in bottom-left corner).
export MESA_GLSL_CACHE_DISABLE=true

# Qt: ensure proper high-DPI scaling so the viewport fills the widget
export QT_AUTO_SCREEN_SCALE_FACTOR=1
export QT_ENABLE_HIGHDPI_SCALING=1

# WSL2-specific settings
if is_wsl; then
    export LIBGL_ALWAYS_INDIRECT=0
    export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
    export __GLX_VENDOR_LIBRARY_NAME=mesa
    export QT_QPA_PLATFORM=xcb
    export GDK_BACKEND=x11
    export LIBGL_DRI3_DISABLE=1

    if [ -z "$XDG_RUNTIME_DIR" ] && [ -d "/mnt/wslg/runtime-dir" ]; then
        export XDG_RUNTIME_DIR="/mnt/wslg/runtime-dir"
    fi
fi

# Verify dependencies
echo "Checking dependencies..."

if ! python3 -c "from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox" 2>/dev/null; then
    echo "ERROR: pythonocc-core is not installed"
    echo "Install with: conda install -c conda-forge pythonocc-core"
    exit 1
fi

if ! python3 -c "
try:
    from PyQt5 import QtWidgets
except ImportError:
    from PySide2 import QtWidgets
" 2>/dev/null; then
    echo "ERROR: PyQt5 or PySide2 is not installed"
    echo "Install with: pip install PyQt5  (or conda install pyqt)"
    exit 1
fi

echo "  pythonocc-core: OK"
echo "  Qt bindings: OK"

# Print configuration
echo ""
echo "Display Configuration:"
echo "  DISPLAY=$DISPLAY"
echo "  QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-auto}"
echo "  OpenGL: Software rendering (Mesa llvmpipe)"

# Test X server
if command -v xset &>/dev/null; then
    if xset -q &>/dev/null; then
        echo "  X server: OK"
    else
        echo "  WARNING: Cannot connect to X server at $DISPLAY"
    fi
fi

echo ""
echo "Starting BREP viewer..."
echo "----------------------------------------"
echo ""

# Run the viewer
exec python3 brep_viewer.py "$@"
