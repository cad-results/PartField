#!/bin/bash
# Wrapper script for PartField Viewer with WSL2/software rendering support

# Function to check if we're running in WSL
is_wsl() {
    [ -f /proc/version ] && grep -qi microsoft /proc/version
}

# Function to check if X server is running
check_x_server() {
    xset -q &>/dev/null
    return $?
}

# Function to test display connection
test_display() {
    local display=$1
    DISPLAY=$display timeout 1 xset -q &>/dev/null
    return $?
}

# Function to find working display
find_working_display() {
    # Common display values for WSL2
    local displays=(":0" ":0.0" "localhost:0" "localhost:0.0")

    # Try host IP for WSLg
    if command -v ip &>/dev/null; then
        local host_ip=$(ip route show | grep -i default | awk '{ print $3 }')
        if [ -n "$host_ip" ]; then
            displays+=("${host_ip}:0" "${host_ip}:0.0")
        fi
    fi

    # Try nameserver from resolv.conf (common for WSL2)
    if [ -f /etc/resolv.conf ]; then
        local nameserver=$(grep nameserver /etc/resolv.conf | awk '{print $2}' | head -1)
        if [ -n "$nameserver" ]; then
            displays+=("${nameserver}:0" "${nameserver}:0.0")
        fi
    fi

    for display in "${displays[@]}"; do
        if test_display "$display"; then
            echo "$display"
            return 0
        fi
    done

    return 1
}

echo "PartField Viewer - Initializing display environment..."

# Detect WSL
if is_wsl; then
    echo "Running in WSL2 environment"

    # Check for WSLg (built-in GUI support in Windows 11)
    if [ -d "/mnt/wslg" ]; then
        echo "WSLg detected - using native WSL GUI support"

        # Open3D/GLFW has issues with Wayland, so ALWAYS force X11
        echo "Forcing X11 mode for Open3D/GLFW compatibility"
        export DISPLAY=:0

        # Aggressively disable Wayland
        unset WAYLAND_DISPLAY
        export WAYLAND_DISPLAY=

        # Force X11 for various toolkits
        export GDK_BACKEND=x11
        export QT_QPA_PLATFORM=xcb
        export SDL_VIDEODRIVER=x11

        # Force GLFW to use X11 (critical for Open3D)
        export PYOPENGL_PLATFORM=x11
        export MPLBACKEND=TkAgg
    else
        # Try to find a working X server
        if [ -z "$DISPLAY" ]; then
            echo "DISPLAY not set, searching for X server..."
            working_display=$(find_working_display)
            if [ -n "$working_display" ]; then
                export DISPLAY="$working_display"
                echo "Found working display: $DISPLAY"
            else
                # Provide setup instructions
                echo ""
                echo "ERROR: No X server detected!"
                echo ""
                echo "To run GUI applications in WSL2, you need one of these options:"
                echo ""
                echo "Option 1: Windows 11 with WSLg (Recommended)"
                echo "  - Update to Windows 11 (build 22000 or later)"
                echo "  - Update WSL: wsl --update"
                echo "  - Restart WSL: wsl --shutdown"
                echo ""
                echo "Option 2: Install VcXsrv on Windows"
                echo "  1. Download VcXsrv from: https://sourceforge.net/projects/vcxsrv/"
                echo "  2. Install and run XLaunch"
                echo "  3. Select 'Multiple windows' and 'Start no client'"
                echo "  4. Check 'Disable access control'"
                echo "  5. In WSL2, run: export DISPLAY=\$(cat /etc/resolv.conf | grep nameserver | awk '{print \$2}'):0"
                echo ""
                echo "Option 3: Install X410 from Microsoft Store"
                echo "  1. Purchase and install X410 from Microsoft Store"
                echo "  2. Run X410 in 'Windowed Apps' mode"
                echo "  3. In WSL2, run: export DISPLAY=\$(cat /etc/resolv.conf | grep nameserver | awk '{print \$2}'):0"
                echo ""
                echo "After setting up X server, run this script again."
                exit 1
            fi
        else
            echo "Using DISPLAY=$DISPLAY"
            # Test if the display works
            if ! check_x_server; then
                echo "WARNING: Cannot connect to display $DISPLAY"
                echo "Make sure your X server is running on Windows"
            fi
        fi
    fi
else
    # Native Linux
    if [ -z "$DISPLAY" ]; then
        if [ -n "$WAYLAND_DISPLAY" ]; then
            echo "Wayland session detected"
            # Try to use XWayland
            export DISPLAY=:0
        else
            echo "No display server detected"
            export DISPLAY=:0
        fi
    fi
fi

# OpenGL configuration for better compatibility
echo "Configuring OpenGL settings..."

# Force software rendering for compatibility
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330
export GALLIUM_DRIVER=llvmpipe

# Additional settings for WSL2
if is_wsl; then
    export LIBGL_ALWAYS_INDIRECT=0
    export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
    export __GLX_VENDOR_LIBRARY_NAME=mesa

    # Force X11 for GLFW/Open3D (critical - repeat to ensure it's set)
    export SDL_VIDEODRIVER=x11
    export QT_QPA_PLATFORM=xcb
    export GDK_BACKEND=x11
    export PYOPENGL_PLATFORM=x11

    # Disable hardware acceleration that might cause issues
    export LIBGL_DRI3_DISABLE=1

    # Set XDG runtime dir if not set (needed for some GUI apps)
    if [ -z "$XDG_RUNTIME_DIR" ] && [ -d "/mnt/wslg/runtime-dir" ]; then
        export XDG_RUNTIME_DIR="/mnt/wslg/runtime-dir"
    fi

    # Additional GLFW-specific settings to force X11
    export GLFW_IM_MODULE=
    export DISABLE_WAYLAND=1
fi

# Verify Python and required packages
echo "Checking Python dependencies..."
if ! python3 -c "import open3d" 2>/dev/null; then
    echo "ERROR: open3d is not installed"
    echo "Install with: pip install open3d"
    exit 1
fi

if ! python3 -c "import trimesh" 2>/dev/null; then
    echo "ERROR: trimesh is not installed"
    echo "Install with: pip install trimesh"
    exit 1
fi

# Print final configuration
echo ""
echo "Display Configuration:"
echo "  DISPLAY=$DISPLAY"
if [ -n "$WAYLAND_DISPLAY" ]; then
    echo "  WARNING: WAYLAND_DISPLAY is set to $WAYLAND_DISPLAY (should be empty for X11)"
else
    echo "  WAYLAND_DISPLAY: (unset - good, using X11)"
fi
echo "  GDK_BACKEND=$GDK_BACKEND"
echo "  SDL_VIDEODRIVER=$SDL_VIDEODRIVER"
echo "  OpenGL: Software rendering (Mesa llvmpipe)"
echo ""

# Test X server connection
if command -v xset &>/dev/null; then
    if xset -q &>/dev/null; then
        echo "X server connection: OK"
    else
        echo "WARNING: Cannot connect to X server at $DISPLAY"
        echo "Make sure WSLg is running or an X server is installed"
    fi
fi

echo ""
echo "Starting viewer..."
echo "----------------------------------------"
echo ""

# Run the viewer with all arguments passed through
exec python3 viewer.py "$@"
