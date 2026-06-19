"""
Landscape Generation Script

Generates different mountain landscapes for the exceptional coursework criteria.
Creates URDF and OBJ files for:
1. Steeper Mountain (sigma=2, height=6)
2. Gentle Mountain (sigma=5, height=3)
3. Rocky/Noisy Mountain (with Perlin noise)

Usage:
    python generate_landscapes.py
"""

import math
import os

# Try to import noise for rocky terrain
try:
    from noise import pnoise2
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False
    print("Note: 'noise' library not available. Rocky terrain will use simple noise.")
    print("Install with: pip install noise")


def write_obj(filename, vertices, faces):
    """Write vertices and faces to OBJ file."""
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Generated: {filename}")


def write_urdf(obj_filename, urdf_filename, name="mountain"):
    """Generate URDF file for the mesh."""
    urdf_content = f'''<?xml version="1.0"?>
<robot name="{name}">
  <link name="baseLink">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{os.path.basename(obj_filename)}" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{os.path.basename(obj_filename)}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
'''
    with open(urdf_filename, 'w') as f:
        f.write(urdf_content)
    print(f"Generated: {urdf_filename}")


def gaussian(x, y, sigma, height):
    """Gaussian height function for mountain shape."""
    return height * math.exp(-((x**2 + y**2) / (2 * sigma**2)))


def simple_noise(x, y, scale=0.5):
    """Simple pseudo-random noise when noise library unavailable."""
    # Simple hash-based noise
    n = int(x * 1000 + y * 7919)
    n = (n << 13) ^ n
    return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0) * scale


def generate_gaussian_mountain(
    output_dir,
    name,
    size=10,
    resolution=0.5,
    sigma=3,
    height=5,
    noise_scale=0,
    noise_factor=0
):
    """
    Generate a Gaussian mountain mesh.

    Args:
        output_dir: Directory to save files
        name: Base name for files (e.g., "steep_mountain")
        size: Size of terrain grid
        resolution: Grid resolution
        sigma: Gaussian spread (smaller = steeper)
        height: Peak height
        noise_scale: Perlin noise scale (0 = no noise)
        noise_factor: Perlin noise amplitude
    """
    vertices = []
    faces = []

    grid_size = int(size / resolution)

    # Generate top surface vertices
    for i in range(grid_size):
        for j in range(grid_size):
            x = -size/2 + i * resolution
            y = -size/2 + j * resolution
            z = gaussian(x, y, sigma, height)

            # Add noise if requested
            if noise_scale > 0 and noise_factor > 0:
                if NOISE_AVAILABLE:
                    z += pnoise2(x * noise_scale, y * noise_scale) * noise_factor
                else:
                    z += simple_noise(x * noise_scale, y * noise_scale) * noise_factor

            vertices.append([x, y, z])

    # Generate bottom surface vertices (flat base)
    for i in range(grid_size):
        for j in range(grid_size):
            x = -size/2 + i * resolution
            y = -size/2 + j * resolution
            vertices.append([x, y, 0])

    # Generate faces for top surface
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            bl = i * grid_size + j
            br = i * grid_size + j + 1
            tl = (i + 1) * grid_size + j
            tr = (i + 1) * grid_size + j + 1

            faces.append([bl, br, tl])
            faces.append([tl, br, tr])

    # Generate faces for bottom surface
    base_offset = grid_size * grid_size
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            bl = base_offset + i * grid_size + j
            br = base_offset + i * grid_size + j + 1
            tl = base_offset + (i + 1) * grid_size + j
            tr = base_offset + (i + 1) * grid_size + j + 1

            faces.append([bl, tl, br])
            faces.append([tl, tr, br])

    # Generate side faces
    for i in range(grid_size - 1):
        for j in range(grid_size):
            top = i * grid_size + j
            bottom = base_offset + i * grid_size + j
            top_next = (i + 1) * grid_size + j
            bottom_next = base_offset + (i + 1) * grid_size + j

            faces.append([top, bottom, top_next])
            faces.append([top_next, bottom, bottom_next])

    # Write files
    obj_file = os.path.join(output_dir, f"{name}.obj")
    urdf_file = os.path.join(output_dir, f"{name}.urdf")

    write_obj(obj_file, vertices, faces)
    write_urdf(obj_file, urdf_file, name)


def generate_all_landscapes():
    """Generate all landscape variants."""
    output_dir = "shapes"
    os.makedirs(output_dir, exist_ok=True)

    print("="*50)
    print("Generating Mountain Landscapes")
    print("="*50)

    # 1. Standard Gaussian (reference - same as original)
    print("\n1. Standard Gaussian Mountain (sigma=3, height=5)")
    generate_gaussian_mountain(
        output_dir,
        name="gaussian_pyramid",
        sigma=3,
        height=5
    )

    # 2. Steep Mountain (smaller sigma = steeper slope)
    print("\n2. Steep Mountain (sigma=2, height=6)")
    generate_gaussian_mountain(
        output_dir,
        name="steep_mountain",
        sigma=2,
        height=6
    )

    # 3. Gentle Mountain (larger sigma = gentler slope)
    print("\n3. Gentle Mountain (sigma=5, height=3)")
    generate_gaussian_mountain(
        output_dir,
        name="gentle_mountain",
        sigma=5,
        height=3
    )

    # 4. Rocky Mountain (with noise)
    print("\n4. Rocky Mountain (with Perlin noise)")
    generate_gaussian_mountain(
        output_dir,
        name="rocky_mountain",
        sigma=3,
        height=5,
        noise_scale=0.5,
        noise_factor=0.5
    )

    # 5. Tall Steep Mountain (challenging)
    print("\n5. Tall Steep Mountain (sigma=1.5, height=8)")
    generate_gaussian_mountain(
        output_dir,
        name="tall_steep_mountain",
        sigma=1.5,
        height=8
    )

    print("\n" + "="*50)
    print("All landscapes generated in 'shapes/' directory")
    print("\nTo use a different landscape, modify mountain_simulation.py:")
    print('  p.loadURDF("steep_mountain.urdf", ...)')


def main():
    generate_all_landscapes()


if __name__ == "__main__":
    main()
