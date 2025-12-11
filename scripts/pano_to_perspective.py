# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Eliview for panorama support.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Panorama to Perspective View Extraction

Extracts multiple perspective views from equirectangular panoramas for use with
MapAnything's feed-forward 3D reconstruction.

Usage:
    python pano_to_perspective.py --input_folder /path/to/panos --output_folder /path/to/output
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def create_perspective_coords(fov_deg: float, width: int, height: int):
    """Create normalized coordinates for perspective projection."""
    fov = np.radians(fov_deg)
    f = width / (2 * np.tan(fov / 2))

    # Create pixel coordinates
    u = np.arange(width) - width / 2
    v = np.arange(height) - height / 2
    u, v = np.meshgrid(u, v)

    # Normalize to camera space
    x = u / f
    y = v / f
    z = np.ones_like(x)

    # Normalize to unit sphere
    norm = np.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm

    return x, y, z


def rotation_matrix(yaw: float, pitch: float, roll: float = 0):
    """Create rotation matrix from yaw, pitch, roll (in radians)."""
    # Yaw (rotation around Y axis)
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # Pitch (rotation around X axis)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    # Roll (rotation around Z axis)
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])

    return Rz @ Rx @ Ry


def xyz_to_equirectangular(x, y, z, pano_width, pano_height):
    """Convert 3D coordinates to equirectangular pixel coordinates."""
    # Convert to spherical coordinates
    theta = np.arctan2(x, z)  # longitude (-pi to pi)
    phi = np.arcsin(np.clip(y, -1, 1))  # latitude (-pi/2 to pi/2)

    # Convert to pixel coordinates
    u = (theta / np.pi + 1) * pano_width / 2
    v = (0.5 - phi / np.pi) * pano_height

    return u, v


def extract_perspective(pano: np.ndarray, yaw: float, pitch: float,
                       fov: float = 90, out_width: int = 640, out_height: int = 480) -> np.ndarray:
    """
    Extract a perspective view from an equirectangular panorama.

    Args:
        pano: Equirectangular panorama image (H, W, 3)
        yaw: Horizontal rotation in degrees (0 = front, 90 = right, etc.)
        pitch: Vertical rotation in degrees (0 = horizon, positive = up)
        fov: Field of view in degrees
        out_width: Output image width
        out_height: Output image height

    Returns:
        Perspective view image
    """
    pano_height, pano_width = pano.shape[:2]

    # Create perspective coordinates
    x, y, z = create_perspective_coords(fov, out_width, out_height)

    # Stack and reshape for rotation
    xyz = np.stack([x, y, z], axis=-1)
    original_shape = xyz.shape
    xyz = xyz.reshape(-1, 3)

    # Apply rotation
    R = rotation_matrix(np.radians(yaw), np.radians(pitch))
    xyz_rotated = xyz @ R.T
    xyz_rotated = xyz_rotated.reshape(original_shape)

    # Convert to equirectangular coordinates
    u, v = xyz_to_equirectangular(
        xyz_rotated[..., 0],
        xyz_rotated[..., 1],
        xyz_rotated[..., 2],
        pano_width, pano_height
    )

    # Wrap coordinates
    u = u % pano_width
    v = np.clip(v, 0, pano_height - 1)

    # Sample from panorama using bilinear interpolation
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    perspective = cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    return perspective


def get_view_directions(num_horizontal: int = 8, include_up_down: bool = True):
    """
    Generate view directions for extracting perspectives from panorama.

    Args:
        num_horizontal: Number of horizontal views (evenly spaced around 360)
        include_up_down: Whether to include upward and downward views

    Returns:
        List of (yaw, pitch, name) tuples
    """
    views = []

    # Horizontal ring at eye level
    for i in range(num_horizontal):
        yaw = i * 360 / num_horizontal
        views.append((yaw, 0, f"h{i:02d}_y{int(yaw):03d}_p0"))

    if include_up_down:
        # Upper ring (looking up 30 degrees)
        for i in range(num_horizontal // 2):
            yaw = i * 360 / (num_horizontal // 2)
            views.append((yaw, -30, f"up{i:02d}_y{int(yaw):03d}_p-30"))

        # Lower ring (looking down 30 degrees)
        for i in range(num_horizontal // 2):
            yaw = i * 360 / (num_horizontal // 2)
            views.append((yaw, 30, f"dn{i:02d}_y{int(yaw):03d}_p30"))

    return views


def process_panorama(pano_path: str, output_dir: str,
                    fov: float = 90, out_width: int = 640, out_height: int = 480,
                    num_views: int = 8, include_vertical: bool = True):
    """
    Process a single panorama and extract perspective views.

    Args:
        pano_path: Path to panorama image
        output_dir: Output directory for perspective images
        fov: Field of view in degrees
        out_width: Output image width
        out_height: Output image height
        num_views: Number of horizontal views
        include_vertical: Include up/down views

    Returns:
        List of output image paths
    """
    # Load panorama
    pano = cv2.imread(pano_path)
    if pano is None:
        raise ValueError(f"Could not load panorama: {pano_path}")

    # Get base name
    base_name = Path(pano_path).stem

    # Get view directions
    views = get_view_directions(num_views, include_vertical)

    output_paths = []
    for yaw, pitch, view_name in views:
        # Extract perspective view
        perspective = extract_perspective(pano, yaw, pitch, fov, out_width, out_height)

        # Save output
        output_path = os.path.join(output_dir, f"{base_name}_{view_name}.jpg")
        cv2.imwrite(output_path, perspective, [cv2.IMWRITE_JPEG_QUALITY, 95])
        output_paths.append(output_path)

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract perspective views from equirectangular panoramas"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing panorama images"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Output folder for perspective images"
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=90,
        help="Field of view in degrees (default: 90)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Output image width (default: 640)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Output image height (default: 480)"
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=8,
        help="Number of horizontal views per panorama (default: 8)"
    )
    parser.add_argument(
        "--no_vertical",
        action="store_true",
        help="Skip up/down views"
    )
    parser.add_argument(
        "--select_panos",
        type=int,
        nargs="+",
        default=None,
        help="Only process specific panorama indices (e.g., --select_panos 0 5 10)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)

    # Get list of panorama files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    pano_files = sorted([
        os.path.join(args.input_folder, f)
        for f in os.listdir(args.input_folder)
        if Path(f).suffix.lower() in image_extensions
    ])

    if not pano_files:
        print(f"No images found in {args.input_folder}")
        return

    # Filter by selection if specified
    if args.select_panos:
        pano_files = [pano_files[i] for i in args.select_panos if i < len(pano_files)]

    print(f"Processing {len(pano_files)} panoramas...")
    print(f"Extracting {args.num_views} horizontal views per panorama")
    if not args.no_vertical:
        print(f"Including up/down views ({args.num_views // 2} each)")

    total_views = 0
    for pano_path in tqdm(pano_files, desc="Processing panoramas"):
        output_paths = process_panorama(
            pano_path,
            args.output_folder,
            fov=args.fov,
            out_width=args.width,
            out_height=args.height,
            num_views=args.num_views,
            include_vertical=not args.no_vertical
        )
        total_views += len(output_paths)

    print(f"\nExtracted {total_views} perspective views to {args.output_folder}")


if __name__ == "__main__":
    main()
