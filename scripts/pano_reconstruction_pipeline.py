# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Eliview for panorama support.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Panorama 3D Reconstruction Pipeline

Full pipeline for reconstructing 3D scenes from street-view panoramas:
1. Extract perspective views from panoramas
2. Select overlapping view sequences for pose estimation
3. Run MapAnything inference with proper pose alignment
4. Export unified 3D reconstruction

Usage:
    python pano_reconstruction_pipeline.py --pano_folder /path/to/panos --output_folder /path/to/output
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from tqdm import tqdm

# Import panorama extraction utilities
from pano_to_perspective import extract_perspective, get_view_directions, process_panorama


def select_sequential_views(pano_folder: str, num_panos: int = 5, stride: int = 2):
    """
    Select a sequence of panoramas for reconstruction.

    For street-view data, sequential panoramas typically have significant overlap.

    Args:
        pano_folder: Folder containing panoramas
        num_panos: Number of panoramas to select
        stride: Step size between panoramas

    Returns:
        List of panorama paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    pano_files = sorted([
        os.path.join(pano_folder, f)
        for f in os.listdir(pano_folder)
        if Path(f).suffix.lower() in image_extensions
    ])

    # Select evenly spaced panoramas
    indices = list(range(0, len(pano_files), stride))[:num_panos]
    selected = [pano_files[i] for i in indices if i < len(pano_files)]

    return selected


def extract_forward_views(pano_paths: list, output_dir: str,
                         fov: float = 70, width: int = 640, height: int = 480):
    """
    Extract forward-facing views from a sequence of panoramas.

    For street-view data along a road, we extract views looking forward,
    slightly left, and slightly right to capture overlapping regions.

    Args:
        pano_paths: List of panorama paths
        output_dir: Output directory
        fov: Field of view
        width: Output width
        height: Output height

    Returns:
        List of output image paths organized by viewing direction
    """
    import cv2

    os.makedirs(output_dir, exist_ok=True)

    # View directions: forward, slight left, slight right
    # This provides overlapping views between consecutive panoramas
    directions = [
        (0, 0, "front"),      # Forward
        (-30, 0, "left30"),   # 30 degrees left
        (30, 0, "right30"),   # 30 degrees right
        (-60, 0, "left60"),   # 60 degrees left
        (60, 0, "right60"),   # 60 degrees right
    ]

    all_outputs = []

    for pano_idx, pano_path in enumerate(tqdm(pano_paths, desc="Extracting views")):
        pano = cv2.imread(pano_path)
        if pano is None:
            print(f"Warning: Could not load {pano_path}")
            continue

        base_name = Path(pano_path).stem

        for yaw, pitch, dir_name in directions:
            perspective = extract_perspective(pano, yaw, pitch, fov, width, height)
            output_path = os.path.join(output_dir, f"pano{pano_idx:03d}_{dir_name}.jpg")
            cv2.imwrite(output_path, perspective, [cv2.IMWRITE_JPEG_QUALITY, 95])
            all_outputs.append({
                'path': output_path,
                'pano_idx': pano_idx,
                'pano_path': pano_path,
                'yaw': yaw,
                'pitch': pitch,
                'direction': dir_name
            })

    return all_outputs


def run_mapanything_inference(image_folder: str, output_path: str, memory_efficient: bool = True):
    """
    Run MapAnything inference on extracted perspective views.

    Args:
        image_folder: Folder containing perspective images
        output_path: Output GLB path
        memory_efficient: Use memory-efficient inference
    """
    from mapanything.models import MapAnything
    from mapanything.utils.geometry import depthmap_to_world_frame
    from mapanything.utils.image import load_images
    from mapanything.utils.viz import predictions_to_glb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading MapAnything model...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)

    # Load images
    print(f"Loading images from: {image_folder}")
    views = load_images(image_folder)
    print(f"Loaded {len(views)} views")

    # Run inference
    print("Running inference...")
    outputs = model.infer(views, memory_efficient_inference=memory_efficient)
    print("Inference complete!")

    # Process outputs
    world_points_list = []
    images_list = []
    masks_list = []

    for view_idx, pred in enumerate(outputs):
        depthmap_torch = pred["depth_z"][0].squeeze(-1)
        intrinsics_torch = pred["intrinsics"][0]
        camera_pose_torch = pred["camera_poses"][0]

        pts3d_computed, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()
        pts3d_np = pts3d_computed.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()

        world_points_list.append(pts3d_np)
        images_list.append(image_np)
        masks_list.append(mask)

    # Stack all views
    world_points = np.stack(world_points_list, axis=0)
    images = np.stack(images_list, axis=0)
    final_masks = np.stack(masks_list, axis=0)

    predictions = {
        "world_points": world_points,
        "images": images,
        "final_masks": final_masks,
    }

    # Export GLB
    print(f"Saving GLB to: {output_path}")
    scene_3d = predictions_to_glb(predictions, as_mesh=True)
    scene_3d.export(output_path)
    print(f"Successfully saved: {output_path}")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Panorama 3D Reconstruction Pipeline"
    )
    parser.add_argument(
        "--pano_folder",
        type=str,
        required=True,
        help="Folder containing panorama images"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Output folder for results"
    )
    parser.add_argument(
        "--num_panos",
        type=int,
        default=5,
        help="Number of panoramas to process (default: 5)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Stride between panoramas (default: 2)"
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=70,
        help="Field of view for perspective extraction (default: 70)"
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
        "--memory_efficient",
        action="store_true",
        default=True,
        help="Use memory efficient inference"
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip perspective extraction (use existing images)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    perspective_dir = os.path.join(args.output_folder, "perspectives")

    # Step 1: Select panoramas
    print("=" * 60)
    print("Step 1: Selecting panoramas")
    print("=" * 60)
    pano_paths = select_sequential_views(args.pano_folder, args.num_panos, args.stride)
    print(f"Selected {len(pano_paths)} panoramas")
    for p in pano_paths:
        print(f"  - {Path(p).name}")

    # Step 2: Extract perspective views
    if not args.skip_extraction:
        print("\n" + "=" * 60)
        print("Step 2: Extracting perspective views")
        print("=" * 60)
        view_info = extract_forward_views(
            pano_paths, perspective_dir,
            fov=args.fov, width=args.width, height=args.height
        )
        print(f"Extracted {len(view_info)} perspective views")
    else:
        print("\n" + "=" * 60)
        print("Step 2: Skipping extraction (using existing images)")
        print("=" * 60)

    # Step 3: Run MapAnything inference
    print("\n" + "=" * 60)
    print("Step 3: Running MapAnything inference")
    print("=" * 60)
    output_glb = os.path.join(args.output_folder, "reconstruction.glb")
    run_mapanything_inference(perspective_dir, output_glb, args.memory_efficient)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"Output: {output_glb}")


if __name__ == "__main__":
    main()
