# Copyright (C) 2025 Junran Wang and Zehao Jin
#
# This file is part of the VLM Privacy Evaluation.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Camera, image-capture and video-stitching utilities for VirtualHome.

Shared by tier1 / tier3 (and potentially tier2) for orbit-camera
rendering and frame capture.
"""

import os
import math
import random
from typing import List, Tuple, Optional

DEFAULT_SSAA: int = 2


def _downsample(img, target_w: int, target_h: int):
    """AREA-interpolation downsample for SSAA. Returns the resized image."""
    import cv2
    h, w = img.shape[:2]
    if w == target_w and h == target_h:
        return img
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def get_object_center(graph: dict, object_id: int) -> Optional[List[float]]:
    """Return ``[x, y, z]`` bounding-box centre for *object_id*, or ``None``."""
    for node in graph["nodes"]:
        if node["id"] == object_id:
            bbox = node.get("bounding_box")
            if bbox and "center" in bbox:
                c = bbox["center"]
                if isinstance(c, dict):
                    return [c["x"], c["y"], c["z"]]
                return [c[0], c[1], c[2]]
    return None


def add_orbit_cameras(
    comm,
    target_position: List[float],
    num_cameras: int = 8,
    radius: float = 2.0,
    height: float = 1.5,
    look_down_angle: float = 30,
    field_view: float = 60,
) -> List[int]:
    """
    Place *num_cameras* cameras in a ring around *target_position*.

    Returns the list of camera IDs that were successfully added.
    """
    camera_ids: List[int] = []
    target_x, target_y, target_z = target_position
    cam_height = target_y + height

    for i in range(num_cameras):
        angle = (2 * math.pi * i) / num_cameras
        cam_x = target_x + radius * math.cos(angle)
        cam_z = target_z + radius * math.sin(angle)

        yaw = math.degrees(math.atan2(target_x - cam_x, target_z - cam_z))
        pitch = look_down_angle

        position = [cam_x, cam_height, cam_z]
        rotation = [pitch, yaw, 0]

        success, message = comm.add_camera(
            position=position, rotation=rotation, field_view=field_view
        )
        if success:
            try:
                cam_id = int(message.split(":")[1])
                camera_ids.append(cam_id)
            except Exception:
                camera_ids.append(i)

    return camera_ids


def add_orbit_cameras_with_jitter(
    comm,
    target_position: List[float],
    num_cameras: int = 8,
    radius: float = 2.0,
    height: float = 1.5,
    look_down_angle: float = 30,
    field_view: float = 60,
    jitter_range: float = 0.15,
) -> List[int]:
    """Place *num_cameras* cameras in a ring, each looking at
    *target_position* plus its own small random offset.

    ``jitter_range`` controls the maximum displacement (metres) applied
    independently to x / y / z of the look-at centre for each camera.
    """
    camera_ids: List[int] = []
    target_x, target_y, target_z = target_position
    cam_height = target_y + height

    for i in range(num_cameras):
        jx = random.uniform(-jitter_range, jitter_range)
        jy = random.uniform(-jitter_range, jitter_range)
        jz = random.uniform(-jitter_range, jitter_range)
        look_x = target_x + jx
        look_y = target_y + jy
        look_z = target_z + jz

        angle = (2 * math.pi * i) / num_cameras
        cam_x = look_x + radius * math.cos(angle)
        cam_z = look_z + radius * math.sin(angle)

        yaw = math.degrees(math.atan2(look_x - cam_x, look_z - cam_z))
        pitch = look_down_angle

        position = [cam_x, cam_height, cam_z]
        rotation = [pitch, yaw, 0]

        success, message = comm.add_camera(
            position=position, rotation=rotation, field_view=field_view,
        )
        if success:
            try:
                cam_id = int(message.split(":")[1])
                camera_ids.append(cam_id)
            except Exception:
                camera_ids.append(i)

    return camera_ids


def capture_sensitive_orbit_images(
    comm,
    target_node: dict,
    output_dir: str = "output",
    prefix: str = "sensitive_orbit",
    num_cameras: int = 8,
    radius: float = 1.0,
    height: float = 0.8,
    jitter_range: float = 0.2,
    image_width: int = 1920,
    image_height: int = 1080,
    # modes: Optional[List[str]] = None,
    ssaa: int = DEFAULT_SSAA,
) -> Tuple[list, List[str]]:
    """Orbit cameras around a sensitive object with per-camera random jitter.

    Identical to :func:`capture_orbit_images` but uses
    :func:`add_orbit_cameras_with_jitter` so that each camera's look-at
    centre is the object position plus a small independent random offset.

    Returns ``(images_list, saved_path_list)``.
    """
    import cv2

    # if modes is None:
    #     modes = ["normal"]
    os.makedirs(output_dir, exist_ok=True)

    render_w, render_h = image_width * ssaa, image_height * ssaa

    bbox = target_node.get("bounding_box")
    if bbox and "center" in bbox:
        center = bbox["center"]
        target_position = (
            [center["x"], center["y"], center["z"]]
            if isinstance(center, dict)
            else [center[0], center[1], center[2]]
        )
    else:
        success, graph = comm.environment_graph()
        target_position = get_object_center(graph, target_node["id"]) if success else None

    if target_position is None:
        print(f"[sensitive_orbit] Error: cannot locate {target_node['class_name']} (id={target_node['id']})")
        return [], []

    print(f"[sensitive_orbit] centre: x={target_position[0]:.3f}, "
          f"y={target_position[1]:.3f}, z={target_position[2]:.3f}, jitter={jitter_range}")

    camera_ids = add_orbit_cameras_with_jitter(
        comm,
        target_position=target_position,
        num_cameras=num_cameras,
        radius=radius,
        height=height,
        field_view=60,
        jitter_range=jitter_range,
    )
    if not camera_ids:
        print("[sensitive_orbit] Error: no cameras were added")
        return [], []

    all_images: list = []
    all_paths: List[str] = []

    print(f"[sensitive_orbit] capturing {len(camera_ids)} images "
          f"(SSAA {ssaa}x: {render_w}×{render_h} -> {image_width}×{image_height}) ...")
    # for mode in modes:
    for i, cam_id in enumerate(camera_ids):
        success, images = comm.camera_image(
            [cam_id], mode="normal",
            image_width=render_w, image_height=render_h,
        )
        if success and len(images) > 0:
            img = images[0]
            # if mode == "depth":
                # img = (img * 255.0 / (img.max() + 1e-6)).astype("uint8")
            img = _downsample(img, image_width, image_height)

            filename = f"{prefix}_{i:03d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, img)

            all_images.append(img)
            all_paths.append(filepath)

    print(f"[sensitive_orbit] saved {len(all_paths)} images to {output_dir}")
    return all_images, all_paths


def capture_orbit_images(
    comm,
    target_node: dict,
    output_dir: str = "output",
    prefix: str = "orbit",
    num_cameras: int = 8,
    radius: float = 2.0,
    height: float = 1.0,
    image_width: int = 1920,
    image_height: int = 1080,
    # modes: Optional[List[str]] = None,
    ssaa: int = DEFAULT_SSAA,
) -> Tuple[list, List[str]]:
    """
    Place orbit cameras around *target_node* and capture images.

    *ssaa* – super-sampling anti-aliasing factor (render at ssaa× resolution
    then downsample with INTER_AREA).  Set to 1 to disable.

    Returns ``(images_list, saved_path_list)``.
    """
    import cv2

    # if modes is None:
    #     modes = ["normal"]
    os.makedirs(output_dir, exist_ok=True)

    render_w, render_h = image_width * ssaa, image_height * ssaa

    bbox = target_node.get("bounding_box")
    if bbox and "center" in bbox:
        center = bbox["center"]
        target_position = [center[0], center[1], center[2]]
    else:
        success, graph = comm.environment_graph()
        target_position = get_object_center(graph, target_node["id"]) if success else None

    if target_position is None:
        print(f"Error: Unable to get the position of {target_node['class_name']}")
        return [], []

    camera_ids = add_orbit_cameras(
        comm,
        target_position=target_position,
        num_cameras=num_cameras,
        radius=radius,
        height=height,
        field_view=60,
    )
    if not camera_ids:
        print("Error: No cameras were successfully added")
        return [], []

    all_images: list = []
    all_paths: List[str] = []

    print(f"\nCapturing {len(camera_ids)} orbit images (SSAA {ssaa}x: {render_w}×{render_h} -> {image_width}×{image_height}) ...")
    # for mode in modes:
    for i, cam_id in enumerate(camera_ids):
        success, images = comm.camera_image(
            [cam_id], mode="normal",
            image_width=render_w, image_height=render_h,
        )
        if success and len(images) > 0:
            img = images[0]
            # if mode == "depth":
            #     img = (img * 255.0 / (img.max() + 1e-6)).astype("uint8")
            img = _downsample(img, image_width, image_height)

            filename = f"{prefix}_{i:03d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, img)

            all_images.append(img)
            all_paths.append(filepath)

    print(f"Saved {len(all_paths)} orbit images to {output_dir}")
    return all_images, all_paths


def create_orbit_video(
    image_paths: List[str],
    output_path: str,
    fps: int = 10,
    loop: int = 1,
) -> bool:
    """Stitch orbit images into an MP4 video.  Returns ``True`` on success."""
    import cv2

    if not image_paths:
        return False

    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        return False

    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for _ in range(loop):
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                out.write(img)

    out.release()
    print(f"Orbit video saved: {output_path}")
    return True


def capture_closeup_images(
    comm,
    target_node: dict,
    output_dir: str = "output",
    prefix: str = "closeup",
    num_angles: int = 4,
    distance: float = 0.8,
    height_offset: float = 0.3,
    look_down_angle: float = 40,
    field_view: float = 50,
    image_width: int = 1920,
    image_height: int = 1080,
    # modes: Optional[List[str]] = None,
    ssaa: int = DEFAULT_SSAA,
) -> Tuple[list, List[str]]:
    """
    Place cameras very close to *target_node* and capture close-up images
    from multiple angles.  Useful for verifying that small / sensitive
    objects are actually visible in the scene.

    *ssaa* – super-sampling anti-aliasing factor.  Set to 1 to disable.

    Returns ``(images_list, saved_path_list)``.
    """
    import cv2

    # if modes is None:
    #     modes = ["normal"]
    os.makedirs(output_dir, exist_ok=True)

    render_w, render_h = image_width * ssaa, image_height * ssaa

    bbox = target_node.get("bounding_box")
    if bbox and "center" in bbox:
        c = bbox["center"]
        target_position = [c["x"], c["y"], c["z"]] if isinstance(c, dict) else [c[0], c[1], c[2]]
    else:
        success, graph = comm.environment_graph()
        target_position = get_object_center(graph, target_node["id"]) if success else None

    if target_position is None:
        print(f"[closeup] Error: cannot locate {target_node['class_name']} (id={target_node['id']})")
        return [], []

    tx, ty, tz = target_position
    print(f"[closeup] target position: x={tx:.3f}, y={ty:.3f}, z={tz:.3f}")

    camera_ids: List[int] = []
    cam_height = ty + height_offset

    for i in range(num_angles):
        angle = (2 * math.pi * i) / num_angles
        cx = tx + distance * math.cos(angle)
        cz = tz + distance * math.sin(angle)

        yaw = math.degrees(math.atan2(tx - cx, tz - cz))
        pitch = look_down_angle

        ok, msg = comm.add_camera(
            position=[cx, cam_height, cz],
            rotation=[pitch, yaw, 0],
            field_view=field_view,
        )
        if ok:
            try:
                camera_ids.append(int(msg.split(":")[1]))
            except Exception:
                camera_ids.append(i)

    if not camera_ids:
        print("[closeup] Error: no cameras were added")
        return [], []

    all_images: list = []
    all_paths: List[str] = []

    print(f"[closeup] capturing {len(camera_ids)} close-up images (SSAA {ssaa}x: {render_w}×{render_h} -> {image_width}×{image_height}) ...")
    # for mode in modes:
    for i, cam_id in enumerate(camera_ids):
        ok, images = comm.camera_image(
            [cam_id], mode="normal",
            image_width=render_w, image_height=render_h,
        )
        if ok and images:
            img = images[0]
            # if mode == "depth":
            #     img = (img * 255.0 / (img.max() + 1e-6)).astype("uint8")
            img = _downsample(img, image_width, image_height)

            filename = f"{prefix}_{i:03d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, img)

            all_images.append(img)
            all_paths.append(filepath)

    print(f"[closeup] saved {len(all_paths)} images to {output_dir}")
    return all_images, all_paths


def capture_images(
    comm,
    output_dir: str = "output",
    prefix: str = "table_scene",
    image_width: int = 1280,
    image_height: int = 720,
    modes: Optional[List[str]] = None,
    ssaa: int = DEFAULT_SSAA,
) -> Tuple[list, List[str]]:
    """
    Capture images from all built-in cameras (up to 6).

    *ssaa* – super-sampling anti-aliasing factor.  Set to 1 to disable.

    Returns ``(images_list, saved_path_list)``.
    """
    import cv2

    if modes is None:
        modes = ["normal"]
    os.makedirs(output_dir, exist_ok=True)

    render_w, render_h = image_width * ssaa, image_height * ssaa

    success, num_cameras = comm.camera_count()
    if not success:
        print("Error: Unable to get the number of cameras")
        return [], []

    print(f"There are {num_cameras} cameras in the scene (SSAA {ssaa}x: {render_w}×{render_h} -> {image_width}×{image_height})")

    all_images: list = []
    all_paths: List[str] = []

    for camera_idx in range(min(num_cameras, 6)):
        for mode in modes:
            success, images = comm.camera_image(
                [camera_idx], mode=mode,
                image_width=render_w, image_height=render_h,
            )
            if success and len(images) > 0:
                img = images[0]
                if mode == "depth":
                    img = (img * 255.0 / (img.max() + 1e-6)).astype("uint8")
                img = _downsample(img, image_width, image_height)

                filename = f"{prefix}_cam{camera_idx}_{mode}.png"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, img)

                all_images.append(img)
                all_paths.append(filepath)
                print(f"  Saved image: {filepath}")

    return all_images, all_paths
