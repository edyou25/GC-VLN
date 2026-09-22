#!/usr/bin/env python3
"""GC-VLN HDF5 episode viewer: detection/depth panoramas, two maps, three graphs.

Requires the existing src/debug_log_reader.py from your GC-VLN repository.
Usage: python scripts/view_hdf5_log.py episode.h5 [--frame 120] [--save out.png]
"""
import argparse
import os
from pathlib import Path
import sys
import textwrap

# Avoid trying to contact a stale Qt/ICE session manager (e.g. old host 'Javis').
# Do not remove DISPLAY: it is needed for the actual GUI.
os.environ.pop('SESSION_MANAGER', None)

import h5py
import matplotlib
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.debug_log_reader import EpisodeReader, mapping_item, read_value


def get(group, path, default=None):
    value = read_value(group.get(path)) if group is not None else None
    return default if value is None else value


def missing(ax, message='Not recorded'):
    ax.clear()
    ax.set_axis_off()
    ax.text(.5, .5, message, ha='center', va='center', color='gray',
            fontsize=8, transform=ax.transAxes)


def show_image(ax, data, **kwargs):
    if data is None or np.asarray(data).size == 0:
        missing(ax)
        return None
    ax.clear()
    ax.set_axis_on()
    artist = ax.imshow(data, aspect='auto', **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    return artist


def summarize(node, depth=0, max_depth=3):
    """Show a small subtree without loading full images or maps."""
    prefix = '  ' * depth
    name = node.name.rsplit('/', 1)[-1] or '/'
    if isinstance(node, h5py.Dataset):
        if node.size <= 12:
            value = np.array2string(np.asarray(read_value(node)), threshold=12,
                                    max_line_width=90)
            return [f'{prefix}{name}: {value}']
        return [f'{prefix}{name}: array {node.shape}, {node.dtype} (open path to inspect)']
    lines = [f'{prefix}{name}/  {dict(node.attrs)}']
    if depth < max_depth:
        for child in node.values():
            lines.extend(summarize(child, depth + 1, max_depth))
    elif len(node):
        lines.append(f'{prefix}  ... {len(node)} children (enter this path to expand)')
    return lines


def stitch_panorama(step, sensor):
    """Fallback for older logs without panorama_{rgb,depth}; match GC-VLN stitching.

    Only joins the Step observation's 12 directions, NOT motion frames.
    If directional observations are incomplete, return None rather than fake a panorama.
    """
    images = []
    for angle in range(0, 360, 30):
        key = sensor if angle == 0 else f'{sensor}_{angle}'
        image = get(step, f'rgbd/{key}')
        if image is None:
            return None
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        width = image.shape[1]
        a, b = int(width * .366), int(width * .634)
        images.append(image[:, a:b, ...])
    pano = np.concatenate(images, axis=1)
    # Original GC-VLN rotates both panoramas 180 degrees after stitching.
    midpoint = pano.shape[1] // 2
    return np.concatenate((pano[:, midpoint:, ...], pano[:, :midpoint, ...]), axis=1)


class EpisodeViewer:
    def __init__(self, reader, frame=0, fps=5, depth_max=10, map_padding=.18, map_min_size=80):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        self.plt = plt
        self.reader = reader
        self.depth_max = depth_max
        self.map_padding = map_padding
        self.map_min_size = map_min_size
        self.index = -1
        self.step_name = None
        self.playing = False
        self.inspectors = []
        self.map_markers = []
        self.map_trails = []
        self.map_base_bounds = None
        # Lazy cache: one (col, row) BEV coordinate per recorded motion frame.
        self._map_xy_cache = []
        self._gt_map_cache = {}
        self.closed = False

        # Two full-width panorama rows; 2 maps in one row; 3 graphs in one row.
        # GT reference path is drawn on FMM, no separate 2D trajectory panel.
        self.fig = plt.figure(figsize=(18, 12))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title(
                f'GC-VLN: {Path(reader.file.filename).name}')
        grid = self.fig.add_gridspec(
            4, 6, left=.025, right=.985, bottom=.105, top=.885,
            hspace=.10, wspace=.09,
            height_ratios=[.85, .85, 1.65, 0.5],
        )
        self.detect_ax = self.fig.add_subplot(grid[0, :])
        self.depth_ax = self.fig.add_subplot(grid[1, :])
        self.map_axes = [self.fig.add_subplot(grid[2, :3]),
                         self.fig.add_subplot(grid[2, 3:])]
        self.graph_ax = self.fig.add_subplot(grid[3, :2])
        self.nav_ax = self.fig.add_subplot(grid[3, 2:4])
        self.dag_ax = self.fig.add_subplot(grid[3, 4:])

        self.heading_text = self.fig.text(.03, .965, '', fontsize=12, weight='bold')
        self.instruction_text = self.fig.text(.03, .942, '', va='top', fontsize=9)
        self.slider = Slider(
            self.fig.add_axes([.13, .071, .77, .018]), 'Frame', 0,
            max(1, len(reader.frames) - 1), valinit=0, valstep=1, valfmt='%d',
        )
        self.slider.set_active(len(reader.frames) > 1)
        self.slider.on_changed(lambda value: self.show_frame(int(value)))
        self.buttons = []
        controls = [
            ('Previous', lambda event: self.seek(self.index - 1)),
            ('Next', lambda event: self.seek(self.index + 1)),
            ('Play / Pause', self.toggle_play),
            ('Previous step', lambda event: self.jump_step(-1)),
            ('Next step', lambda event: self.jump_step(1)),
            ('Inspect HDF5', self.inspect),
        ]
        for i, (label, callback) in enumerate(controls):
            button = Button(self.fig.add_axes([.06 + i * .152, .022, .142, .030]), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.timer = self.fig.canvas.new_timer(interval=max(1, round(1000 / fps)))
        self.timer.add_callback(self.tick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', lambda event: self.close())

        instruction = get(reader.file, 'episode/instruction', {})
        if isinstance(instruction, dict):
            instruction = instruction.get('text_English') or instruction.get('text') or str(instruction)
        self.instruction_text.set_text(
            textwrap.fill(str(instruction), 175, max_lines=2, placeholder=' ...'))
        self.draw_graph(self.dag_ax, get(reader.file, 'episode/dag'), 'Instruction DAG', spatial=False)
        print('Panels: Detection RGB panorama | Depth panorama | '
              'BEV+Wall+Thin, FMM+GT | Scene graph, Navigation tree, DAG', flush=True)
        print('Merged map colors: BEV=gray; Wall=red; Thin=yellow; '
              'traveled path=blue; current pose=cyan', flush=True)
        if reader.frames:
            self.seek(frame)
        else:
            self.heading_text.set_text('Frame — | Step —')
            for ax in [self.depth_ax, self.detect_ax, *self.map_axes,
                       self.graph_ax, self.nav_ax]:
                missing(ax)

    def seek(self, index):
        if not self.reader.frames:
            return
        index = int(np.clip(index, 0, len(self.reader.frames) - 1))
        if self.slider.val != index:
            self.slider.set_val(index)
        else:
            self.show_frame(index)

    def jump_step(self, direction):
        if self.index < 0:
            return
        current = self.reader.frames[self.index].step
        index = self.index + direction
        while 0 <= index < len(self.reader.frames):
            if self.reader.frames[index].step != current:
                if direction < 0:
                    target = self.reader.frames[index].step
                    while index > 0 and self.reader.frames[index - 1].step == target:
                        index -= 1
                self.seek(index)
                return
            index += direction

    def toggle_play(self, event=None):
        if not self.reader.frames:
            return
        self.playing = not self.playing
        if self.playing:
            if self.index == len(self.reader.frames) - 1:
                self.seek(0)
            self.timer.start()
        else:
            self.timer.stop()

    def tick(self):
        if self.index >= len(self.reader.frames) - 1:
            self.playing = False
            self.timer.stop()
        else:
            self.seek(self.index + 1)

    def on_key(self, event):
        actions = {
            'right': lambda: self.seek(self.index + 1),
            'left': lambda: self.seek(self.index - 1),
            'up': lambda: self.jump_step(1),
            'down': lambda: self.jump_step(-1),
            ' ': self.toggle_play,
            'i': self.inspect,
            'home': lambda: self.seek(0),
            'end': lambda: self.seek(len(self.reader.frames) - 1),
        }
        if event.key in actions:
            actions[event.key]()

    def draw_graph(self, ax, graph, name, spatial=True):
        if graph is None or not len(graph):
            missing(ax)
            print(f'{name}: empty / not recorded', flush=True)
            return
        ax.clear()
        ax.set_axis_off()
        positions = {}
        if spatial:
            for node, attrs in graph.nodes(data=True):
                point = attrs.get('center', attrs.get('location'))
                if point is not None and np.asarray(point).size >= 2:
                    row, col = np.asarray(point).ravel()[:2]
                    positions[node] = (col, -row)
        is_spatial = len(positions) == len(graph)
        if not is_spatial:
            positions = nx.spring_layout(graph, seed=7)
        nx.draw_networkx(
            graph, pos=positions, ax=ax, node_size=45, width=.5,
            font_size=6, with_labels=len(graph) <= 35,
            arrows=graph.is_directed(),
        )
        print(f'{name}: {len(graph)} nodes, {graph.number_of_edges()} edges '
              f'({"grid" if is_spatial else "topology"} layout)', flush=True)

    def draw_detections(self, step, rgb):
        if rgb is None:
            missing(self.detect_ax)
            return
        image = np.asarray(rgb).copy()
        masks = step.get('perception/detections/masks')
        boxes = get(step, 'perception/detections/boxes', np.empty((0, 4)))
        labels = get(step, 'perception/detections/labels', [])
        scores = get(step, 'perception/detections/scores', [])
        colors = matplotlib.colormaps['tab20']
        if isinstance(masks, h5py.Dataset) and masks.ndim == 3:
            for i in range(masks.shape[0]):
                mask = masks[i].astype(bool)
                if mask.shape == image.shape[:2]:
                    tint = np.asarray(colors(i % 20)[:3]) * 255
                    image[mask, :3] = (.65 * image[mask, :3] + .35 * tint).astype(image.dtype)
        show_image(self.detect_ax, image)
        from matplotlib.patches import Rectangle
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = colors(i % 20)
            self.detect_ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                               fill=False, edgecolor=color, linewidth=1))
            label = str(labels[i]) if i < len(labels) else str(i)
            if i < len(scores):
                label += f' {scores[i]:.2f}'
            self.detect_ax.text(x1, y1, label, fontsize=6, color='white',
                                bbox={'facecolor': color, 'alpha': .7, 'pad': 1})
        print(f'Perception: {len(boxes)} detections', flush=True)

    def map_crop_bounds(self, step):
        """Common ROI in original BEV (col, row) coordinates for all four maps.

        Use the currently observed BEV foreground, with agent and selected goal
        included. No image resampling or modification to map coordinates.
        """
        arrays = {}
        for key in ('bev', 'wall', 'thin', 'fmm'):
            data = get(step, f'mapping/{key}')
            arrays[key] = np.asarray(data) if data is not None else None

        # The BEV's non-zero region identifies observed/traversable pixels.
        # Wall/thin/FMM may have an inverted background, so don't use their
        # full extent to infer the ROI when the BEV is available.
        base = arrays['bev']
        if base is None or base.ndim != 2:
            base = next((a for a in arrays.values() if a is not None and a.ndim == 2), None)
        if base is None:
            return None
        h, w = base.shape
        foreground = np.isfinite(base) & (base != 0)
        if foreground.all():
            foreground = np.zeros_like(foreground)

        rows, cols = np.nonzero(foreground)
        points = []
        if len(rows):
            points.extend(((float(cols.min()), float(rows.min())),
                           (float(cols.max()), float(rows.max()))))

        # These all use row/col pixel coordinates on the original map.
        selected = get(step, 'planning/selected_point')
        if selected is not None:
            target = np.asarray(selected).ravel()
            if target.size >= 2 and np.isfinite(target[:2]).all():
                points.append((float(target[1]), float(target[0])))

        gps = get(step, 'rgbd/gps')
        map_size = get(self.reader.file, 'episode/config/POLICY_CONFIG/MAP/SIZE')
        resolution = get(self.reader.file, 'episode/config/POLICY_CONFIG/MAP/RESOLUTION')
        if gps is not None and map_size is not None and resolution is not None:
            gps = np.asarray(gps).ravel()
            if gps.size >= 2 and np.isfinite(gps[:2]).all():
                points.append((float(gps[0] * resolution + map_size // 2),
                               float(-gps[1] * resolution + map_size // 2)))

        # GT is world XYZ, not map row/col: project it using logged world/GPS pairs.
        gt_map = self.ground_truth_map(map_size, resolution)
        if len(gt_map):
            gt_valid = gt_map[np.isfinite(gt_map).all(axis=1)]
            if len(gt_valid):
                points.extend((tuple(gt_valid.min(axis=0)), tuple(gt_valid.max(axis=0))))

        if points:
            xy = np.asarray(points)
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            side = max(float(np.max(hi - lo)), self.map_min_size)
            side *= 1.0 + 2.0 * self.map_padding
            center = (lo + hi) / 2.0
        else:
            side = self.map_min_size * (1.0 + 2.0 * self.map_padding)
            center = np.array([w / 2, h / 2], dtype=float)
        side = min(side, max(w, h))
        # Common bounds for both maps; show the complete reference path.
        return (center[0] - side / 8, center[0] + side / 8,
                center[1] - side / 8, center[1] + side / 8)

    @staticmethod
    def feature_mask(data, roi_bounds=None):
        """Pick the minority value in a binary-like map as its line features.

        Wall maps commonly use black lines on a white background, whereas Thin
        maps commonly use white lines on black. Use the visible ROI to find the
        minority class rather than assuming the foreground is always nonzero.
        """
        array = np.asarray(data)
        if array.ndim != 2:
            return None
        valid = np.isfinite(array)
        if not np.any(valid):
            return None
        sample = array
        if roi_bounds is not None:
            xmin, xmax, ymin, ymax = roi_bounds
            x0 = max(0, int(np.floor(xmin)))
            x1 = min(array.shape[1], int(np.ceil(xmax)))
            y0 = max(0, int(np.floor(ymin)))
            y1 = min(array.shape[0], int(np.ceil(ymax)))
            if x1 > x0 and y1 > y0:
                sample = array[y0:y1, x0:x1]
        subset = sample[np.isfinite(sample)]
        if subset.size == 0:
            return None
        background_is_zero = np.count_nonzero(subset == 0) > subset.size / 2
        mask = valid & ((array != 0) if background_is_zero else (array == 0))
        return mask

    def ground_truth_map(self, map_size, resolution):
        """Project reference_path world XYZ into FMM pixels via logged pose/GPS.

        World X/Z coordinates cannot be placed directly on an egocentric BEV.
        Fit a rigid 2D transform from matched (world X,Z) and (local GPS x,y)
        observations, then use the exact GPS -> map pixel mapping of the agent.
        Do not silently display an incorrect GT if calibration is impossible.
        """
        if map_size is None or resolution is None:
            return np.empty((0, 2), dtype=float)
        size, scale = float(np.asarray(map_size).item()), float(np.asarray(resolution).item())
        cache_key = (size, scale)
        if cache_key in self._gt_map_cache:
            return self._gt_map_cache[cache_key]

        empty = np.empty((0, 2), dtype=float)
        gt = np.asarray(self.reader.ground_truth(), dtype=float)
        world = np.asarray(self.reader.positions, dtype=float)
        if gt.ndim != 2 or gt.shape[1] < 3 or not len(gt):
            self._gt_map_cache[cache_key] = empty
            print('FMM GT: reference_path missing; not drawn', flush=True)
            return empty
        if world.ndim != 2 or world.shape[1] < 3 or not len(world):
            self._gt_map_cache[cache_key] = empty
            print('FMM GT: logged world poses missing; cannot align', flush=True)
            return empty

        # Sample across the whole episode, including non-collinear movement.
        sample_indices = np.unique(np.linspace(
            0, min(len(world), len(self.reader.frames)) - 1,
            min(128, len(world), len(self.reader.frames)), dtype=int))
        world_samples, gps_samples = [], []
        for i in sample_indices:
            gps, _ = self.reader.local_pose(int(i))
            if gps is None:
                continue
            gps = np.asarray(gps, dtype=float).ravel()
            w = world[i, [0, 2]]
            if gps.size >= 2 and np.isfinite(gps[:2]).all() and np.isfinite(w).all():
                world_samples.append(w)
                gps_samples.append(gps[:2])

        if len(world_samples) < 2:
            self._gt_map_cache[cache_key] = empty
            print('FMM GT: insufficient matched GPS / world positions', flush=True)
            return empty
        w, g = np.asarray(world_samples), np.asarray(gps_samples)
        wc, gc = w.mean(axis=0), g.mean(axis=0)
        w0, g0 = w - wc, g - gc
        if np.max(np.linalg.norm(w0, axis=1)) < .20:
            self._gt_map_cache[cache_key] = empty
            print('FMM GT: too little displacement to calibrate map direction', flush=True)
            return empty

        u, singular, vt = np.linalg.svd(w0.T @ g0)
        transform = u @ vt    # Includes a reflection if the logged GPS convention requires it.
        # For nearly straight logs, handedness cannot be established from motion alone.
        if singular[1] < .02 * singular[0]:
            if np.linalg.det(transform) < 0:
                u[:, -1] *= -1
                transform = u @ vt
            print('FMM GT: motion nearly collinear; GT lateral alignment is uncertain', flush=True)
        fitted = w0 @ transform + gc
        rmse = float(np.sqrt(np.mean(np.sum((fitted - g) ** 2, axis=1))))
        if rmse > .75:
            self._gt_map_cache[cache_key] = empty
            print(f'FMM GT: world/GPS mismatch (RMSE={rmse:.2f}m); GT omitted', flush=True)
            return empty

        gt_xy = gt[:, [0, 2]]
        local = (gt_xy - wc) @ transform + gc
        result = np.column_stack((local[:, 0] * scale + size / 2,
                                  -local[:, 1] * scale + size / 2))
        result[~np.isfinite(gt_xy).all(axis=1)] = np.nan
        self._gt_map_cache[cache_key] = result
        print(f'FMM GT: {len(result)} reference points; alignment RMSE={rmse:.3f}m', flush=True)
        return result

    def draw_maps(self, step):
        """Overlay Wall/Thin on BEV; show projected GT on the separate FMM map."""
        self.map_markers = []
        self.map_trails = []
        bounds = self.map_crop_bounds(step)
        self.map_base_bounds = bounds
        bev = get(step, 'mapping/bev')
        wall = get(step, 'mapping/wall')
        thin = get(step, 'mapping/thin')
        fmm = get(step, 'mapping/fmm')

        merged_ax, fmm_ax = self.map_axes
        show_image(merged_ax, bev, cmap='gray', vmin=0, vmax=1, origin='upper')

        show_image(fmm_ax, fmm, cmap='gray', vmin=0, vmax=1, origin='upper')
        if bev is not None:
            for name, data, color in (
                ('Wall', wall, (1.0, .10, .10, .80)),
                ('Thin', thin, (1.0, .85, .10, .95)),
            ):
                if data is None:
                    print(f'{name}: missing', flush=True)
                    continue
                mask = self.feature_mask(data, bounds)
                if mask is None or mask.shape != np.shape(bev):
                    print(f'{name}: incompatible map shape {np.shape(data)}', flush=True)
                    continue
                overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                overlay[mask] = color
                merged_ax.imshow(overlay, origin='upper', interpolation='nearest')
                fmm_ax.imshow(overlay, origin='upper', interpolation='nearest')
                print(f'{name}: {int(np.count_nonzero(mask))} highlighted pixels', flush=True)
        # Keep motion overlays above all map layers on both axes.
        for ax in self.map_axes:
            if not ax.images:
                continue
            trail, = ax.plot([], [], '-', color='deepskyblue', linewidth=1.5,
                             alpha=.95, zorder=8)
            marker, = ax.plot([], [], 'o', color='cyan', markersize=5, zorder=10)
            heading, = ax.plot([], [], color='cyan', linewidth=2, zorder=10)
            self.map_trails.append(trail)
            self.map_markers.append((marker, heading))

        if fmm is not None:
            size = get(self.reader.file, 'episode/config/POLICY_CONFIG/MAP/SIZE')
            resolution = get(self.reader.file, 'episode/config/POLICY_CONFIG/MAP/RESOLUTION')
            gt_map = self.ground_truth_map(size, resolution)
            if len(gt_map):
                # Full GT in green, blue traversed prefix grows with each Frame.
                fmm_ax.plot(gt_map[:, 0], gt_map[:, 1], '--', color='limegreen',
                            linewidth=1.8, zorder=6)
                start, goal = gt_map[0], gt_map[-1]
                if np.isfinite(start).all():
                    fmm_ax.plot(start[0], start[1], 'o', color='limegreen',
                                markersize=5, zorder=7)
                if np.isfinite(goal).all():
                    fmm_ax.plot(goal[0], goal[1], 'o', color='red',
                                markersize=6, zorder=7)
            stage_masks = mapping_item(step.get('planning/masks'),
                                       get(step, 'planning/stage_before'))
            if stage_masks is not None:
                valid = get(stage_masks, 'valid_mask')
                if valid is not None and np.shape(valid) == np.shape(fmm):
                    overlay = np.zeros((*valid.shape, 4))
                    overlay[np.asarray(valid).astype(bool)] = [0, 1, 0, .2]
                    fmm_ax.imshow(overlay, origin='upper')
            candidates = step.get('planning/waypoints')
            if candidates is None:
                candidates = step.get('planning/candidates')
            if candidates is not None:
                for entry in candidates.values():
                    points = entry.get('points')
                    if isinstance(points, h5py.Dataset) and points.ndim == 2 and points.shape[1] >= 2:
                        points = points[::max(1, int(np.ceil(len(points) / 3000)))]
                        fmm_ax.scatter(points[:, 1], points[:, 0], s=5,
                                       color='black', alpha=.45)
            selected = get(step, 'planning/selected_point')
            if selected is not None and np.asarray(selected).size == 2:
                row, col = selected
                fmm_ax.plot(col, row, 'o', color='green', markersize=8, zorder=11)

        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            for ax in self.map_axes:
                if ax.images:
                    ax.set_xlim(xmax, xmin)  # mirrored, as in Habitat comparison
                    ax.set_ylim(ymax, ymin)
                    ax.set_aspect('equal', adjustable='box')
            print(f'Map ROI: col [{xmin:.0f}, {xmax:.0f}], '
                  f'row [{ymin:.0f}, {ymax:.0f}]', flush=True)

    def map_path_prefix(self, index, map_size, resolution):
        """Map-space (col,row) trail through frame index, using logged GPS.

        Uses the *same* conversion as the cyan agent marker. Frames are read
        lazily and cached so playback does not repeatedly scan prior frames.
        """
        if map_size is None or resolution is None:
            return np.empty((0, 2), dtype=float)
        while len(self._map_xy_cache) <= index:
            frame_id = len(self._map_xy_cache)
            gps, _ = self.reader.local_pose(frame_id)
            point = np.array([np.nan, np.nan], dtype=float)
            if gps is not None:
                values = np.asarray(gps, dtype=float).ravel()
                if values.size >= 2 and np.isfinite(values[:2]).all():
                    point = np.array([
                        values[0] * resolution + map_size // 2,
                        -values[1] * resolution + map_size // 2,
                    ], dtype=float)
            self._map_xy_cache.append(point)
        return np.asarray(self._map_xy_cache[:index + 1], dtype=float)

    def update_map_path(self, index, map_size, resolution):
        """Update both map paths and keep the traveled trail within the ROI."""
        trail = self.map_path_prefix(index, map_size, resolution)
        for artist in self.map_trails:
            artist.set_data(trail[:, 0], trail[:, 1])

        base = self.map_base_bounds
        if base is None or len(trail) == 0:
            return
        valid = trail[np.isfinite(trail).all(axis=1)]
        if not len(valid):
            return

        xmin, xmax, ymin, ymax = base
        lo = np.minimum(valid.min(axis=0), (xmin, ymin))
        hi = np.maximum(valid.max(axis=0), (xmax, ymax))
        if (lo[0] < xmin or hi[0] > xmax or lo[1] < ymin or hi[1] > ymax):
            # Equal X/Y scale for all four panels, also after a long motion.
            side = max(float(np.max(hi - lo)), 1.0) * 1.05
            center = (lo + hi) / 2.0
            xmin, xmax = center[0] - side / 2, center[0] + side / 2
            ymin, ymax = center[1] - side / 2, center[1] + side / 2

        for ax in self.map_axes:
            if ax.images:
                ax.set_xlim(xmax, xmin)  # mirror consistently with the 2D path
                ax.set_ylim(ymax, ymin)
                ax.set_aspect('equal', adjustable='box')

    def show_frame(self, index):
        if not 0 <= index < len(self.reader.frames) or index == self.index:
            return
        self.index = index
        ref = self.reader.frames[index]
        step, frame = self.reader.step(index), self.reader.frame(index)
        new_step = self.step_name != ref.step
        if new_step:
            self.step_name = ref.step
            print(f'\n--- Step {ref.step} ---', flush=True)
            rgb = get(step, 'rgbd/panorama_rgb')
            if rgb is None:
                rgb = self.reader.panorama(index)
            if rgb is None:
                rgb = stitch_panorama(step, 'rgb')
            depth = get(step, 'rgbd/panorama_depth')
            if depth is None:
                depth = stitch_panorama(step, 'depth')
            if depth is not None:
                depth = np.asarray(depth)
                if depth.ndim == 3 and depth.shape[-1] == 1:
                    depth = depth[..., 0]
                depth = np.ma.masked_where(~np.isfinite(depth) | (depth <= 0), depth)
            normalized = get(self.reader.file, 'episode/cameras/depth/NORMALIZE_DEPTH', False)
            limit = 1 if normalized else self.depth_max
            show_image(self.depth_ax, depth, cmap='viridis', vmin=0, vmax=limit)
            print(f'Panoramas: RGB={np.shape(rgb) if rgb is not None else "missing"}, '
                  f'Depth={np.shape(depth) if depth is not None else "missing"}; '
                  f'Depth range=0..{limit} {"normalized" if normalized else "m"}', flush=True)
            self.draw_detections(step, rgb)
            self.draw_maps(step)
            self.draw_graph(self.graph_ax, get(step, 'scene_graph'), 'Scene graph')
            self.draw_graph(self.nav_ax, get(step, 'planning/navigation_tree/navigation_tree'),
                            'Navigation tree')

        self.heading_text.set_text(
            f'Frame {index}/{len(self.reader.frames)-1} | Step {ref.step} / {ref.local_index}')
        action = get(frame, 'action') if ref.event != 'planning' else None
        names = {0: 'STOP', 1: 'FORWARD', 2: 'TURN_LEFT', 3: 'TURN_RIGHT'}
        action_name = names.get(action, str(action)) if np.isscalar(action) else '—'
        pose = get(frame, 'after/position')
        heading = get(frame, 'after/heading')
        stage = get(step, 'planning/stage', '—')
        reason = get(frame, 'reason', '')
        error = get(step, 'error', get(step, 'motion/error', ''))
        error_line = str(error).strip().splitlines()[-1] if error else ''
        print(f'Frame {index} | Step {ref.step}/{ref.local_index} | Event: {ref.event} | Action: {action_name}')
        print(f'  Stage: {stage}; {get(step, "planning/stage_result", "")}')
        print(f'  Position XYZ: {np.array2string(np.asarray(pose), precision=3) if pose is not None else "—"}')
        print(f'  Heading: {heading}; Stuck: {get(frame, "stuck_time", "—")}; '
              f'Teleport: {get(frame, "teleport", False)}')
        print(f'  Result: {get(step, "motion/action_valid", "—")}; '
              f'Goal: {get(step, "planning/selected_point", "—")}')
        if reason:
            print(f'  Reason: {reason}')
        if error_line:
            print(f'  ERROR: {error_line}', flush=True)

        gps, compass = self.reader.local_pose(index)
        size = get(self.reader.file, 'episode/config/POLICY_CONFIG/MAP/SIZE')
        resolution = get(self.reader.file, 'episode/config/POLICY_CONFIG/MAP/RESOLUTION')
        for marker, arrow in self.map_markers:
            marker.set_data([], [])
            arrow.set_data([], [])
            if gps is not None and size is not None and resolution is not None:
                row, col = -gps[1] * resolution + size // 2, gps[0] * resolution + size // 2
                marker.set_data([col], [row])
                if compass is not None:
                    angle = np.asarray(compass).ravel()[0]
                    arrow.set_data([col, col + resolution * np.cos(angle)],
                                   [row, row + resolution * np.sin(angle)])
        # The current map snapshot is fixed per Step, but the trail grows per Frame.
        self.update_map_path(index, size, resolution)
        self.fig.canvas.draw_idle()

    def inspect(self, event=None):
        from matplotlib.widgets import Slider, TextBox
        if self.playing:
            self.toggle_play()
        fig = self.plt.figure(figsize=(12, 8))
        ax = fig.add_axes([.03, .14, .93, .73])
        ax.set_axis_off()
        output = ax.text(0, 1, '', va='top', family='monospace', fontsize=9, transform=ax.transAxes)
        initial = f'/steps/{self.step_name}/planning/constraints' if self.step_name else '/episode'
        box = TextBox(fig.add_axes([.12, .91, .8, .04]), 'HDF5 path', initial=initial)
        slider = Slider(fig.add_axes([.12, .05, .8, .025]), 'Line', 0, 1, valstep=1)
        lines = []

        def scroll(value):
            start = int(value)
            output.set_text('\n'.join(lines[start:start + 36]))
            fig.canvas.draw_idle()

        def load(path):
            nonlocal lines
            node = self.reader.file.get(path)
            if node is None:
                lines = [f'Not recorded: {path}']
            elif isinstance(node, h5py.Dataset) and node.size > 12:
                lines = [f'{node.name}: shape={node.shape}, dtype={node.dtype}',
                         'Array preview (first slice):',
                         str(node[tuple(slice(0, min(4, n)) for n in node.shape)])]
            else:
                lines = summarize(node)
            lines = [part for line in lines for part in line.splitlines()]
            slider.valmax = max(1, len(lines) - 36)
            slider.ax.set_xlim(0, slider.valmax)
            slider.set_val(0)
            scroll(0)

        box.on_submit(load)
        slider.on_changed(scroll)
        fig.canvas.mpl_connect('scroll_event', lambda event: slider.set_val(
            np.clip(slider.val - event.step * 3, 0, slider.valmax)))
        self.inspectors.append((fig, box, slider))
        load(initial)
        fig.show()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.timer.stop()
        for fig, _, _ in self.inspectors:
            self.plt.close(fig)
        self.reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path, help='Episode .h5 file')
    parser.add_argument('--frame', type=int, default=0, help='Initial global frame index (zero based)')
    parser.add_argument('--fps', type=float, default=5, help='Playback frames per second')
    parser.add_argument('--depth-max', type=float, default=10, help='Depth display limit in meters')
    parser.add_argument('--map-padding', type=float, default=.0, help='Extra ROI margin as a fraction of its side (default 0)')
    parser.add_argument('--map-min-size', type=int, default=80, help='Minimum ROI side in map pixels (default 80)')
    parser.add_argument('--backend', help='Matplotlib backend, e.g. TkAgg or QtAgg')
    parser.add_argument('--save', type=Path, help='Save one frame as PNG without opening a GUI')
    args = parser.parse_args()
    if args.fps <= 0 or args.depth_max <= 0 or args.frame < 0 or args.map_padding < 0 or args.map_min_size <= 0:
        parser.error('--fps, --depth-max, --map-min-size must be positive; --frame, --map-padding nonnegative')
    if args.save:
        matplotlib.use('Agg')
    elif args.backend:
        matplotlib.use(args.backend)
    import matplotlib.pyplot as plt

    print(f'Opening {args.episode} ...', flush=True)
    try:
        reader = EpisodeReader(args.episode)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    viewer = None
    try:
        print(f'Building viewer: {len(reader.frames)} frames ...', flush=True)
        viewer = EpisodeViewer(reader, args.frame, args.fps, args.depth_max,
                               args.map_padding, args.map_min_size)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            viewer.fig.savefig(args.save, dpi=140)
            print(f'Saved {args.save}', flush=True)
        else:
            if matplotlib.get_backend().lower() == 'agg':
                parser.error('No GUI backend. Use --backend TkAgg/QtAgg or --save preview.png.')
            plt.show()
    finally:
        if viewer is not None:
            viewer.close()
            plt.close(viewer.fig)
        else:
            reader.close()


if __name__ == '__main__':
    main()