#!/usr/bin/env python3
"""Interactive Matplotlib episode viewer. Run with --help for controls."""
import argparse
from pathlib import Path
import sys
import textwrap

import h5py
import matplotlib
import numpy as np
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.debug_log_reader import EpisodeReader, mapping_item, read_value


def get(group, path, default=None):
    value = read_value(group.get(path)) if group is not None else None
    return default if value is None else value


def missing(ax, title, message='Not recorded'):
    ax.clear()
    ax.set_title(title, fontsize=10)
    ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes, color='gray')
    ax.set_axis_off()


def show_image(ax, data, title, **kwargs):
    if data is None or np.asarray(data).size == 0:
        missing(ax, title)
        return None
    ax.clear()
    ax.set_axis_on()
    artist = ax.imshow(data, **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return artist


def summarize(node, depth=0, max_depth=3):
    """Bound each inspection to one subtree, without loading large arrays."""
    prefix = '  ' * depth
    name = node.name.rsplit('/', 1)[-1] or '/'
    if isinstance(node, h5py.Dataset):
        if node.size <= 12:
            value = np.array2string(np.asarray(read_value(node)), threshold=12, max_line_width=90)
            return [f'{prefix}{name}: {value}']
        return [f'{prefix}{name}: array {node.shape}, {node.dtype} (open path to inspect)']
    lines = [f'{prefix}{name}/  {dict(node.attrs)}']
    if depth < max_depth:
        for child in node.values():
            lines.extend(summarize(child, depth + 1, max_depth))
    elif len(node):
        lines.append(f'{prefix}  ... {len(node)} children (enter this path to expand)')
    return lines


class EpisodeViewer:
    def __init__(self, reader, frame=0, fps=5, depth_max=10):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
        self.plt = plt
        self.reader = reader
        self.depth_max = depth_max
        self.index = -1
        self.step_name = None
        self.view_index = 0
        self.image_key = None
        self.playing = False
        self.inspectors = []
        self.map_markers = []
        self.closed = False
        self.fig = plt.figure(figsize=(19, 12))
        self.fig.canvas.manager.set_window_title(f'GC-VLN: {Path(reader.file.filename).name}')
        grid = self.fig.add_gridspec(4, 4, left=.04, right=.98, bottom=.16, top=.87,
                                    hspace=.55, wspace=.28, height_ratios=[1.2, .8, 1.35, 1.3])
        self.rgb_ax = self.fig.add_subplot(grid[0, :2])
        self.depth_ax = self.fig.add_subplot(grid[0, 2:])
        self.detect_ax = self.fig.add_subplot(grid[1, :3])
        self.info_ax = self.fig.add_subplot(grid[1, 3])
        self.map_axes = [self.fig.add_subplot(grid[2, i]) for i in range(4)]
        self.graph_ax = self.fig.add_subplot(grid[3, 0])
        self.nav_ax = self.fig.add_subplot(grid[3, 1])
        self.dag_ax = self.fig.add_subplot(grid[3, 2])
        self.trajectory_ax = self.fig.add_subplot(grid[3, 3], projection='3d')
        self.heading = self.fig.text(.04, .965, '', fontsize=13, weight='bold')
        self.instruction_text = self.fig.text(.04, .94, '', va='top', fontsize=10)
        self.status = self.fig.text(.04, .135, '', fontsize=9)
        self.fig.text(.04, .012, 'Drag slider | Left/Right: frame | Up/Down: step | Space: play/pause | V: camera | I: inspect', fontsize=9)
        self.slider = Slider(self.fig.add_axes([.12, .087, .77, .022]), 'Frame', 0,
                             max(1, len(reader.frames)-1), valinit=0, valstep=1, valfmt='%d')
        self.slider.set_active(len(reader.frames) > 1)
        self.slider.on_changed(lambda value: self.show_frame(int(value)))
        self.buttons = []
        for i, (label, callback) in enumerate([
            ('Previous', lambda event: self.seek(self.index-1)),
            ('Next', lambda event: self.seek(self.index+1)),
            ('Play / Pause', self.toggle_play),
            ('Previous step', lambda event: self.jump_step(-1)),
            ('Next step', lambda event: self.jump_step(1)),
            ('Camera +', self.cycle_view),
            ('Inspect HDF5', self.inspect),
        ]):
            button = Button(self.fig.add_axes([.06 + i*.13, .038, .12, .028]), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.timer = self.fig.canvas.new_timer(interval=max(1, round(1000/fps)))
        self.timer.add_callback(self.tick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', lambda event: self.close())
        instruction = get(reader.file, 'episode/instruction', {})
        text = (instruction.get('text_English') or instruction.get('text') or str(instruction)) if isinstance(instruction, dict) else str(instruction)
        self.instruction_text.set_text(textwrap.fill(text, 180, max_lines=3, placeholder=' ... [full text: Inspect HDF5]'))
        self.draw_graph(self.dag_ax, get(reader.file, 'episode/dag'), 'Instruction DAG', spatial=False)
        self.draw_trajectory()
        if reader.frames:
            self.seek(frame)
        else:
            self.heading.set_text('Episode contains no frames / steps. Use Inspect HDF5 for metadata and errors.')
            for ax in [self.rgb_ax, self.depth_ax, self.detect_ax, self.info_ax, *self.map_axes, self.graph_ax, self.nav_ax]:
                missing(ax, '')

    def seek(self, index):
        if not self.reader.frames:
            return
        index = int(np.clip(index, 0, len(self.reader.frames)-1))
        if self.slider.val != index:
            self.slider.set_val(index)
        else:
            self.show_frame(index)

    def cycle_view(self, event=None):
        self.view_index = (self.view_index + 1) % len(self.reader.views)
        if self.index >= 0:
            self.show_frame(self.index)

    def jump_step(self, direction):
        if self.index < 0:
            return
        current = self.reader.frames[self.index].step
        index = self.index + direction
        while 0 <= index < len(self.reader.frames):
            if self.reader.frames[index].step != current:
                if direction < 0:
                    target = self.reader.frames[index].step
                    while index > 0 and self.reader.frames[index-1].step == target:
                        index -= 1
                self.seek(index)
                return
            index += direction

    def toggle_play(self, event=None):
        if not self.reader.frames:
            return
        self.playing = not self.playing
        if self.playing:
            if self.index == len(self.reader.frames)-1:
                self.seek(0)
            self.timer.start()
        else:
            self.timer.stop()

    def tick(self):
        if self.index >= len(self.reader.frames)-1:
            self.playing = False
            self.timer.stop()
        else:
            self.seek(self.index+1)

    def on_key(self, event):
        actions = {'right': lambda: self.seek(self.index+1), 'left': lambda: self.seek(self.index-1),
                   'up': lambda: self.jump_step(1), 'down': lambda: self.jump_step(-1),
                   ' ': self.toggle_play, 'v': self.cycle_view, 'i': self.inspect,
                   'home': lambda: self.seek(0), 'end': lambda: self.seek(len(self.reader.frames)-1)}
        if event.key in actions:
            actions[event.key]()

    def draw_graph(self, ax, graph, title, spatial=True):
        if graph is None or not len(graph):
            missing(ax, title, 'Empty / not recorded')
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
        nx.draw_networkx(graph, pos=positions, ax=ax, node_size=45, width=.5,
                         font_size=6, with_labels=len(graph) <= 35, arrows=graph.is_directed())
        ax.set_title(f'{title}: {len(graph)} nodes / {graph.number_of_edges()} edges\n'
                     f'{"grid positions" if is_spatial else "topology layout"}', fontsize=9)

    def draw_detections(self, step):
        rgb = self.reader.panorama(self.index)
        if rgb is None:
            missing(self.detect_ax, 'Step perception panorama')
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
                    image[mask, :3] = (.65*image[mask, :3] + .35*np.asarray(colors(i % 20)[:3])*255).astype(image.dtype)
        show_image(self.detect_ax, image,
                   f'Step {self.step_name} perception: {len(boxes)} detections (before motion)')
        from matplotlib.patches import Rectangle
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = colors(i % 20)
            self.detect_ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=1))
            label = str(labels[i]) if i < len(labels) else str(i)
            if i < len(scores):
                label += f' {scores[i]:.2f}'
            self.detect_ax.text(x1, y1, label, fontsize=6, color='white',
                                bbox={'facecolor': color, 'alpha': .7, 'pad': 1})

    def draw_maps(self, step):
        self.map_markers = []
        for ax, key, title in zip(self.map_axes, ('bev', 'wall', 'thin', 'fmm'),
                                  ('BEV', 'Wall', 'Thin map', 'FMM + planning')):
            data = get(step, f'mapping/{key}')
            show_image(ax, data, title, cmap='gray', vmin=0, vmax=1, origin='upper')
            if data is not None:
                marker, = ax.plot([], [], 'o', color='cyan', markersize=5)
                heading, = ax.plot([], [], color='cyan', linewidth=2)
                self.map_markers.append((marker, heading))
        ax = self.map_axes[-1]
        if step.get('mapping/fmm') is None:
            return
        stage = get(step, 'planning/stage_before')
        stage_masks = mapping_item(step.get('planning/masks'), stage)
        if stage_masks is not None:
            valid = get(stage_masks, 'valid_mask')
            if valid is not None:
                overlay = np.zeros((*valid.shape, 4))
                overlay[valid.astype(bool)] = [0, 1, 0, .2]
                ax.imshow(overlay, origin='upper')
        candidates = step.get('planning/waypoints')
        if candidates is None:
            candidates = step.get('planning/candidates')  # legacy logs
        if candidates is not None:
            for entry in candidates.values():
                points = entry.get('points')
                if isinstance(points, h5py.Dataset) and points.ndim == 2 and points.shape[1] >= 2:
                    # Scatter sampling only; the file and inspector retain all points.
                    points = points[::max(1, int(np.ceil(len(points)/3000)))]
                    ax.scatter(points[:, 1], points[:, 0], s=2, color='orange', alpha=.45)
        selected = get(step, 'planning/selected_point')
        if selected is not None and np.asarray(selected).size == 2:
            row, col = selected
            ax.plot(col, row, '*', color='red', markersize=13)
        ax.set_title('FMM: orange=waypoints; red=goal', fontsize=8)

    def draw_trajectory(self):
        ax = self.trajectory_ax
        points = self.reader.positions
        valid = points[np.isfinite(points).all(axis=1)]
        if len(valid):
            ax.plot(points[:, 0], points[:, 2], points[:, 1], color='.8', linewidth=1)
        gt = self.reader.ground_truth()
        if len(gt):
            ax.plot(gt[:, 0], gt[:, 2], gt[:, 1], '--', color='green', linewidth=1, label='GT')
            ax.legend(fontsize=6)
        self.trail, = ax.plot([], [], [], color='royalblue', linewidth=1.5)
        self.current_pose, = ax.plot([], [], [], 'o', color='red', markersize=5)
        ax.set_title('World trajectory / GT (meters)', fontsize=9)
        ax.set_xlabel('X', fontsize=7)
        ax.set_ylabel('Z', fontsize=7)
        ax.set_zlabel('Y (up)', fontsize=7)
        ax.tick_params(labelsize=6)

    def show_frame(self, index):
        if not 0 <= index < len(self.reader.frames):
            return
        self.index = index
        ref = self.reader.frames[index]
        step, frame = self.reader.step(index), self.reader.frame(index)
        if self.step_name != ref.step:
            self.step_name = ref.step
            self.draw_detections(step)
            self.draw_maps(step)
            self.draw_graph(self.graph_ax, get(step, 'scene_graph'), 'Scene graph')
            self.draw_graph(self.nav_ax, get(step, 'planning/navigation_tree/navigation_tree'), 'Navigation tree')
        view = self.reader.views[self.view_index]
        if self.image_key != (ref.step, view):
            self.image_key = (ref.step, view)
            rgb, depth = self.reader.images(index, view)
            show_image(self.rgb_ax, rgb, f'Step {ref.step} RGB — {view} (fixed during motion)')
            if depth is not None:
                depth = np.asarray(depth)
                if depth.ndim == 3 and depth.shape[-1] == 1:
                    depth = depth[..., 0]
                depth = np.ma.masked_where(~np.isfinite(depth) | (depth <= 0), depth)
            depth_sensor = view.replace('rgb', 'depth', 1)
            normalized = get(self.reader.file, f'episode/cameras/{depth_sensor}/NORMALIZE_DEPTH', False)
            limit, unit = (1, 'normalized') if normalized else (self.depth_max, 'm')
            show_image(self.depth_ax, depth, f'Step {ref.step} depth — {depth_sensor} [0–{limit:g} {unit}]',
                       cmap='viridis', vmin=0, vmax=limit)
        episode_id = get(self.reader.file, 'episode/id', self.reader.file.attrs.get('episode_id', '?'))
        scene = get(self.reader.file, 'episode/scene', '?')
        self.heading.set_text(f'Episode {episode_id} | {Path(str(scene)).name} | '
                              f'Frame {index}/{len(self.reader.frames)-1} | Step {ref.step} / {ref.local_index} | {ref.event}')
        action = get(frame, 'action') if ref.event != 'planning' else None
        names = {0: 'STOP', 1: 'FORWARD', 2: 'TURN_LEFT', 3: 'TURN_RIGHT'}
        action_name = names.get(action, str(action)) if np.isscalar(action) else '—'
        pose = get(frame, 'after/position')
        heading = get(frame, 'after/heading')
        stage = get(step, 'planning/stage', '—')
        info = [f'Action: {action_name}  Event: {ref.event}',
                f'Stage: {stage}  {get(step, "planning/stage_result", "")}',
                f'Position XYZ: {np.array2string(pose, precision=3) if pose is not None else "—"}',
                f'Heading: {float(heading):.3f} rad' if heading is not None else 'Heading: —',
                f'Stuck count: {get(frame, "stuck_time", "—")}',
                f'Teleport: {get(frame, "teleport", False)}',
                f'Result: {get(step, "motion/action_valid", "—")}',
                f'Goal: {get(step, "planning/selected_point", "—")}']
        reason = get(frame, 'reason', '')
        if reason:
            info.append(textwrap.fill(str(reason), 42))
        self.info_ax.clear()
        self.info_ax.set_axis_off()
        self.info_ax.text(0, 1, '\n'.join(info), va='top', fontsize=8, transform=self.info_ax.transAxes)
        gps, compass = self.reader.local_pose(index)
        size = get(self.reader.file, 'episode/config/POLICY_CONFIG/MAP/SIZE')
        resolution = get(self.reader.file, 'episode/config/POLICY_CONFIG/MAP/RESOLUTION')
        for marker, arrow in self.map_markers:
            marker.set_data([], [])
            arrow.set_data([], [])
            if gps is not None and size is not None and resolution is not None:
                row, col = -gps[1]*resolution + size//2, gps[0]*resolution + size//2
                marker.set_data([col], [row])
                if compass is not None:
                    angle = np.asarray(compass).ravel()[0]
                    arrow.set_data([col, col+resolution*np.cos(angle)], [row, row+resolution*np.sin(angle)])
        points = self.reader.positions[:index+1]
        self.trail.set_data_3d(points[:, 0], points[:, 2], points[:, 1])
        point = points[-1]
        self.current_pose.set_data_3d([point[0]], [point[2]], [point[1]])
        error = get(step, 'error', get(step, 'motion/error', ''))
        error_line = str(error).strip().splitlines()[-1] if error else ''
        self.status.set_text(f'{"ERROR: " + error_line if error else "RGB-D = Step observation; motion poses update per frame. Cyan = current agent."}'
                             '  Dense waypoints are sampled for display.')
        self.fig.canvas.draw_idle()

    def inspect(self, event=None):
        from matplotlib.widgets import Slider, TextBox
        if self.playing:
            self.toggle_play()
        fig = self.plt.figure(figsize=(12, 8))
        ax = fig.add_axes([.03, .14, .93, .73])
        ax.set_axis_off()
        text = ax.text(0, 1, '', va='top', family='monospace', fontsize=9, transform=ax.transAxes)
        initial = f'/steps/{self.step_name}/planning/constraints' if self.step_name else '/episode'
        box = TextBox(fig.add_axes([.12, .91, .8, .04]), 'HDF5 path', initial=initial)
        slider = Slider(fig.add_axes([.12, .05, .8, .025]), 'Line', 0, 1, valstep=1)
        lines = []

        def scroll(value):
            start = int(value)
            text.set_text('\n'.join(lines[start:start+36]))
            fig.canvas.draw_idle()

        def load(path):
            nonlocal lines
            node = self.reader.file.get(path)
            if node is None:
                lines = [f'Not recorded: {path}']
            elif isinstance(node, h5py.Dataset) and node.size > 12:
                # A selected large dataset is still summarized, never dumped in RAM.
                lines = [f'{node.name}: shape={node.shape}, dtype={node.dtype}',
                         'Array preview (first slice):', str(node[tuple(slice(0, min(4, n)) for n in node.shape)])]
            else:
                lines = summarize(node)
            lines = [part for line in lines for part in line.splitlines()]
            slider.valmax = max(1, len(lines)-36)
            slider.ax.set_xlim(0, slider.valmax)
            slider.set_val(0)
            scroll(0)

        box.on_submit(load)
        slider.on_changed(scroll)
        fig.canvas.mpl_connect('scroll_event', lambda event: slider.set_val(
            np.clip(slider.val - event.step*3, 0, slider.valmax)))
        # Hold widgets/callbacks alive for the lifetime of the window.
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
    parser.add_argument('--backend', help='Matplotlib backend, e.g. TkAgg or QtAgg')
    parser.add_argument('--save', type=Path, help='Save one frame as PNG without opening a GUI')
    args = parser.parse_args()
    if args.fps <= 0 or args.depth_max <= 0 or args.frame < 0:
        parser.error('--fps and --depth-max must be positive; --frame must be nonnegative')
    if args.save:
        matplotlib.use('Agg')
    elif args.backend:
        matplotlib.use(args.backend)
    import matplotlib.pyplot as plt
    try:
        reader = EpisodeReader(args.episode)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    viewer = None
    try:
        viewer = EpisodeViewer(reader, args.frame, args.fps, args.depth_max)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            viewer.fig.savefig(args.save, dpi=140)
            print(f'Saved {args.save}')
        else:
            if matplotlib.get_backend().lower() == 'agg':
                parser.error('No interactive backend. Use --backend TkAgg/QtAgg on a desktop, or --save preview.png.')
            plt.show()
    finally:
        if viewer is not None:
            viewer.close()
            plt.close(viewer.fig)
        else:
            reader.close()


if __name__ == '__main__':
    main()
