"""Read the debug-log schema without Habitat, CUDA, or pickle."""
from dataclasses import dataclass

import h5py
import networkx as nx
import numpy as np


def read_value(node):
    if node is None:
        return None
    if isinstance(node, h5py.Dataset):
        if h5py.check_string_dtype(node.dtype) is not None:
            return node.asstr()[()]
        return node[()]
    kind = node.attrs.get('type', 'dict')
    if kind == 'none':
        return None
    if kind == 'networkx':
        directed, multi = node.attrs['directed'], node.attrs['multigraph']
        cls = (nx.MultiDiGraph if directed else nx.MultiGraph) if multi else (nx.DiGraph if directed else nx.Graph)
        graph = cls()
        graph.graph.update(read_value(node.get('attributes')) or {})
        for entry in node['nodes'].values():
            graph.add_node(read_value(entry['id']))
            graph.nodes[read_value(entry['id'])].update(read_value(entry['attributes']))
        for entry in node['edges'].values():
            source, target = read_value(entry['source']), read_value(entry['target'])
            attrs = read_value(entry['attributes'])
            if multi:
                key = read_value(entry['key'])
                graph.add_edge(source, target, key=key)
                graph.edges[source, target, key].update(attrs)
            else:
                graph.add_edge(source, target)
                graph.edges[source, target].update(attrs)
        return graph
    if kind in ('list', 'tuple', 'set'):
        values = [read_value(node[key]) for key in sorted(node)]
        if 'array_shape' in node.attrs:
            return np.array(values, dtype=object).reshape(node.attrs['array_shape'])
        return {'list': list, 'tuple': tuple, 'set': set}[kind](values)
    if kind == 'mapping':
        return {read_value(entry['key']): read_value(entry['value']) for entry in node.values()}
    return {key: read_value(value) for key, value in node.items()}


def mapping_item(node, key):
    """Find a typed mapping entry without loading its potentially large masks."""
    if node is None or not isinstance(node, h5py.Group) or node.attrs.get('type') == 'none':
        return None
    if node.attrs.get('type') == 'mapping':
        for entry in node.values():
            if read_value(entry['key']) == key:
                return entry['value']
        return None
    return node.get(str(key))


@dataclass(frozen=True)
class FrameRef:
    step: str
    path: str
    event: str
    local_index: str


class EpisodeReader:
    """Index small frame metadata; read images/maps only when requested."""
    def __init__(self, path):
        self.file = h5py.File(path, 'r')
        try:
            if 'steps' not in self.file:
                raise ValueError('Not an episode log: missing /steps')
            self.frames = []
            self.positions = []
            self.views = set()
            self._image_cache_key = None
            self._image_cache = (None, None)
            for step_name, step in sorted(self.file['steps'].items()):
                rgbd = step.get('rgbd')
                if rgbd is not None:
                    self.views.update(key for key in rgbd if key == 'rgb' or key.startswith('rgb_'))
                frames = step.get('motion/frames')
                entries = sorted(frames.items()) if frames is not None and len(frames) else [(None, step)]
                for frame_name, frame in entries:
                    self.frames.append(FrameRef(step_name, frame.name,
                        str(frame.attrs.get('event', 'planning')), frame_name or '-'))
                    position = read_value(frame.get('after/position'))
                    self.positions.append(position if position is not None else [np.nan]*3)
            self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3)
            self.views = sorted(self.views, key=lambda key: 0 if key == 'rgb' else int(key.split('_')[-1])) or ['rgb']
        except BaseException:
            self.file.close()
            raise

    def close(self):
        self.file.close()

    def step(self, index):
        return self.file['steps'][self.frames[index].step]

    def frame(self, index):
        return self.file[self.frames[index].path]

    def images(self, index, view='rgb'):
        """Step observation, cached while moving through its motion-only frames."""
        key = (self.frames[index].step, view)
        if key != self._image_cache_key:
            group = self.step(index).get('rgbd')
            self._image_cache = (None, None) if group is None else (
                read_value(group.get(view)), read_value(group.get(view.replace('rgb', 'depth', 1))))
            self._image_cache_key = key
        return self._image_cache

    def panorama(self, index):
        """Reproduce rgbs_to_panorama + rotate_180 exactly from stored views."""
        group = self.step(index).get('rgbd')
        if group is None:
            return None
        if 'panorama_rgb' in group:  # schema v1
            return read_value(group['panorama_rgb'])
        names = ['rgb'] + [f'rgb_{angle}' for angle in range(30, 360, 30)]
        if any(name not in group for name in names):
            return None
        width = group['rgb'].shape[1]
        strips = [group[name][:, int(width*.366):int(width*.634), :] for name in names]
        panorama = np.concatenate(strips, axis=1)
        split = panorama.shape[1]//2
        return np.concatenate([panorama[:, split:], panorama[:, :split]], axis=1)

    @staticmethod
    def _rotation(quaternion):
        x, y, z, w = np.asarray(quaternion, dtype=float)
        norm = x*x + y*y + z*z + w*w
        if norm == 0:
            raise ValueError('Zero quaternion in pose log')
        return np.eye(3) + 2/norm * np.array([
            [-y*y-z*z, x*y-z*w, x*z+y*w],
            [x*y+z*w, -x*x-z*z, y*z-x*w],
            [x*z-y*w, y*z+x*w, -x*x-y*y]])

    def local_pose(self, index):
        """Recover Habitat GPS/compass from world pose; no per-frame sensors needed."""
        frame = self.frame(index)
        # Legacy frame observations and steps without motion are still readable.
        if 'rgbd/gps' in frame:
            return read_value(frame['rgbd/gps']), read_value(frame.get('rgbd/compass'))
        position = read_value(frame.get('after/position'))
        rotation = read_value(frame.get('after/rotation_xyzw'))
        origin = read_value(self.file.get('episode/start_position'))
        start_rotation = read_value(self.file.get('episode/start_rotation'))
        if position is None or origin is None or start_rotation is None:
            return None, None
        start = self._rotation(start_rotation)
        local = start.T @ (np.asarray(position) - origin)
        compass = None
        if rotation is not None:
            forward = self._rotation(rotation).T @ start @ np.array([0, 0, -1])
            compass = np.array([np.arctan2(forward[0], -forward[2])])
        return np.array([-local[2], local[0]]), compass

    def ground_truth(self):
        for path in ('episode/gt/reference_path', 'episode/gt/trajectory/locations'):
            value = read_value(self.file.get(path))
            if value is not None:
                points = np.asarray(value, dtype=float)
                if points.ndim == 2 and points.shape[1] == 3:
                    return points
        return np.empty((0, 3))
