"""Lossless HDF5 debug snapshots, without importing Habitat or GPU models.

The policy and environment worker take turns opening each episode file. No HDF5
handle crosses a process boundary or stays open across VectorEnv.step().
"""
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4
import time

import h5py
import networkx as nx
import numpy as np


def write_value(parent, name, value, compression="lzf"):
    """Write nested values as datasets/groups (never pickle or truncate graphs)."""
    if name in parent:
        del parent[name]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if value is None:
        group = parent.create_group(name)
        group.attrs["type"] = "none"
        return group
    if isinstance(value, nx.Graph):
        group = parent.create_group(name)
        group.attrs.update(type="networkx", directed=value.is_directed(),
                           multigraph=value.is_multigraph())
        write_value(group, "attributes", value.graph, compression)
        nodes = group.create_group("nodes")
        for i, (node, attrs) in enumerate(value.nodes(data=True)):
            write_value(nodes, f"{i:06d}", {"id": node, "attributes": attrs}, compression)
        edges = group.create_group("edges")
        iterator = value.edges(data=True, keys=True) if value.is_multigraph() else value.edges(data=True)
        for i, edge in enumerate(iterator):
            data = {"source": edge[0], "target": edge[1], "attributes": edge[-1]}
            if value.is_multigraph():
                data["key"] = edge[2]
            write_value(edges, f"{i:06d}", data, compression)
        return group
    if isinstance(value, Mapping):
        group = parent.create_group(name)
        # Non-string keys are stored explicitly: 1, "1", tuples cannot collide.
        plain = all(isinstance(k, str) and k and "/" not in k and "\x00" not in k
                    and k not in (".", "..") for k in value)
        group.attrs["type"] = "dict" if plain else "mapping"
        for i, (key, item) in enumerate(value.items()):
            if plain:
                write_value(group, key, item, compression)
            else:
                entry = group.create_group(f"{i:06d}")
                write_value(entry, "key", key, compression)
                write_value(entry, "value", item, compression)
        return group
    if isinstance(value, (list, tuple, set)):
        group = parent.create_group(name)
        group.attrs["type"] = type(value).__name__
        for i, item in enumerate(value):
            write_value(group, f"{i:06d}", item, compression)
        return group
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "U":
            return parent.create_dataset(name, data=value.astype(object), dtype=h5py.string_dtype("utf-8"))
        if value.dtype.kind == "O":
            group = write_value(parent, name, value.tolist(), compression)
            group.attrs["array_shape"] = value.shape
            return group
        options = {"compression": compression, "shuffle": True} if value.ndim and value.size and compression else {}
        return parent.create_dataset(name, data=value, **options)
    if isinstance(value, (str, np.str_)):
        return parent.create_dataset(name, data=str(value), dtype=h5py.string_dtype("utf-8"))
    if isinstance(value, (bool, int, float, complex, np.number, np.bool_)):
        return parent.create_dataset(name, data=value)
    if isinstance(value, Path) or type(value).__name__ == "device":
        return write_value(parent, name, str(value), compression)
    # Habitat episodes use attrs (including slots); constraints use __dict__.
    if hasattr(value, "__attrs_attrs__"):
        data = {field.name: getattr(value, field.name) for field in value.__attrs_attrs__}
    elif hasattr(value, "__dict__"):
        data = vars(value)
    else:
        raise TypeError(f"Unsupported debug value at {parent.name}/{name}: {type(value)}")
    group = write_value(parent, name, data, compression)
    group.attrs["python_class"] = f"{type(value).__module__}.{type(value).__name__}"
    return group


def rgbd_snapshot(observation):
    """Keep sensor output dtype/shape/values, including all panorama views."""
    return {key: value for key, value in observation.items()
            if key in ("gps", "compass") or key == "rgb" or key == "depth"
            or key.startswith(("rgb_", "depth_"))}


def constraint_parameters(value):
    """Keep constraint geometry/relations, excluding generated raster masks."""
    if isinstance(value, Mapping):
        return {key: constraint_parameters(item) for key, item in value.items()
                if key not in ('mask', 'device')}
    if isinstance(value, (list, tuple)):
        return [constraint_parameters(item) for item in value]
    if hasattr(value, '__dict__') and not hasattr(value, 'detach'):
        return constraint_parameters(vars(value))
    return value


def planning_snapshot(solver, stage_before, stage_result, selected_point):
    tree = solver.navigation_tree
    return {
        'stage_before': stage_before, 'stage': solver.stage,
        'stage_result': stage_result,
        'constraints': constraint_parameters(solver.constraints),
        'waypoints': solver.debug_candidates,
        'selected_point': selected_point, 'best_point': solver.best_point,
        'navigation_mode': solver.navigation_mode,
        'navigation_tree': {key: getattr(tree, key) for key in
                            ('navigation_tree', 'waypoints_tree', 'path', 'stage_begin')
                            if hasattr(tree, key)},
    }


class EpisodeLog:
    def __init__(self, path, compression="lzf"):
        self.path = str(path)
        self.compression = None if compression == "none" else compression

    @classmethod
    def create(cls, directory, episode_id, metadata, compression="lzf"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        name = quote(str(episode_id), safe="")[:100]
        log = cls(directory / f"episode_{name}_{uuid4().hex[:12]}.h5", compression)
        with h5py.File(log.path, "x") as file:
            file.attrs.update(schema_version=2, status="running", episode_id=str(episode_id),
                              created_at=datetime.now(timezone.utc).isoformat())
            write_value(file, "episode", metadata, log.compression)
            file.create_group("steps")
        return log

    def write(self, path, **values):
        with h5py.File(self.path, "a") as file:
            group = file.require_group(path)
            for name, value in values.items():
                write_value(group, name, value, self.compression)
            file.flush()

    def begin_step(self, step, observation):
        self.write(f"steps/{step:06d}", rgbd=rgbd_snapshot(observation))
        # Explicitly unavailable until that module has run; no stale snapshots.
        self.write(f"steps/{step:06d}", **{k: None for k in
                   ("perception", "mapping", "scene_graph", "planning")})

    def motion_request(self, step):
        return {"path": self.path, "step": step, "compression": self.compression}

    def finish(self, status, **values):
        with h5py.File(self.path, "a") as file:
            for name, value in values.items():
                write_value(file, name, value, self.compression)
            file.attrs["status"] = status
            file.attrs["finished_at"] = datetime.now(timezone.utc).isoformat()


class MotionLog:
    """Worker-owned writer, closed before returning a VectorEnv step result."""
    def __init__(self, request):
        self.request = request
        self.file = None
        self.index = 0

    def __enter__(self):
        self.file = h5py.File(self.request["path"], "a")
        try:
            self.group = self.file.require_group(f"steps/{self.request['step']:06d}/motion")
            self.frames = self.group.create_group("frames")
            self.group.attrs["status"] = "running"
        except BaseException:
            self.file.close()
            raise
        return self

    def record(self, event, before, after, action=None, stuck_time=0, teleport=False, reason=''):
        frame = self.frames.create_group(f"{self.index:06d}")
        frame.attrs.update(event=event, frame_index=self.index, step_index=self.request["step"],
                           timestamp_ns=time.time_ns(),
                           planning_context=f"/steps/{self.request['step']:06d}")
        def pose_only(pose):
            if pose is None:
                return None
            return {key: pose.get(key) for key in ('position', 'rotation_xyzw', 'heading')}
        values = dict(action=action, before=pose_only(before), after=pose_only(after),
                      stuck_time=stuck_time, teleport=teleport, reason=reason)
        for name, value in values.items():
            write_value(frame, name, value, self.request["compression"])
        self.index += 1
        self.file.flush()

    def __exit__(self, exc_type, exc, tb):
        try:
            self.group.attrs["status"] = "error" if exc is not None else "complete"
            if exc is not None:
                write_value(self.group, "error", str(exc))
            self.file.flush()
        finally:
            self.file.close()
