"""CPU-only tests: python -m unittest discover -s tests -v."""
import ast
import multiprocessing
import json
import os
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import h5py
import networkx as nx
import numpy as np

from src.debug_log import EpisodeLog, MotionLog, write_value, planning_snapshot


def observation():
    return {'rgb': np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
            'rgb_30': np.zeros((2, 3, 3), dtype=np.uint8),
            'depth': np.array([0, np.nan, np.inf, 1.234567, 9.9, 0.1], dtype=np.float32).reshape(2, 3, 1),
            'depth_30': np.zeros((2, 3, 1), dtype=np.float32),
            'gps': np.zeros(2), 'compass': np.zeros(1)}


def worker_write(request):
    with MotionLog(request) as writer:
        writer.record('action', None, {'position': np.zeros(3)}, action=1)


def load_methods(path, class_name, names, namespace):
    """Exercise real integration methods without importing Habitat/GLIP/CUDA."""
    tree = ast.parse(Path(path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.bases = []
    cls.decorator_list = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), path, 'exec'), namespace)
    return namespace[class_name]


class DebugLogTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = EpisodeLog.create(self.temp.name, '测试/42', {'instruction': '向左转', 'gt': None})
        self.log.begin_step(0, observation())

    def test_lossless_arrays_graphs_constraints_and_keys(self):
        graph = nx.MultiDiGraph(description='完整图')
        graph.add_node(1, center=np.array([2, 3], dtype=np.int32))
        graph.add_node('1', caption='椅子')
        graph.add_edge(1, '1', key='a', score=np.float32(0.8))
        graph.add_edge(1, '1', key='b', constraint=SimpleNamespace(center=np.ones(2), mask=None))
        self.log.write('steps/000000', scene_graph=graph,
                       constraints={1: 'integer', '1': 'string', ('x/y', 2): []},
                       masks=np.empty((0, 2, 3), dtype=np.uint8),
                       labels=np.array(['椅子', '门']), scalar=np.array(3))
        with h5py.File(self.log.path) as file:
            depth = file['steps/000000/rgbd/depth']
            self.assertEqual(depth.dtype, np.float32)
            np.testing.assert_array_equal(depth[:], observation()['depth'])
            self.assertEqual(depth.compression, 'lzf')
            self.assertEqual(file['episode/instruction'].asstr()[()], '向左转')
            graph = file['steps/000000/scene_graph']
            self.assertEqual(len(graph['nodes']), 2)
            self.assertEqual(len(graph['edges']), 2)
            self.assertTrue(graph.attrs['directed'])
            self.assertTrue(graph.attrs['multigraph'])
            self.assertEqual(graph['nodes/000001/id'].asstr()[()], '1')
            self.assertEqual(graph['edges/000001/attributes/constraint/mask'].attrs['type'], 'none')
            self.assertEqual(file['steps/000000/masks'].shape, (0, 2, 3))
            self.assertEqual(len(file['steps/000000/constraints']), 3)
            self.assertEqual(file['steps/000000/labels'].asstr()[:].tolist(), ['椅子', '门'])

    def test_worker_handoff_and_reopen(self):
        process = multiprocessing.get_context('spawn').Process(
            target=worker_write, args=(self.log.motion_request(0),))
        process.start()
        process.join(20)
        if process.is_alive():
            process.terminate()
            process.join()
            self.fail('worker timed out')
        self.assertEqual(process.exitcode, 0)
        self.log.write('steps/000000', done=True)
        self.log.finish('complete', metrics={'spl': 1.0})
        with h5py.File(self.log.path) as file:
            self.assertEqual(file.attrs['status'], 'complete')
            self.assertEqual(file['steps/000000/motion/frames/000000/action'][()], 1)
            self.assertEqual(file['metrics/spl'][()], 1)

    def test_exception_keeps_frames_and_closes_handle(self):
        with self.assertRaisesRegex(RuntimeError, 'simulation failed'):
            with MotionLog(self.log.motion_request(0)) as writer:
                writer.record('start', None, None)
                raise RuntimeError('simulation failed')
        self.log.finish('interrupted')
        with h5py.File(self.log.path) as file:
            self.assertEqual(file['steps/000000/motion'].attrs['status'], 'error')
            self.assertEqual(len(file['steps/000000/motion/frames']), 1)
            self.assertEqual(file['steps/000000/motion/error'].asstr()[()], 'simulation failed')

    def test_repeated_episode_never_overwrites(self):
        second = EpisodeLog.create(self.temp.name, '测试/42', {})
        self.assertNotEqual(second.path, self.log.path)
        self.assertEqual(len(list(Path(self.temp.name).glob('*.h5'))), 2)

    def test_compact_frames_and_planning_exclude_large_intermediates(self):
        large = np.ones((128, 128), dtype=np.float32)
        pose = {'position': np.zeros(3), 'rotation_xyzw': np.array([0, 0, 0, 1]),
                'heading': 0., 'sensors': {'depth': large}}
        with MotionLog(self.log.motion_request(0)) as writer:
            writer.record('action', pose, pose, action=1)
        solver = SimpleNamespace(stage=1, debug_candidates=[{'stage': 1, 'points': np.ones((2, 2))}],
            best_point=np.ones(2), navigation_mode='direction',
            constraints={1: {'chair': {'constraint': [SimpleNamespace(
                type='near', center=np.zeros(2), mask=large, device='cuda')]}}},
            navigation_tree=SimpleNamespace(navigation_tree=nx.Graph(), waypoints_tree=nx.Graph(),
                path=[], stage_begin={}, existed_point_mask=large, instruction_graph=nx.Graph()))
        self.log.write('steps/000000', planning=planning_snapshot(solver, 1, 'continue', [1, 1]))
        with h5py.File(self.log.path) as file:
            self.assertEqual(file.attrs['schema_version'], 2)
            frame = file['steps/000000/motion/frames/000000']
            self.assertEqual(set(frame), {'action', 'before', 'after', 'stuck_time', 'teleport', 'reason'})
            self.assertEqual(set(frame['after']), {'position', 'rotation_xyzw', 'heading'})
            planning = file['steps/000000/planning']
            self.assertNotIn('masks', planning)
            self.assertNotIn('candidates', planning)
            self.assertNotIn('existed_point_mask', planning['navigation_tree'])
            constraint = planning['constraints/000000/value/chair/constraint/000000']
            self.assertEqual(set(constraint), {'type', 'center'})

    def test_compression_options_and_empty_detections(self):
        for compression in ('none', 'gzip'):
            log = EpisodeLog.create(self.temp.name, compression, {}, compression)
            log.begin_step(0, observation())
            log.write('steps/000000', perception={'boxes': np.empty((0, 4)),
                                                 'labels': np.array([], dtype=str)})
            with h5py.File(log.path) as file:
                self.assertEqual(file['steps/000000/rgbd/depth'].compression,
                                 None if compression == 'none' else compression)
                self.assertEqual(file['steps/000000/perception/boxes'].shape, (0, 4))


class MotionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = EpisodeLog.create(self.temp.name, 'motion', {})
        self.log.begin_step(0, observation())
        cls = load_methods('src/agent/environments.py', 'GCVLNEnv',
            {'_record_motion', 'teleport', 'wrap_act', 'step', '_step_impl'},
            {'np': np, 'MotionLog': MotionLog, 'write_value': write_value})
        self.env = cls()
        env = self.env
        env._motion_log = None
        env.stuck_time = 0
        env.video_option = []
        env.tele_flag = False
        env.pre_episode_id = 'motion'
        env.gcvln_config = {'seed': 42}
        self.position = np.zeros(3)
        self.executed = []
        sim = SimpleNamespace(
            sensor_suite=SimpleNamespace(sensors={'depth': SimpleNamespace(
                config={'HFOV': 90, 'NORMALIZE_DEPTH': False})}),
            get_agent_state=lambda: SimpleNamespace(position=self.position, rotation=None),
            step_without_obs=self.executed.append,
            set_agent_state=lambda position, rotation: setattr(self, 'position', np.array(position)))
        measurements = SimpleNamespace(update_measures=lambda **kwargs: None)
        task = SimpleNamespace(measurements=measurements)
        env._env = SimpleNamespace(sim=sim, _task=task, task=task,
            current_episode=SimpleNamespace(episode_id='motion'),
            step=lambda action: (self.executed.append(action) or observation()))
        env._debug_pose = lambda: {'position': self.position.copy(), 'heading': 0.0}
        env.get_observation_at = lambda *args: observation()
        env.get_reward = lambda obs: 0
        env.get_done = lambda obs: True
        env.get_info = lambda obs: {'spl': 1.0}

    def test_stop_records_motion_only_without_rendering(self):
        self.env.get_observation_at = lambda *args: self.fail('logging must not render RGB-D')
        action = {'act': 0, 'next_point': None, 'current_pos': [0, 0],
                  'current_heading': 0, 'location': [0, 0]}
        result = self.env.step(action, None, None, debug_log=self.log.motion_request(0))
        self.assertTrue(result[2])
        self.assertEqual(self.executed, [0])
        self.assertIsNone(self.env._motion_log)
        with h5py.File(self.log.path) as file:
            frames = file['steps/000000/motion/frames']
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames['000001/action'][()], 0)
            self.assertNotIn('rgbd', frames['000001'])
            np.testing.assert_array_equal(file['steps/000000/rgbd/depth'][:], observation()['depth'])

    def test_every_low_level_action_teleport_and_stuck(self):
        self.env._env.sim._prev_sim_obs = {'collided': True}
        self.env.get_observation_at = lambda *args: self.fail('logging must not render RGB-D')
        with MotionLog(self.log.motion_request(0)) as writer:
            self.env._motion_log = writer
            for action in (1, 2, 3):
                self.env.wrap_act(action, None)
            self.env._record_motion('stuck', self.env._debug_pose(), reason='blocked')
            self.env.teleport([1, 2, 3], None, reason='test recovery')
        self.env._motion_log = None
        self.assertEqual(self.executed, [1, 2, 3])
        self.assertEqual(self.env._env.sim._prev_sim_obs, {'collided': True})
        with h5py.File(self.log.path) as file:
            frames = file['steps/000000/motion/frames']
            self.assertEqual([frame.attrs['event'] for frame in frames.values()],
                             ['action', 'action', 'action', 'stuck', 'teleport'])
            np.testing.assert_array_equal(frames['000004/before/position'][:], [0, 0, 0])
            np.testing.assert_array_equal(frames['000004/after/position'][:], [1, 2, 3])

    def test_disabled_logging_does_not_render_extra_frames(self):
        self.env.get_observation_at = lambda *args: self.fail('unexpected render')
        self.env.wrap_act(1, None)
        self.assertEqual(self.executed, [1])


class PolicyIntegrationTests(unittest.TestCase):
    def test_episode_alignment_after_initial_and_mid_rollout_pause(self):
        with TemporaryDirectory() as directory:
            namespace = dict(np=np, os=os, json=json, traceback=traceback, EpisodeLog=EpisodeLog,
                planning_snapshot=planning_snapshot,
                get_instruction_graphs=lambda obs: ([1]*3, [nx.DiGraph() for _ in obs], [[]]*3, []),
                get_pose_matrix=lambda obs: np.eye(4),
                rgbs_to_panorama=lambda obs: [obs['rgb'], obs['depth']],
                rotate_180=lambda *args: args, save_video=lambda *args: None)
            cls = load_methods('src/agent/agent_policy.py', 'Agent',
                               {'rollout', '_pause_envs'}, namespace)
            agent = cls()
            episodes = [SimpleNamespace(episode_id=str(i)) for i in range(3)]
            class Envs:
                num_envs = 3
                calls = 0

                def resume_all(self):
                    pass

                def reset(self):
                    return [observation() for _ in episodes]

                def current_episodes(self):
                    return episodes

                def pause_at(self, index):
                    episodes.pop(index)
                    self.num_envs -= 1

                def step(self, actions):
                    self.calls += 1
                    results = []
                    for ep, action in zip(episodes, actions):
                        # The worker can open the parent's file while executing.
                        worker_write(action['debug_log'])
                        results.append((observation(), 0, ep.episode_id == '1' or self.calls == 2,
                                        {'action_valid': 'complete', 'spl': 1.0}))
                    return results

            agent.envs = Envs()
            agent.stat_eps = {'0': {}}
            agent.episode_logs = {}
            paths = {}
            def start_logs(obs):
                for i in (1, 2):
                    log = EpisodeLog.create(directory, str(i), {'id': str(i)})
                    agent.episode_logs[i] = log
                    paths[str(i)] = log.path
            agent._start_episode_logs = start_logs
            agent._reset_policy = lambda *args: None
            agent.sg_list = []
            agent.rs_list = []
            for i in range(3):
                map_draft = SimpleNamespace(**{k: np.full((4, 4), i, dtype=np.int32)
                    for k in ('bev_map', 'bev_map_fmm', 'bev_wall', 'bev_thin')}, scene_graph=nx.Graph())
                agent.sg_list.append(SimpleNamespace(map_draft=map_draft,
                    get_scenegraph=lambda *args: False, last_room_detection=None,
                    last_detection={'boxes': np.empty((0, 4)), 'labels': np.array([], dtype=str),
                                    'masks': np.empty((0, 2, 3)), 'scores': np.empty(0)}))
                agent.rs_list.append(SimpleNamespace(stage=i, constraints={}, masks={},
                    debug_candidates=[], final_pts={}, best_point=np.array([i, i]),
                    navigation_mode='direction',
                    navigation_tree=SimpleNamespace(stage_begin={1: [i, i]}, waypoints_tree=nx.Graph()),
                    get_next_point=lambda *args, point=i: (np.array([point, point]), 'continue')))
            agent.max_len = 2
            agent.thin_type = 1
            agent.camera_matrix = np.eye(3)
            agent.config = SimpleNamespace(
                POLICY_CONFIG=SimpleNamespace(MAP=SimpleNamespace(RESOLUTION=20, SIZE=4)),
                TASK_CONFIG=SimpleNamespace(DATASET=SimpleNamespace(SPLIT='test')),
                DATASET=SimpleNamespace(SPLIT_INDEX=0, SPLIT_NUM=1))
            agent.pbar = SimpleNamespace(update=lambda: None)
            agent.calculate_metric = lambda info: {'spl': info['spl']}
            agent.log_save_dir = directory
            agent.visualization_save_dir = directory
            agent.local_rank = 0
            agent.world_size = 1
            agent.rollout()
            self.assertFalse(agent.episode_logs)
            for episode_id, step_count in [('1', 1), ('2', 2)]:
                with h5py.File(paths[episode_id]) as file:
                    self.assertEqual(file.attrs['status'], 'complete')
                    self.assertEqual(len(file['steps']), step_count)
                    for step in file['steps'].values():
                        np.testing.assert_array_equal(step['mapping/bev'][:], int(episode_id))
                        self.assertEqual(step['planning/stage'][()], int(episode_id))
                        self.assertEqual(step['perception/detections/boxes'].shape, (0, 4))
                        self.assertIn('motion/frames/000000', step)


if __name__ == '__main__':
    unittest.main()
