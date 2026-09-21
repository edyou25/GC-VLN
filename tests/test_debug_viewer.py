from pathlib import Path
import ast
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
import networkx as nx
import numpy as np

from src.debug_log import EpisodeLog, MotionLog, write_value
from src.debug_log_reader import EpisodeReader, read_value
from scripts.view_hdf5_log import EpisodeViewer


def make_episode(directory):
    dag = nx.DiGraph()
    dag.add_edge(1, 'chair', relation='near')
    log = EpisodeLog.create(directory, 'viewer_test', {
        'id': 'viewer_test', 'scene': 'sample_scene.glb',
        'instruction': {'text': 'Walk past the chair and turn left.'},
        'dag': dag,
        'config': {'POLICY_CONFIG': {'MAP': {'SIZE': 32, 'RESOLUTION': 2}}},
        'start_position': np.zeros(3), 'start_rotation': np.array([0, 0, 0, 1]),
        'gt': {'reference_path': [[0, 0, 0], [1, 0, 1], [2, 0, 1]]},
    })
    def rgbd(red):
        rgb = np.zeros((12, 24, 3), dtype=np.uint8)
        rgb[:, :, 0] = red
        obs = {'rgb': rgb, 'depth': np.full((12, 24, 1), 2, dtype=np.float32),
               'gps': np.array([1, 2]), 'compass': np.array([0.0])}
        for angle in range(30, 360, 30):
            obs[f'rgb_{angle}'] = np.full_like(rgb, 100)
            obs[f'depth_{angle}'] = np.ones((12, 24, 1))
        return obs
    for step in (0, 1):
        obs = rgbd(50 + step)
        log.begin_step(step, obs)
        graph = nx.Graph()
        graph.add_node(7, center=np.array([10, 15]), caption='chair', score=.8)
        graph.add_node(8, center=np.array([12, 20]), caption='table')
        graph.add_edge(7, 8, distance=2.0)
        nav = nx.DiGraph()
        nav.add_node('10,10', location=np.array([10, 10]))
        nav.add_node('20,20', location=np.array([20, 20]))
        nav.add_edge('10,10', '20,20')
        mask = np.zeros((1, 12, 84), dtype=np.uint8)
        mask[:, 2:9, 3:15] = 1
        log.write(f'steps/{step:06d}',
            perception={'detections': {'boxes': np.array([[3, 2, 15, 9]]),
                'masks': mask, 'labels': np.array(['chair']), 'scores': np.array([.95])}},
            mapping={name: np.eye(32, dtype=np.uint8) for name in ('bev', 'wall', 'thin', 'fmm')},
            scene_graph=graph,
            planning={'stage': 1, 'stage_before': 1, 'stage_result': 'continue',
                'selected_point': np.array([20, 20]), 'constraints': {1: {'chair': {'type': 'near'}}},
                'waypoints': [{'points': np.array([[10, 10], [20, 20]])}],
                'navigation_tree': {'navigation_tree': nav}})
        with MotionLog(log.motion_request(step)) as writer:
            for frame in range(3 if step == 0 else 1):
                writer.record('teleport' if frame == 2 else 'action',
                    {'position': np.zeros(3)},
                    {'position': np.array([2+frame, 0, -1-frame-step]), 'heading': 0.0,
                     'rotation_xyzw': np.array([0, 0, 0, 1])},
                    action=1, teleport=frame == 2, stuck_time=0)
    # Aborted step, before perception or motion: must remain inspectable.
    log.begin_step(2, rgbd(20))
    log.write('steps/000002', error='RuntimeError: planner failed')
    log.finish('interrupted')
    return log.path


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = make_episode(self.temp.name)
        self.reader = EpisodeReader(self.path)
        self.addCleanup(self.reader.close)
        self.addCleanup(plt.close, 'all')

    def test_reader_timeline_and_typed_graph(self):
        self.assertEqual(len(self.reader.frames), 5)
        self.assertEqual(self.reader.frames[2].event, 'teleport')
        self.assertEqual(self.reader.frames[4].event, 'planning')
        self.assertEqual(len(self.reader.views), 12)
        graph = read_value(self.reader.step(0)['scene_graph'])
        self.assertEqual(list(graph.nodes), [7, 8])
        self.assertEqual(graph.edges[7, 8]['distance'], 2)
        constraints = read_value(self.reader.step(0)['planning/constraints'])
        self.assertIn(1, constraints)
        self.assertNotIn('1', constraints)
        self.assertEqual(self.reader.ground_truth().shape, (3, 3))
        # Frames share Step images; the reader caches the same arrays within a Step.
        self.assertEqual(self.reader.images(0)[0][0, 0, 0], 50)
        self.assertIs(self.reader.images(0)[0], self.reader.images(2)[0])
        self.assertNotIn('rgbd', self.reader.frame(0))
        self.assertNotIn('panorama_rgb', self.reader.step(0)['rgbd'])
        self.assertEqual(self.reader.panorama(0).shape, (12, 84, 3))

    def test_reconstructed_panorama_matches_mapping_input(self):
        # Exercise the actual mapping panorama functions without importing Habitat.
        path = Path('src/agent/panorama_utils.py')
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        namespace = {'np': np}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        obs = read_value(self.reader.step(0)['rgbd'])
        panoramas = namespace['rgbs_to_panorama'](obs)
        expected, _, _ = namespace['rotate_180'](panoramas, np.zeros((12, 3, 3)), np.zeros((12, 4, 4)))
        np.testing.assert_array_equal(self.reader.panorama(0), expected[0])

    def test_world_pose_projection_with_rotated_episode_origin(self):
        self.reader.close()
        with h5py.File(self.path, 'a') as file:
            write_value(file['episode'], 'start_position', np.array([10, 1, 20]))
            write_value(file['episode'], 'start_rotation', np.array([0, 2**-.5, 0, 2**-.5]))
            after = file['steps/000000/motion/frames/000000/after']
            write_value(after, 'position', np.array([9, 1, 18]))
            write_value(after, 'rotation_xyzw', np.array([0, 1, 0, 0]))
        reader = EpisodeReader(self.path)
        self.addCleanup(reader.close)
        gps, compass = reader.local_pose(0)
        np.testing.assert_allclose(gps, [1, 2])
        np.testing.assert_allclose(compass, [np.pi/2])

    def test_legacy_frame_rgbd_does_not_override_step_images(self):
        self.reader.close()
        with h5py.File(self.path, 'a') as file:
            file.attrs['schema_version'] = 1
            write_value(file['steps/000000/rgbd'], 'panorama_rgb', np.zeros((12, 84, 3), dtype=np.uint8))
            write_value(file['steps/000000/motion/frames/000000'], 'rgbd', {
                'rgb': np.full((12, 24, 3), 250, dtype=np.uint8),
                'gps': np.array([3, 4]), 'compass': np.array([.2])})
        reader = EpisodeReader(self.path)
        self.addCleanup(reader.close)
        self.assertEqual(reader.images(0)[0][0, 0, 0], 50)
        self.assertEqual(reader.panorama(0).shape, (12, 84, 3))
        np.testing.assert_array_equal(reader.local_pose(0)[0], [3, 4])

    def test_slider_keyboard_playback_and_sensor_switch(self):
        viewer = EpisodeViewer(self.reader)
        self.addCleanup(viewer.close)
        viewer.slider.set_val(2)
        self.assertEqual(viewer.index, 2)
        self.assertIn('teleport', viewer.heading.get_text())
        self.assertEqual(viewer.rgb_ax.images[0].get_array()[0, 0, 0], 50)
        self.assertIn('Step 000000', viewer.rgb_ax.get_title())
        x, y = viewer.map_markers[0][0].get_data()
        self.assertEqual((x[0], y[0]), (22, 8))
        viewer.cycle_view()
        self.assertEqual(viewer.rgb_ax.images[0].get_array()[0, 0, 0], 100)
        viewer.jump_step(1)
        self.assertEqual(viewer.index, 3)
        viewer.jump_step(-1)
        self.assertEqual(viewer.index, 0)
        viewer.toggle_play()
        viewer.tick()
        self.assertEqual(viewer.index, 1)
        viewer.seek(4)
        viewer.tick()
        self.assertFalse(viewer.playing)
        self.assertIn('planner failed', viewer.status.get_text())
        self.assertEqual(len(viewer.map_markers), 0)
        viewer.seek(0)
        x, y = viewer.map_markers[0][0].get_data()
        self.assertEqual((x[0], y[0]), (18, 12))

    def test_mouse_drag_and_keyboard_events(self):
        viewer = EpisodeViewer(self.reader)
        self.addCleanup(viewer.close)
        canvas = viewer.fig.canvas
        canvas.draw()
        for name, frame in [('button_press_event', 0), ('motion_notify_event', 3),
                            ('button_release_event', 3)]:
            x, y = viewer.slider.ax.transData.transform((frame, .5))
            event = MouseEvent(name, canvas, x, y, button=1)
            canvas.callbacks.process(name, event)
        self.assertEqual(viewer.index, 3)
        canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', canvas, key='left'))
        self.assertEqual(viewer.index, 2)
        canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', canvas, key='v'))
        self.assertEqual(viewer.view_index, 1)

    def test_headless_export_and_missing_modules(self):
        viewer = EpisodeViewer(self.reader, frame=4)
        self.addCleanup(viewer.close)
        target = Path(self.temp.name)/'preview.png'
        viewer.fig.savefig(target)
        self.assertGreater(target.stat().st_size, 1000)
        self.assertIn('Not recorded', [text.get_text() for text in viewer.map_axes[0].texts])

    def test_inspector_paths_and_large_array_preview(self):
        viewer = EpisodeViewer(self.reader)
        self.addCleanup(viewer.close)
        with patch('matplotlib.figure.Figure.show'):
            viewer.inspect()
        fig, box, slider = viewer.inspectors[-1]
        text = fig.axes[0].texts[0]
        self.assertIn('chair', text.get_text())
        box.set_val('/steps/000000/mapping/bev')
        self.assertIn('shape=(32, 32)', text.get_text())
        self.assertIn('Array preview', text.get_text())
        box.set_val('/episode/config')
        self.assertIn('POLICY_CONFIG', text.get_text())
        box.set_val('/missing')
        self.assertIn('Not recorded', text.get_text())

    def test_single_and_empty_episode(self):
        for with_step in (False, True):
            log = EpisodeLog.create(self.temp.name, 'empty', {'instruction': 'test'})
            if with_step:
                log.begin_step(0, {})
            reader = EpisodeReader(log.path)
            viewer = EpisodeViewer(reader)
            self.assertFalse(viewer.slider.active)
            viewer.seek(20)
            viewer.cycle_view()
            viewer.close()
            plt.close(viewer.fig)

    def test_multigraph_and_array_roundtrip(self):
        log = EpisodeLog(self.path)
        self.reader.close()
        graph = nx.MultiDiGraph()
        graph.add_edge(('tuple', 1), '1', key=3, relation='near')
        graph.add_edge(('tuple', 1), '1', key=4, relation='through')
        log.write('extra', graph=graph, none=None, strings=np.array(['椅子', '门']))
        with h5py.File(self.path) as file:
            restored = read_value(file['extra/graph'])
            self.assertEqual(set(restored.edges(keys=True)), set(graph.edges(keys=True)))
            self.assertEqual(restored.edges[('tuple', 1), '1', 4]['relation'], 'through')
            self.assertIsNone(read_value(file['extra/none']))
            self.assertEqual(read_value(file['extra/strings']).tolist(), ['椅子', '门'])


if __name__ == '__main__':
    unittest.main()
