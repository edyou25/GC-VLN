from pathlib import Path
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

from src.debug_log import EpisodeLog, MotionLog
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
        'gt': {'reference_path': [[0, 0, 0], [1, 0, 1], [2, 0, 1]]},
    })
    def rgbd(red):
        rgb = np.zeros((12, 24, 3), dtype=np.uint8)
        rgb[:, :, 0] = red
        return {'rgb': rgb, 'depth': np.full((12, 24, 1), 2, dtype=np.float32),
                'rgb_30': np.full_like(rgb, 100), 'depth_30': np.ones((12, 24, 1)),
                'gps': np.array([1, 2]), 'compass': np.array([0.0])}
    for step in (0, 1):
        obs = rgbd(50)
        log.begin_step(step, obs)
        log.write(f'steps/{step:06d}/rgbd', panorama_rgb=obs['rgb'])
        graph = nx.Graph()
        graph.add_node(7, center=np.array([10, 15]), caption='chair', score=.8)
        graph.add_node(8, center=np.array([12, 20]), caption='table')
        graph.add_edge(7, 8, distance=2.0)
        nav = nx.DiGraph()
        nav.add_node('10,10', location=np.array([10, 10]))
        nav.add_node('20,20', location=np.array([20, 20]))
        nav.add_edge('10,10', '20,20')
        mask = np.zeros((1, 12, 24), dtype=np.uint8)
        mask[:, 2:9, 3:15] = 1
        log.write(f'steps/{step:06d}',
            perception={'detections': {'boxes': np.array([[3, 2, 15, 9]]),
                'masks': mask, 'labels': np.array(['chair']), 'scores': np.array([.95])}},
            mapping={name: np.eye(32, dtype=np.uint8) for name in ('bev', 'wall', 'thin', 'fmm')},
            scene_graph=graph,
            planning={'stage': 1, 'stage_before': 1, 'stage_result': 'continue',
                'selected_point': np.array([20, 20]), 'constraints': {1: {'chair': {'type': 'near'}}},
                'masks': {1: {'valid_mask': np.eye(32, dtype=bool)}},
                'candidates': [{'points': np.array([[10, 10], [20, 20]])}],
                'navigation_tree': {'navigation_tree': nav}})
        with MotionLog(log.motion_request(step)) as writer:
            for frame in range(3 if step == 0 else 1):
                writer.record('teleport' if frame == 2 else 'action', rgbd(150+frame),
                    {'position': np.zeros(3)},
                    {'position': np.array([frame+step, 0, frame]), 'heading': 0.1},
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
        self.assertEqual(self.reader.views, ['rgb', 'rgb_30'])
        graph = read_value(self.reader.step(0)['scene_graph'])
        self.assertEqual(list(graph.nodes), [7, 8])
        self.assertEqual(graph.edges[7, 8]['distance'], 2)
        constraints = read_value(self.reader.step(0)['planning/constraints'])
        self.assertIn(1, constraints)
        self.assertNotIn('1', constraints)
        self.assertEqual(self.reader.ground_truth().shape, (3, 3))
        # A motion image differs from the planning image: never reuse overlays.
        self.assertEqual(self.reader.images(0)[0][0, 0, 0], 150)
        self.assertEqual(self.reader.step(0)['rgbd/panorama_rgb'][0, 0, 0], 50)

    def test_slider_keyboard_playback_and_sensor_switch(self):
        viewer = EpisodeViewer(self.reader)
        self.addCleanup(viewer.close)
        viewer.slider.set_val(2)
        self.assertEqual(viewer.index, 2)
        self.assertIn('teleport', viewer.heading.get_text())
        self.assertEqual(viewer.rgb_ax.images[0].get_array()[0, 0, 0], 152)
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
