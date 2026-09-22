from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import networkx as nx
import numpy as np

from src.bev_overlays import constraint_sector
from src.debug_log import EpisodeLog, scene_graph_snapshot, planning_snapshot
from src.debug_log_reader import EpisodeReader
from scripts.view_hdf5_log import EpisodeViewer


def sector(**changes):
    params = dict(type='direction', center=np.array([12, 18]), inner_radius=2,
                  outer_radius=8, angle_start=-np.pi/4, angle_end=np.pi/4,
                  angle_direction=0, draw_angle_agent=np.pi/2)
    params.update(changes)
    return params


class OverlayTests(unittest.TestCase):
    def test_point_cloud_projection_is_sparse_and_does_not_mutate_graph(self):
        graph = nx.Graph()
        graph.add_node(0, center=np.array([10, 10]))
        points = np.array([[0, 0, 3], [.1, .1, 9], [1, 2, 0], [np.nan, 0, 0], [-100, 0, 0]])
        snapshot = scene_graph_snapshot(graph, [{'pcd': SimpleNamespace(points=points)}], 2, 20)
        np.testing.assert_array_equal(snapshot.nodes[0]['bev_cells'], [[10, 10], [12, 14]])
        self.assertNotIn('bev_cells', graph.nodes[0])

    def test_sector_orientation_annulus_and_wrap(self):
        geometry = constraint_sector(sector())
        self.assertAlmostEqual(geometry['theta1'], 45)
        self.assertAlmostEqual(geometry['theta2'], 135)
        np.testing.assert_array_equal(geometry['center'], [12, 18])
        self.assertFalse(geometry['approximate'])
        ring = constraint_sector(sector(type='near', angle_start=-np.pi, angle_end=np.pi,
                                        draw_angle_agent=None))
        self.assertAlmostEqual(ring['theta2']-ring['theta1'], 360)
        through = constraint_sector(sector(type='through', angle_start=np.deg2rad(170),
                                           angle_end=np.deg2rad(-170)))
        self.assertAlmostEqual(through['theta2']-through['theta1'], 20)
        self.assertAlmostEqual(through['theta1'], 170)
        old = sector()
        del old['draw_angle_agent']
        self.assertIsNone(constraint_sector(old))
        self.assertTrue(constraint_sector(old, legacy_angle=0)['approximate'])
        self.assertIsNone(constraint_sector({'type': 'near'}))

    def test_overlays_colors_layers_and_step_changes(self):
        with TemporaryDirectory() as directory:
            log = EpisodeLog.create(directory, 'overlays', {'id': 'overlays',
                'instruction': {'text': 'Pass the chair, then turn toward the table.'}})
            graph = nx.Graph()
            graph.add_node(7, caption='chair', center=np.array([12, 18]),
                           bev_cells=np.array([[11, 17], [11, 18], [12, 17], [12, 18]]))
            graph.add_node(8, caption='table', center=np.array([20, 23]),
                           scope=np.array([[18, 21], [22, 25]]))
            for step in (0, 1):
                log.begin_step(step, {})
                log.write(f'steps/{step:06d}',
                    mapping={name: np.eye(40) for name in ('bev', 'wall', 'thin', 'fmm')},
                    scene_graph=graph,
                    planning={'stage_before': 1, 'stage': 1,
                              'constraints': {1: {'chair_0': {'nodes': [7], 'relation': 'left',
                                                            'constraint': [sector()]}}},
                              'navigation_constraint': sector(center=[20, 20])})
            reader = EpisodeReader(log.path)
            viewer = EpisodeViewer(reader)
            try:
                self.assertNotEqual(tuple(viewer.object_colors[7]), tuple(viewer.object_colors[8]))
                original = viewer.object_colors[7]
                self.assertIn('1 footprints, 1 legacy bounds', viewer.overlay_status.get_text())
                self.assertIn('2 sectors', viewer.overlay_status.get_text())
                wedges = [a for a in viewer.constraint_artists if isinstance(a, Wedge)]
                self.assertEqual(len(wedges), 4)  # both maps
                self.assertEqual(wedges[-1].center, (18, 12))
                viewer.layer_controls.set_active(0)
                self.assertTrue(all(not a.get_visible() for a in viewer.object_artists))
                viewer.seek(1)
                self.assertEqual(viewer.object_colors[7], original)
                self.assertTrue(all(not a.get_visible() for a in viewer.object_artists))
                viewer.layer_controls.set_active(1)
                self.assertTrue(all(not a.get_visible() for a in viewer.constraint_artists))
                target = Path(directory)/'overlays.png'
                viewer.fig.savefig(target)
                self.assertGreater(target.stat().st_size, 1000)
            finally:
                viewer.close()
                plt.close(viewer.fig)

    def test_logged_navigation_constraint_drops_generated_mask(self):
        nav = sector(mask=np.zeros((100, 100)), device='cuda')
        solver = SimpleNamespace(stage=1, constraints={}, debug_navigation_constraint=nav,
                                 debug_candidates=[], best_point=None, navigation_mode='direction',
                                 navigation_tree=SimpleNamespace())
        snapshot = planning_snapshot(solver, 1, 'continue', None)
        self.assertNotIn('mask', snapshot['navigation_constraint'])
        self.assertAlmostEqual(snapshot['navigation_constraint']['draw_angle_agent'], np.pi/2)


if __name__ == '__main__':
    unittest.main()
