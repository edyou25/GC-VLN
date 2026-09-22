"""Small CPU geometry helpers for object footprints and RS sector overlays."""
import numpy as np


def constraint_sector(parameters, legacy_angle=None):
    """Geometry of Constraint.draw_mask before connectivity/clearance filtering.

    Coordinates follow the solver: center=[row, column], angles in radians
    measured using atan2(delta_row, delta_column). None means insufficient data.
    """
    required = ('center', 'inner_radius', 'outer_radius', 'angle_start', 'angle_end')
    if not isinstance(parameters, dict) or any(key not in parameters for key in required):
        return None
    center = np.asarray(parameters['center'], dtype=float).ravel()
    if center.size != 2 or not np.isfinite(center).all():
        return None
    inner, outer = float(parameters['inner_radius']), float(parameters['outer_radius'])
    start, end = float(parameters['angle_start']), float(parameters['angle_end'])
    if not np.isfinite([inner, outer, start, end]).all() or not 0 <= inner < outer:
        return None
    def wrap(angle):
        return (angle + np.pi) % (2*np.pi) - np.pi
    # Annuli and through/weave geometry do not need an agent-relative heading.
    fixed = parameters.get('type') in ('through', 'weave')
    full = abs(wrap(start) - wrap(end)) < np.pi/18
    approximate = False
    if not fixed and not full:
        angle = parameters.get('draw_angle_agent')
        if angle is None:
            angle = legacy_angle
            approximate = True
        if angle is None:
            return None
        offset = float(angle) + float(parameters.get('angle_direction', 0))
        start, end = start + offset, end + offset
    start, end = wrap(start), wrap(end)
    full = full or abs(start-end) < np.pi/18
    span = 2*np.pi if full else (end-start) % (2*np.pi)
    return {'center': center, 'inner': inner, 'outer': outer,
            'theta1': np.degrees(start), 'theta2': np.degrees(start+span),
            'approximate': approximate}


def legacy_constraint_angle(parameters, stage, planning, dag):
    """Estimate older logs' draw angle from stage origins and the instruction DAG."""
    origins = planning.get('navigation_tree', {}).get('stage_begin', {})
    origin = origins.get(stage, origins.get(str(stage)))
    if stage == 1:
        first_angle = 0.0
    else:
        previous = origins.get(stage-1, origins.get(str(stage-1)))
        if origin is None or previous is None:
            return None
        delta = np.asarray(origin) - previous
        first_angle = np.arctan2(delta[0], delta[1])
    if dag is None or not dag.has_edge(stage, stage+1):
        return None
    if 'object' in planning.get('navigation_mode', '') and parameters.get('type') != 'near':
        if origin is None:
            return None
        delta = np.asarray(parameters['center']) - origin
        return np.arctan2(delta[0], delta[1])
    return first_angle + float(dag.edges[stage, stage+1].get('position2next', 0))
