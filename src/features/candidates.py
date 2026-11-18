"""
Candidate generation for PathSnap map-matching.

Finds K nearest road arcs for each GPS observation using:
1. KDTree on arc midpoints for fast initial filtering (radius search)
2. Perpendicular distance to line segments for precise ranking
3. Configurable K and radius parameters

Logs candidate miss rate when ground truth labels are available.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import warnings

try:
    from scipy.spatial import KDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    KDTree = None

from src.data.loader import TripData


@dataclass
class Candidate:
    """A candidate road arc for a GPS observation."""
    arc_id: int
    distance: float          # Perpendicular distance to arc segment (meters)
    along_track_dist: float  # Distance along arc from start (meters)
    bearing_diff: float      # Absolute bearing difference (degrees)
    midpoint_dist: float     # Distance to arc midpoint (meters) - for initial filtering
    
    def __lt__(self, other):
        return self.distance < other.distance


@dataclass
class CandidateSet:
    """Candidate arcs for a single GPS observation."""
    obs_idx: int
    gps_lon: float
    gps_lat: float
    timestamp: float
    candidates: List[Candidate]
    ground_truth_arc: Optional[int] = None
    
    @property
    def top_k(self) -> List[Candidate]:
        return sorted(self.candidates)  # Already sorted by distance
    
    @property
    def miss(self) -> bool:
        """True if ground truth arc not in candidates."""
        if self.ground_truth_arc is None:
            return False
        return self.ground_truth_arc not in [c.arc_id for c in self.candidates]


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate great-circle distance between two points in meters."""
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Earth radius in meters
    return c * r


def bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate initial bearing from point 1 to point 2 in degrees."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.arctan2(y, x)
    return np.degrees(brng) % 360


def perpendicular_distance_to_segment(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float
) -> Tuple[float, float]:
    """
    Calculate perpendicular distance from point to line segment.
    
    Returns:
        (perpendicular_distance, along_track_distance)
        Both in same units as input coordinates (degrees here, convert to meters)
    """
    # Vector from start to end of segment
    dx = x2 - x1
    dy = y2 - y1
    
    # Vector from start to point
    dpx = px - x1
    dpy = py - y1
    
    # Segment length squared
    seg_len_sq = dx*dx + dy*dy
    
    if seg_len_sq == 0:
        # Degenerate segment (point)
        dist = np.sqrt(dpx*dpx + dpy*dpy)
        return dist, 0.0
    
    # Projection parameter t
    t = max(0.0, min(1.0, (dpx*dx + dpy*dy) / seg_len_sq))
    
    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Perpendicular distance
    perp_dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    # Along-track distance
    along_dist = t * np.sqrt(seg_len_sq)
    
    return perp_dist, along_dist


def compute_arc_geometry(nodes: np.ndarray, arcs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute arc geometry for efficient candidate generation.
    
    Returns:
        midpoints: (n_arcs, 2) array of (lon, lat) for each arc midpoint
        start_points: (n_arcs, 2) array of (lon, lat) for each arc start node
        end_points: (n_arcs, 2) array of (lon, lat) for each arc end node
    """
    n_arcs = len(arcs)
    midpoints = np.zeros((n_arcs, 2), dtype=np.float64)
    start_points = np.zeros((n_arcs, 2), dtype=np.float64)
    end_points = np.zeros((n_arcs, 2), dtype=np.float64)
    
    for arc_id, (from_node, to_node) in enumerate(arcs):
        start_lon, start_lat = nodes[from_node]
        end_lon, end_lat = nodes[to_node]
        
        start_points[arc_id] = [start_lon, start_lat]
        end_points[arc_id] = [end_lon, end_lat]
        midpoints[arc_id] = [(start_lon + end_lon) / 2, (start_lat + end_lat) / 2]
    
    return midpoints, start_points, end_points


def generate_candidates_for_trip(
    trip: TripData,
    k: int = 10,
    radius_meters: float = 100.0,
    use_kdtree: bool = True,
    ground_truth_route: Optional[np.ndarray] = None,
) -> List[CandidateSet]:
    """
    Generate candidate arcs for all GPS observations in a trip.
    
    Args:
        trip: TripData object with track, nodes, arcs
        k: Number of candidates to keep per observation
        radius_meters: Search radius in meters (for midpoint KDTree)
        use_kdtree: Whether to use KDTree for initial filtering (requires scipy)
        ground_truth_route: Optional ground truth arc sequence for miss rate logging
        
    Returns:
        List of CandidateSet, one per GPS observation
    """
    if not SCIPY_AVAILABLE and use_kdtree:
        warnings.warn("scipy not available, falling back to brute force search")
        use_kdtree = False
    
    track = trip.track  # (N, 3): lon, lat, timestamp
    nodes = trip.nodes  # (M, 2): lon, lat
    arcs = trip.arcs    # (K, 2): from_node, to_node
    
    n_obs = len(track)
    n_arcs = len(arcs)
    
    if n_arcs == 0:
        warnings.warn(f"Trip {trip.trip_id} has no arcs")
        return [CandidateSet(i, track[i,0], track[i,1], track[i,2], []) for i in range(n_obs)]
    
    # Precompute arc geometry
    midpoints, start_points, end_points = compute_arc_geometry(nodes, arcs)
    
    # Build KDTree on arc midpoints for fast radius search
    if use_kdtree:
        kdtree = KDTree(midpoints)
        # Convert radius from meters to degrees (approximate at this latitude)
        # At lat ~50, 1 degree lat ~= 111km, 1 degree lon ~= 71km
        radius_deg = radius_meters / 111000.0  # Conservative (latitude)
    else:
        kdtree = None
        radius_deg = radius_meters / 111000.0
    
    # Align ground truth route to observations (approximate)
    gt_arc_per_obs = None
    if ground_truth_route is not None and len(ground_truth_route) > 0:
        # Simple alignment: map each observation to route arc by index proportion
        route_len = len(ground_truth_route)
        gt_arc_per_obs = np.zeros(n_obs, dtype=np.int64)
        for i in range(n_obs):
            # Map observation index to route index proportionally
            route_idx = min(int(i * route_len / max(1, n_obs)), route_len - 1)
            gt_arc_per_obs[i] = ground_truth_route[route_idx]
    
    candidate_sets = []
    
    for obs_idx in range(n_obs):
        gps_lon, gps_lat, timestamp = track[obs_idx]
        gt_arc = gt_arc_per_obs[obs_idx] if gt_arc_per_obs is not None else None
        
        # Step 1: Find candidate arc IDs within radius using KDTree on midpoints
        if use_kdtree:
            candidate_arc_ids = kdtree.query_ball_point([gps_lon, gps_lat], radius_deg)
            if len(candidate_arc_ids) == 0:
                # Fallback: expand radius or use nearest neighbor
                distances, indices = kdtree.query([gps_lon, gps_lat], k=min(k*2, n_arcs))
                if np.isscalar(distances):
                    candidate_arc_ids = [int(indices)]
                else:
                    candidate_arc_ids = indices.tolist()
        else:
            # Brute force: compute midpoint distance for all arcs
            midpoint_dists = np.sqrt(
                (midpoints[:, 0] - gps_lon)**2 + (midpoints[:, 1] - gps_lat)**2
            )
            candidate_arc_ids = np.argsort(midpoint_dists)[:min(k*3, n_arcs)].tolist()
        
        # Step 2: For each candidate, compute precise perpendicular distance
        candidates = []
        for arc_id in candidate_arc_ids:
            if arc_id >= n_arcs:
                continue
                
            start_lon, start_lat = start_points[arc_id]
            end_lon, end_lat = end_points[arc_id]
            
            # Perpendicular distance in degrees
            perp_dist_deg, along_dist_deg = perpendicular_distance_to_segment(
                gps_lon, gps_lat, start_lon, start_lat, end_lon, end_lat
            )
            
            # Convert to meters (approximate at this latitude)
            perp_dist_m = perp_dist_deg * 111000.0 * np.cos(np.radians(gps_lat))
            along_dist_m = along_dist_deg * 111000.0 * np.cos(np.radians(gps_lat))
            
            # Midpoint distance in meters
            mid_lon, mid_lat = midpoints[arc_id]
            mid_dist_m = haversine_distance(gps_lon, gps_lat, mid_lon, mid_lat)
            
            # Bearing difference
            arc_bearing = bearing(start_lon, start_lat, end_lon, end_lat)
            # GPS bearing from previous point (if available)
            if obs_idx > 0:
                prev_lon, prev_lat = track[obs_idx-1, 0], track[obs_idx-1, 1]
                gps_bearing = bearing(prev_lon, prev_lat, gps_lon, gps_lat)
                bearing_diff = min(
                    abs(gps_bearing - arc_bearing),
                    360 - abs(gps_bearing - arc_bearing)
                )
            else:
                bearing_diff = 180.0  # Unknown
            
            candidates.append(Candidate(
                arc_id=arc_id,
                distance=perp_dist_m,
                along_track_dist=along_dist_m,
                bearing_diff=bearing_diff,
                midpoint_dist=mid_dist_m
            ))
        
        # Sort by perpendicular distance and keep top-K
        candidates.sort()
        candidates = candidates[:k]
        
        candidate_sets.append(CandidateSet(
            obs_idx=obs_idx,
            gps_lon=gps_lon,
            gps_lat=gps_lat,
            timestamp=timestamp,
            candidates=candidates,
            ground_truth_arc=gt_arc
        ))
    
    # Log miss rate if ground truth available
    if gt_arc_per_obs is not None:
        misses = sum(1 for cs in candidate_sets if cs.miss)
        miss_rate = misses / len(candidate_sets) if candidate_sets else 0
        print(f"Trip {trip.trip_id}: Candidate miss rate = {miss_rate:.2%} ({misses}/{len(candidate_sets)})")
    
    return candidate_sets


def generate_candidates_for_partition(
    trips: List[TripData],
    k: int = 10,
    radius_meters: float = 100.0,
    use_kdtree: bool = True,
) -> dict:
    """
    Generate candidates for all trips in a partition.
    
    Returns:
        Dict mapping trip_id -> List[CandidateSet]
    """
    all_candidates = {}
    total_misses = 0
    total_obs = 0
    
    for trip in trips:
        candidates = generate_candidates_for_trip(
            trip, k=k, radius_meters=radius_meters, 
            use_kdtree=use_kdtree, ground_truth_route=trip.route
        )
        all_candidates[trip.trip_id] = candidates
        
        total_obs += len(candidates)
        total_misses += sum(1 for cs in candidates if cs.miss)
    
    if total_obs > 0:
        print(f"\nOverall miss rate: {total_misses/total_obs:.2%} ({total_misses}/{total_obs})")
    
    return all_candidates


if __name__ == "__main__":
    # Quick test with synthetic data
    if not SCIPY_AVAILABLE:
        print("scipy not available, skipping KDTree test")
    else:
        # Test perpendicular distance
        dist, along = perpendicular_distance_to_segment(0, 0, -1, 0, 1, 0)
        print(f"Distance to segment (-1,0)-(1,0) from (0,0): {dist}")
        assert abs(dist) < 1e-10
        
        dist, along = perpendicular_distance_to_segment(0, 1, -1, 0, 1, 0)
        print(f"Distance to segment (-1,0)-(1,0) from (0,1): {dist}")
        assert abs(dist - 1.0) < 1e-10
        
        print("Geometry tests passed")