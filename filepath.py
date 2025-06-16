import os
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist

# ============================
# 1. Scan Context Funktionen
# ============================
def compute_scan_context(pcd, num_sectors=3600, num_rings=20, max_range=None):
    points = np.asarray(pcd.points)
    points[:, 2] = 0  # Z = 0 für alle Punkte
    pcd.points = o3d.utility.Vector3dVector(points)
    if points.size == 0:
        return np.zeros((num_rings, num_sectors))

    pts_xy = points[:, :2]
    rho = np.linalg.norm(pts_xy, axis=1)
    phi = np.mod(np.arctan2(pts_xy[:,1], pts_xy[:,0]), 2*np.pi)

    if max_range is None:
        max_range = rho.max()

    sector_width = 2 * np.pi / num_sectors
    ring_height = max_range / num_rings if max_range > 0 else float('inf')

    sc_matrix = np.zeros((num_rings, num_sectors))
    for r, angle in zip(rho, phi):
        if r > max_range:
            continue
        ring_idx = int(r / ring_height)
        if ring_idx >= num_rings:
            ring_idx = num_rings - 1
        sector_idx = int(angle / sector_width)
        if sector_idx >= num_sectors:
            sector_idx = num_sectors - 1
        sc_matrix[ring_idx, sector_idx] = max(sc_matrix[ring_idx, sector_idx], r)
    return sc_matrix

def compare_scan_context(sc1, sc2, yaw_range_deg=10):
    num_sectors = sc1.shape[1]
    max_shift = int((yaw_range_deg / 360.0) * num_sectors)
    best_score = -1.0
    best_yaw_shift = 0.0
    eps = 1e-9

    norms1 = np.linalg.norm(sc1, axis=0) + eps
    norms2 = np.linalg.norm(sc2, axis=0) + eps
    normed_sc1 = sc1 / norms1
    normed_sc2 = sc2 / norms2

    for shift in range(-max_shift, max_shift + 1):
        sc2_shifted = np.roll(normed_sc2, shift, axis=1)
        cosines = np.sum(normed_sc1 * sc2_shifted, axis=0)
        both_empty = (norms1 < 1e-8) & (norms2 < 1e-8)
        if both_empty.any():
            cosines[both_empty] = 1.0
        score = float(np.mean(cosines))
        if score > best_score:
            best_score = score
            best_yaw_shift = shift

    return best_yaw_shift, best_score


# ============================
# 3. API-Funktion
# ============================
def match_pointcloud(query_pcd_path, db_folder, use_accuracy=False, visualize=False):
    query_pcd = o3d.io.read_point_cloud(query_pcd_path)
    query_sc = compute_scan_context(query_pcd)

    best_score = -1.0
    best_match = None
    best_yaw_shift = 0.0
    best_pcd = None

    for fname in os.listdir(db_folder):
        if not fname.endswith(".ply"):
            continue
        db_path = os.path.join(db_folder, fname)
        db_pcd = o3d.io.read_point_cloud(db_path)
        db_sc = compute_scan_context(db_pcd)

        shift, score = compare_scan_context(query_sc, db_sc)
        if score > best_score:
            best_score = score
            best_yaw_shift = shift
            best_match = fname
            best_pcd = db_pcd

    num_sectors = query_sc.shape[1]
    yaw_shift_deg = best_yaw_shift * 360.0 / num_sectors

    result = {
        'matched_file': best_match,
        'scan_context_score': best_score,
        'yaw_shift_deg': yaw_shift_deg,
        'translation_x_cm': None,
        'translation_y_cm': None,
        'icp_fitness': None,
        'icp_rmse': None,
        'transform': None
    }


    return result
