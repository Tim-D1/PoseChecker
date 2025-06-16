import numpy as np
import open3d as o3d

def align_lidar_scan_to_map(map_pcd, scan_pcd, threshold = 1, iters = 2000, initial_pose=(0, 0, 0), vis = False):
    """
    Registriert einen Scan in eine Karten-Punktwolke.
    Arbeitet destruktiv (verändert scan_pcd).
    """
    # Sicherstellen, dass Scan in XY-Ebene liegt (Z=0)
    scan_points = np.asarray(scan_pcd.points)
    scan_points[:, 2] = 0.0
    scan_pcd.points = o3d.utility.Vector3dVector(scan_points)


    # Grobe Initial-Transformation
    x, y, yaw_deg = initial_pose
    yaw = np.deg2rad(yaw_deg)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    trans_init = np.array([
        [cos_yaw, -sin_yaw, 0.0, x],
        [sin_yaw,  cos_yaw, 0.0, y],
        [0.0,      0.0,     1.0, 0.0],
        [0.0,      0.0,     0.0, 1.0]
    ], dtype=np.float64)
    #print(trans_init)
    # 
    
    scan_pcd.transform(trans_init.copy())
    if vis:
        map_pcd.paint_uniform_color([0.8, 0.8, 0.8])      # Karte hellgrau
        scan_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        o3d.visualization.draw_geometries(
            [map_pcd, scan_pcd],
            window_name="Scan-Alignment", width=800, height=600
        )
    coarse_scan = scan_pcd  # ist nun schon transformiert
    trans = np.array([
        [1.0, -0.0, 0.0, 0.0],
        [0.0,  1.0, 0.0, 0.0],
        [0.0,  0.0, 1.0, 0.0],
        [0.0,  0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    reg_result_coarse = o3d.pipelines.registration.registration_icp(
        scan_pcd, map_pcd, threshold, trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)
    )

    T_coarse = reg_result_coarse.transformation.copy()
    theta = np.arctan2(T_coarse[1,0], T_coarse[0,0])
    theta_deg = np.rad2deg(theta)

    scan_pcd.transform(T_coarse.copy())
    aligned_scan = scan_pcd
    if vis:
        map_pcd.paint_uniform_color([0.8, 0.8, 0.8])      # Karte hellgrau
        aligned_scan.paint_uniform_color([0.0, 1.0, 0.0])
        o3d.visualization.draw_geometries(
            [map_pcd, aligned_scan],
            window_name="Scan-Alignment", width=800, height=600
        )
    total_rot_deg = np.rad2deg(theta)
    final_transform =  T_coarse @ trans_init
    dx = final_transform[0, 3]
    dy = final_transform[1, 3]

    return {
        'final_transform': final_transform,
        'coarse_transform': T_coarse,
        'fitness': reg_result_coarse.fitness,
        'rmse': reg_result_coarse.inlier_rmse,
        'aligned_scan': aligned_scan,
        'coarse_scan': coarse_scan,
        'theta_deg': theta_deg,
        'total_rot_deg': total_rot_deg,
        'dx': dx,
        'dy': dy
    }
