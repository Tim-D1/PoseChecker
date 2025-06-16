import os
import json
import numpy as np
import open3d as o3d
from filepath import match_pointcloud  
from icpsingle import align_lidar_scan_to_map
from scipy.spatial.transform import Rotation as R
import pandas as pd
import re


def scan_range(scan_str):
    # Extrahiere die Nummer aus dem String "Scan123" → 123
    match = re.search(r'\d+', scan_str)
    if match:
        num = int(match.group())
        if 0 <= num < 100:
            return "0-99"
        elif 100 <= num < 200:
            return "100-199"
        elif 200 <= num < 300:
            return "200-299"
        elif 300 <= num < 400:
            return "300-399"
        elif 400 <= num < 500:
            return "400-499"
        else:
            return ">=500"
    return "unknown"

def roll_to_quaternion(roll_deg):
    """
    Wandelt eine Drehung um die x-Achse (roll, in Grad) in einen Quaternion [x, y, z, w] um.
    """
    roll_deg = roll_deg % 360
    roll = np.radians(roll_deg)
    qx = np.sin(roll / 2)
    qy = 0.0
    qz = 0.0
    qw = np.cos(roll / 2)
    return [qx, qy, qz, qw]
def average_quaternions(q1, q2):
    q1 = np.array(q1)
    q2 = np.array(q2)

    # Ensure quaternions are in the same hemisphere
    if np.dot(q1, q2) < 0:
        q2 = -q2

    mean_quat = (q1 + q2) / 2.0
    mean_quat /= np.linalg.norm(mean_quat)
    return mean_quat

def quaternion_to_roll(q):
    x, y, z, w = q['x'], q['y'], q['z'], q['w']
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    return np.degrees(roll)

def load_pose(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)
    
def apply_offset(x_ref, y_ref, roll_ref_deg, x_off, y_off):
    roll_ref_rad = np.deg2rad(roll_ref_deg)
    cos_r = np.cos(roll_ref_rad)
    sin_r = np.sin(roll_ref_rad)
    # Rotation:
    x_offset_rot = x_off * cos_r - y_off * sin_r
    y_offset_rot = x_off * sin_r + y_off * cos_r
    # Addieren zur Referenzposition
    x_new = x_ref + x_offset_rot
    y_new = y_ref + y_offset_rot
    return x_new, y_new

def berechne_ecken_position(rotation_grad, dx, dy, x_m=0, y_m=0):
    """
    Berechnet die globale Position einer Viereck-Ecke mit NumPy:
    - Mittelpunkt (x_m, y_m) [Standard: (0,0)]
    - Abstand zur Ecke im lokalen System (dx, dy)
    - Drehung in Grad (rotation_grad)
    """
    theta = np.radians(rotation_grad)  # Grad zu Bogenmaß
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    x_e = x_m + dx * cos_theta - dy * sin_theta
    y_e = y_m + dx * sin_theta + dy * cos_theta
    
    return (x_e, y_e)


base = "Pointcloud"
outbase = "Ausgewertet"
reffrontdb = "Pointcloud/refscanfornegativ/Frontscan"
refreardb = "Pointcloud/refscanfornegativ/Rearscan"
loaded_map = o3d.io.read_point_cloud("karte.ply")
pose_errors = []   # List of dicts with all error metrics
total_matches = 0
total_files = 0
for pos in sorted(os.listdir(base)):
    if not pos.startswith("pos"):
        continue
    pos_path = os.path.join(base, pos)
    if not os.path.isdir(pos_path):
        continue
    print(f"=== {pos} ===")
    ref_dir = os.path.join(pos_path, "Scan800")
    ref_pose_path = os.path.join(ref_dir, "pose.json")
    ref_pose = load_pose(ref_pose_path)
    
    # Für Front und Rear separat
    #for kind in ["front", "rear"]:
    for scan in sorted(os.listdir(pos_path)):
        Fx, Fy, Fquat = .0,.0,roll_to_quaternion(0.0)
        Rx, Ry, Rquat = .0,.0,roll_to_quaternion(0.0)
        Fitt = 0.0
        quadcounter = 0.0
        x_final = 0.0
        y_final = 0.0
        quat_final = roll_to_quaternion(0.0)
        num = 0
        """output_dir = os.path.join(outbase, pos)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{kind}.txt")"""
        for kind in ["front", "rear"]:
            output_dir = os.path.join(outbase, pos)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{kind}.txt")
        #for scan in sorted(os.listdir(pos_path)):
            if not scan.startswith("Scan700"):
                #print(scan)
                continue
            """if len(scan)>6:
                #print(scan)
                continue"""
            scan_dir = os.path.join(pos_path, scan)
            print(scan)
            query_pcd = os.path.join(scan_dir, f"{kind}{pos}{scan.lower()}.ply")
            if not os.path.exists(query_pcd):
                continue
            query_pose_path = os.path.join(scan_dir, "pose.json")
            if not os.path.exists(query_pose_path):
                continue
            gt_pose = load_pose(query_pose_path)
            num += 1

            if kind == "front":
                result1 = match_pointcloud(query_pcd, reffrontdb, use_accuracy=False, visualize=False)
                #print(result)

            elif kind == "rear":
                result1 = match_pointcloud(query_pcd, refreardb, use_accuracy=False, visualize=False)

            matched_file = result1['matched_file']
            
            # Trefferkriterium
            is_hit = (matched_file is not None) and (kind+pos in matched_file)
            if is_hit:
                total_matches += 1
            else:
                print(pos, scan, kind, "kein matching")
                break
            #print("aktuelle hits:", total_matches)
        
            # Positionsabweichung aus JSON
            x_gt = gt_pose["position"]["x"] 
            y_gt = gt_pose["position"]["y"] 
            x_ref = ref_pose["position"]["x"]
            y_ref = ref_pose["position"]["y"]
            roll_ref = quaternion_to_roll(ref_pose["orientation"])
            

            scanforicp = o3d.io.read_point_cloud(query_pcd)
            scanbackup = o3d.io.read_point_cloud(query_pcd)
            if kind == "front":
                winkel = 0
                x_pos, y_pos = apply_offset(x_ref, y_ref, -roll_ref,-0.42,-.4 )
                #print(x_pos,y_pos)
                result = align_lidar_scan_to_map(loaded_map,scanforicp, iters=2000, threshold=1.5,initial_pose=(x_pos,y_pos,-roll_ref-135),vis=False)
                #print(result)
                #print(x_pos,y_pos,-roll_ref-135)
                if result["fitness"] <= 0.8:
                    scanforicp = scanbackup
                    result = align_lidar_scan_to_map(loaded_map,scanforicp, iters=20000, threshold=1.5, initial_pose=(x_pos,y_pos,-roll_ref-135),vis=False)
                    #print(result)
                if result["fitness"] >= 0.8:
                    matrix = result["final_transform"]

                    result = align_lidar_scan_to_map(loaded_map,scanforicp, threshold=0.5,iters=1000, initial_pose=(0.,0,0),vis=False)
                    matrix = result["final_transform"] @ matrix
                    result = align_lidar_scan_to_map(loaded_map,scanforicp, threshold=0.1,iters=1000, initial_pose=(0.,0,0), vis=False)
                    matrix = result["final_transform"] @ matrix
                    result = align_lidar_scan_to_map(loaded_map,scanforicp, threshold=0.01,iters=1000, initial_pose=(0.,0,0),vis=False)
                    matrix = result["final_transform"] @ matrix

                    
                    #print(x_center_new, y_center_new, winkel)
                    #print(matrix)
                    #print(Rx,Ry,Rquat)
                    winkel = np.arctan2(matrix[1,0], matrix[0,0])
                    winkeldeg = 135 + np.rad2deg(winkel)
                    Fx = matrix[0, 3]
                    Fy = matrix[1, 3]
                    #print(Fx,Fy)
                    x_center_new, y_center_new = berechne_ecken_position(winkeldeg,.42,.4,0,0)
                    print(x_center_new, y_center_new, winkeldeg)
                    Fx, Fy , Fquat = x_center_new+Fx * 1 ,y_center_new+Fy * 1, roll_to_quaternion(-winkeldeg)
                    
                    print(Fx,Fy,Fquat)
                    #print(result)
                    Fitt = result["fitness"]

                    quadcounter = 1.0
                    Fx, Fy = Fx * result["fitness"] ,Fy * result["fitness"]
                else:
                    print("icp machting bad result", result["fitness"], pos,kind,scan)
            elif kind == "rear":
                winkel = 0
                x_pos, y_pos = apply_offset(x_ref, y_ref, -roll_ref,0.42,0.4 )
                #print("rear")
                #print(x_pos,y_pos)
                result = align_lidar_scan_to_map(loaded_map,scanforicp, iters=2000, threshold=1.5, initial_pose=(x_pos,y_pos,-roll_ref+45),vis=False)
                #print(result)
                
                if result["fitness"] <= 0.90:
                    scanforicp = scanbackup
                    result = align_lidar_scan_to_map(loaded_map,scanforicp, iters=20000, threshold=1.5, initial_pose=(x_pos,y_pos,-roll_ref+45),vis=False)
                    #print(result)
                if result["fitness"] >= 0.8:
                    matrix = result["final_transform"]

                    result = align_lidar_scan_to_map(loaded_map,scanforicp, threshold=0.5,iters=1000, initial_pose=(0.,0,0),vis=False)
                    matrix = result["final_transform"] @ matrix
                    result = align_lidar_scan_to_map(loaded_map,scanforicp, threshold=0.1,iters=1000, initial_pose=(0.,0,0), vis=False)
                    matrix = result["final_transform"] @ matrix
                    result = align_lidar_scan_to_map(loaded_map,scanforicp, threshold=0.01,iters=1000, initial_pose=(0.,0,0),vis=False)
                    matrix = result["final_transform"] @ matrix

                    
                    #print(x_center_new, y_center_new, winkel)
                    #print(matrix)
                    print(Rx,Ry,Rquat)
                    winkel = np.arctan2(matrix[1,0], matrix[0,0])
                    winkeldeg = np.rad2deg(winkel)-45
                    Rx = matrix[0, 3]
                    Ry = matrix[1, 3]
                    #print(Rx,Ry)
                    x_center_new, y_center_new = berechne_ecken_position(winkeldeg,-.42,-.4,0,0)
                    print(x_center_new, y_center_new, winkeldeg)
                    Rx, Ry , Rquat = x_center_new+Rx * 1 ,y_center_new+Ry * 1, roll_to_quaternion(-winkeldeg)

                    print(Rx,Ry,Rquat)
                    #print(result)


                    quadcounter += 1.0
                    Rx, Ry = Rx * result["fitness"] ,Ry * result["fitness"]
                    Fitt = Fitt + result["fitness"]
                else:
                    print("icp machting bad result", result["fitness"], pos,kind,scan, result["fitness"])
                    
                if Fitt <= 0.3:
                    print("ICP Failed")
                else:
                    x_final = (Rx + Fx)/Fitt
                    y_final = (Ry + Fy)/Fitt
                    mean_quat = average_quaternions(Fquat,Rquat)
                    print(x_final, y_final, mean_quat)
                    x_error = np.abs(np.abs(x_final) - np.abs(x_gt))
                    y_error = np.abs(np.abs(y_final) - np.abs(y_gt))
                    pos_error = np.sqrt(x_error ** 2 + y_error ** 2)
                    gt_quat = [
                        gt_pose["orientation"]["x"],
                        gt_pose["orientation"]["y"],
                        gt_pose["orientation"]["z"],
                        gt_pose["orientation"]["w"]
                    ]
                    mean_quat_list = mean_quat.tolist()
                    rot_gt = R.from_quat(gt_quat)
                    rot_est = R.from_quat(mean_quat_list)
                    ori_error_deg = rot_gt.inv() * rot_est
                    angle_error_deg = ori_error_deg.magnitude() * 180 / np.pi
                    x_errorRef = np.abs(np.abs(x_ref) - np.abs(x_gt))
                    y_errorRef = np.abs(np.abs(y_ref) - np.abs(y_gt))
                    pos_errorRef = np.sqrt(x_errorRef ** 2 + y_errorRef ** 2)
                    # Fehlerdaten speichern
                    pose_errors.append({
                        "pos": pos,
                        "scan": scan,
                        "x_gt": x_gt,
                        "y_gt": y_gt,
                        "x_est": x_final,
                        "y_est": y_final,
                        "x_error": x_error,
                        "y_error": y_error,
                        "pos_error_m": pos_error,
                        "x_errorref": x_errorRef,
                        "y_errorref": y_errorRef,
                        "pos_errorref": pos_errorRef,
                        "ori_error_deg": angle_error_deg
                    })
                loaded_map.paint_uniform_color([0.8, 0.8, 0.8])    # Karte in Hellgrau
                aligned_scan = result['aligned_scan']
                scan_pcd_coarse = result['coarse_scan']
                scan_pcd_coarse.paint_uniform_color([1.0, 0.0, 0.0])     # ursprüngliche Scanposition in Rot
                aligned_scan.paint_uniform_color([0.0, 1.0, 0.0])    # registrierte Scanposition in Grün
                # Offene3D-Visualisierung: alle Punktwolken zusammen darstellen
                """o3d.visualization.draw_geometries(
                    [loaded_map, aligned_scan,scan_pcd_coarse],
                    window_name="Scan-Alignment", width=800, height=600
                )"""
                total_files = total_files + 1
            

os.makedirs(outbase, exist_ok=True)

df = pd.DataFrame(pose_errors)
csv_path = os.path.join(outbase, "pose_errors.csv")
df.to_csv(csv_path, index=False, sep=";", decimal=",")
print(f"Fehlerdaten als CSV gespeichert: {csv_path}")
print("Fertig ausgewertet.")
summary = {
    "total_files": total_files,
    "total_matches": total_matches,
    "match_ratio": total_matches / total_files if total_files > 0 else 0.0,
    "mean_x_error": np.mean([e["x_error"] for e in pose_errors]) if pose_errors else None,
    "mean_y_error": np.mean([e["y_error"] for e in pose_errors]) if pose_errors else None,
    "mean_pos_error_m": np.mean([e["pos_error_m"] for e in pose_errors]) if pose_errors else None,
    "mean_ori_error_deg": np.mean([e["ori_error_deg"] for e in pose_errors]) if pose_errors else None
}
summary_df = pd.DataFrame([summary])
summary_csv_path = os.path.join(outbase, "summary.csv")
summary_df.to_csv(summary_csv_path, index=False, sep=";", decimal=",")
print(f"Zusammenfassung als CSV gespeichert: {summary_csv_path}")

df['scan_range'] = df['scan'].apply(scan_range)

# Summary pro pos
summary_pos = df.groupby('pos').agg({
    "x_error": "mean",
    "y_error": "mean",
    "pos_error_m": "mean",
    "ori_error_deg": "mean",
    "x_gt": "count"
}).rename(columns={"x_gt": "num_scans"})
summary_pos.reset_index(inplace=True)
summary_pos.to_csv(os.path.join(outbase, "summary_by_pos.csv"), index=False, sep=";", decimal=",")

# Summary pro scan range
summary_scan_range = df.groupby('scan_range').agg({
    "x_error": "mean",
    "y_error": "mean",
    "pos_error_m": "mean",
    "ori_error_deg": "mean",
    "x_gt": "count"
}).rename(columns={"x_gt": "num_scans"})
summary_scan_range.reset_index(inplace=True)
summary_scan_range.to_csv(os.path.join(outbase, "summary_by_scan_range.csv"), index=False, sep=";", decimal=",")