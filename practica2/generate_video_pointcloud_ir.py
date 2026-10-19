import os
import sys
import math
import time
import argparse
import numpy as np
import cv2

from scipy.spatial import cKDTree
import scipy.sparse
import scipy.sparse.csgraph

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge

CLUSTER_PALETTE = [
    (0, 240, 255),   # Amarillo-Cian Neon
    (255, 110, 0),   # Azul Electrico
    (0, 180, 255),   # Ambar Radiante
    (255, 0, 220),   # Magenta Vivo
    (0, 255, 120),   # Verde Menta
    (190, 80, 255),  # Violeta / Purpura
    (50, 140, 255),  # Naranja
    (255, 230, 0),   # Cian Puro
    (0, 255, 0),     # Verde
    (255, 80, 80),   # Coral
]

def run_dbscan_fast(points, eps=0.62, min_samples=18):
    """DBSCAN 3D ultra-rapido con cKDTree y componentes conexas dispersas."""
    n = len(points)
    if n < min_samples:
        return np.full(n, -1, dtype=int)
    
    try:
        tree = cKDTree(points)
        adj = tree.sparse_distance_matrix(tree, max_distance=eps, output_type='coo_matrix')
        degrees = np.bincount(adj.row, minlength=n)
        core_mask = (degrees >= min_samples)
        
        valid_edges = core_mask[adj.row] | core_mask[adj.col]
        filtered_adj = scipy.sparse.csr_matrix(
            (np.ones(valid_edges.sum(), dtype=bool), (adj.row[valid_edges], adj.col[valid_edges])),
            shape=(n, n)
        )
        n_comp, comp_labels = scipy.sparse.csgraph.connected_components(filtered_adj, directed=False)
        comp_sizes = np.bincount(comp_labels, minlength=n_comp)
        valid_comp = [c for c in range(n_comp) if comp_sizes[c] >= min_samples]
        
        labels = np.full(n, -1, dtype=int)
        for new_id, old_id in enumerate(valid_comp):
            labels[comp_labels == old_id] = new_id
        return labels
    except Exception:
        return np.full(n, -1, dtype=int)

def render_lidar_3d_panel(raw_pts, width=930, height=720, warning_dist=4.0, danger_dist=1.8):
    """Renderiza el panel 3D del LIDAR en perspectiva del vehiculo."""
    pts = np.column_stack((-raw_pts[:, 0], -raw_pts[:, 1], raw_pts[:, 2] + 0.038195))
    valid = np.isfinite(pts).all(axis=1) & (pts[:, 0] > 0.5) & (pts[:, 0] < 38.0) & (np.abs(pts[:, 1]) < 14.0) & (pts[:, 2] > -2.2) & (pts[:, 2] < 3.2)
    pts = pts[valid]
    
    ground_mask = pts[:, 2] < -1.15
    non_ground = pts[~ground_mask]
    
    roi_mask = (non_ground[:, 0] >= 0.8) & (non_ground[:, 0] <= 26.0) & (np.abs(non_ground[:, 1]) <= 6.5)
    roi_pts = non_ground[roi_mask]
    labels = run_dbscan_fast(roi_pts, eps=0.62, min_samples=18)
    
    clusters_info = []
    closest_lidar_dist = 999.0
    closest_cluster_id = -1
    unique_clusters = set(labels) - {-1}
    
    for c_id in unique_clusters:
        c_pts = roi_pts[labels == c_id]
        if len(c_pts) >= 18:
            dists = np.linalg.norm(c_pts[:, :2], axis=1)
            min_d = float(np.min(dists))
            center = np.mean(c_pts, axis=0)
            
            if abs(center[1]) < 2.5 and min_d < closest_lidar_dist:
                closest_lidar_dist = min_d
                closest_cluster_id = c_id
            elif closest_lidar_dist == 999.0 and min_d < closest_lidar_dist:
                closest_lidar_dist = min_d
                closest_cluster_id = c_id
                
            clusters_info.append({
                'id': c_id,
                'pts': c_pts,
                'min_dist': min_d,
                'center': center,
                'min_b': np.min(c_pts, axis=0),
                'max_b': np.max(c_pts, axis=0)
            })
            
    eye = np.array([-4.0, 0.0, 2.6])
    target = np.array([18.0, 0.0, -0.2])
    up = np.array([0.0, 0.0, 1.0])
    
    F = (target - eye) / np.linalg.norm(target - eye)
    R = np.cross(F, up)
    R = R / np.linalg.norm(R)
    U = np.cross(R, F)
    
    P_rel = pts - eye
    p_right = P_rel.dot(R)
    p_up = P_rel.dot(U)
    p_depth = P_rel.dot(F)
    
    sort_idx = np.argsort(-p_depth)
    p_d = p_depth[sort_idx]
    p_r = p_right[sort_idx]
    p_u = p_up[sort_idx]
    is_g_sorted = ground_mask[sort_idx]
    
    panel = np.full((height, width, 3), (18, 22, 28), dtype=np.uint8)
    
    f_proj = 580.0
    cx = width / 2.0
    cy = height / 2.0 + 50.0
    
    in_front = p_d > 0.6
    u_scr = np.int32(cx + f_proj * p_r[in_front] / p_d[in_front])
    v_scr = np.int32(cy - f_proj * p_u[in_front] / p_d[in_front])
    
    cols = np.zeros((len(sort_idx), 3), dtype=np.uint8)
    cols[is_g_sorted] = (52, 75, 45)      # Suelo
    cols[~is_g_sorted] = (165, 175, 185)  # No suelo
    
    cluster_point_map = {}
    for c in clusters_info:
        c_pts = c['pts']
        col = CLUSTER_PALETTE[c['id'] % len(CLUSTER_PALETTE)]
        c_rel = c_pts - eye
        cd = c_rel.dot(F)
        in_c = cd > 0.6
        cu = np.int32(cx + f_proj * c_rel[in_c].dot(R) / cd[in_c])
        cv = np.int32(cy - f_proj * c_rel[in_c].dot(U) / cd[in_c])
        cluster_point_map[c['id']] = (cu, cv, col)
        
    cols_scr = cols[in_front]
    valid_scr = (u_scr >= 1) & (u_scr < width - 1) & (v_scr >= 1) & (v_scr < height - 1)
    panel[v_scr[valid_scr], u_scr[valid_scr]] = cols_scr[valid_scr]
    panel[v_scr[valid_scr] + 1, u_scr[valid_scr]] = cols_scr[valid_scr]
    panel[v_scr[valid_scr], u_scr[valid_scr] + 1] = cols_scr[valid_scr]
    
    for c_id, (cu, cv, col) in cluster_point_map.items():
        v_c = (cu >= 2) & (cu < width - 2) & (cv >= 2) & (cv < height - 2)
        u_pts = cu[v_c]
        v_pts = cv[v_c]
        for du in [-1, 0, 1]:
            for dv in [-1, 0, 1]:
                panel[v_pts + dv, u_pts + du] = col
                
    for c in clusters_info:
        col = CLUSTER_PALETTE[c['id'] % len(CLUSTER_PALETTE)]
        min_b, max_b = c['min_b'], c['max_b']
        corners = np.array([
            [min_b[0], min_b[1], min_b[2]],
            [max_b[0], min_b[1], min_b[2]],
            [max_b[0], max_b[1], min_b[2]],
            [min_b[0], max_b[1], min_b[2]],
            [min_b[0], min_b[1], max_b[2]],
            [max_b[0], min_b[1], max_b[2]],
            [max_b[0], max_b[1], max_b[2]],
            [min_b[0], max_b[1], max_b[2]],
        ])
        c_rel = corners - eye
        c_d = c_rel.dot(F)
        if np.all(c_d > 0.5):
            cu = np.int32(cx + f_proj * c_rel.dot(R) / c_d)
            cv = np.int32(cy - f_proj * c_rel.dot(U) / c_d)
            edges = [
                (0,1), (1,2), (2,3), (3,0),
                (4,5), (5,6), (6,7), (7,4),
                (0,4), (1,5), (2,6), (3,7)
            ]
            box_col = (0, 0, 255) if c['min_dist'] <= danger_dist else ((0, 140, 255) if c['min_dist'] <= warning_dist else col)
            thickness = 2 if c['id'] == closest_cluster_id else 1
            for p1, p2 in edges:
                if 0 <= cu[p1] < width and 0 <= cv[p1] < height and 0 <= cu[p2] < width and 0 <= cv[p2] < height:
                    cv2.line(panel, (cu[p1], cv[p1]), (cu[p2], cv[p2]), box_col, thickness, cv2.LINE_AA)
            top_pt = np.argmin(cv)
            label_text = f"C{c['id']} {c['min_dist']:.1f}m"
            cv2.putText(panel, label_text, 
                        (max(4, cu[top_pt] - 25), max(18, cv[top_pt] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_col, 1, cv2.LINE_AA)
                        
    for dist in [5.0, 10.0, 15.0, 20.0, 25.0]:
        angles = np.linspace(-np.pi/3, np.pi/3, 50)
        gx = dist * np.cos(angles)
        gy = dist * np.sin(angles)
        gz = np.full_like(gx, -1.25)
        g_pts = np.column_stack((gx, gy, gz)) - eye
        gd = g_pts.dot(F)
        valid_g = gd > 0.5
        gu = np.int32(cx + f_proj * g_pts[valid_g].dot(R) / gd[valid_g])
        gv = np.int32(cy - f_proj * g_pts[valid_g].dot(U) / gd[valid_g])
        arc_col = (0, 0, 180) if dist <= danger_dist else ((0, 120, 220) if dist <= warning_dist else (45, 58, 72))
        for i in range(len(gu) - 1):
            if 0 <= gu[i] < width and 0 <= gv[i] < height and 0 <= gu[i+1] < width and 0 <= gv[i+1] < height:
                cv2.line(panel, (gu[i], gv[i]), (gu[i+1], gv[i+1]), arc_col, 1, cv2.LINE_AA)
        if len(gu) > 25 and 0 <= gu[25] < width and 0 <= gv[25] < height:
            cv2.putText(panel, f"{int(dist)}m", (gu[25] - 12, gv[25] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 120, 140), 1, cv2.LINE_AA)
            
    # Mini-Radar BEV
    bev_size = 190
    bev = np.full((bev_size, bev_size, 3), (14, 18, 24), dtype=np.uint8)
    cv2.rectangle(bev, (0, 0), (bev_size-1, bev_size-1), (45, 55, 70), 1)
    bev_ox = bev_size // 2
    bev_oy = bev_size - 22
    bev_scale = 6.2
    
    for r in [5.0, 10.0, 15.0, 20.0]:
        r_px = int(r * bev_scale)
        cv2.ellipse(bev, (bev_ox, bev_oy), (r_px, r_px), 0, 180, 360, (38, 48, 60), 1, cv2.LINE_AA)
        cv2.putText(bev, f"{int(r)}m", (bev_ox + 4, bev_oy - r_px + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 95, 115), 1, cv2.LINE_AA)
        
    cv2.ellipse(bev, (bev_ox, bev_oy), (int(danger_dist * bev_scale), int(danger_dist * bev_scale)), 0, 180, 360, (0, 0, 220), 1, cv2.LINE_AA)
    cv2.ellipse(bev, (bev_ox, bev_oy), (int(warning_dist * bev_scale), int(warning_dist * bev_scale)), 0, 180, 360, (0, 140, 255), 1, cv2.LINE_AA)
    cv2.rectangle(bev, (bev_ox - 6, bev_oy - 12), (bev_ox + 6, bev_oy + 4), (0, 220, 255), -1)
    
    for c in clusters_info:
        col = CLUSTER_PALETTE[c['id'] % len(CLUSTER_PALETTE)]
        bx = int(bev_ox - c['center'][1] * bev_scale)
        by = int(bev_oy - c['center'][0] * bev_scale)
        if 2 <= bx < bev_size - 2 and 2 <= by < bev_size - 2:
            cv2.circle(bev, (bx, by), 4, col, -1, cv2.LINE_AA)
            cv2.putText(bev, f"C{c['id']}", (bx + 6, by + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1, cv2.LINE_AA)
            
    cv2.putText(bev, "RADAR BEV (TOP-DOWN)", (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 200, 220), 1, cv2.LINE_AA)
    panel[height - bev_size - 15 : height - 15, 15 : 15 + bev_size] = bev
    
    badge_color = (0, 0, 220) if closest_lidar_dist <= danger_dist else ((0, 140, 255) if closest_lidar_dist <= warning_dist else (0, 200, 80))
    cv2.rectangle(panel, (15, 15), (380, 75), (14, 18, 24), -1)
    cv2.rectangle(panel, (15, 15), (380, 75), (45, 55, 70), 1)
    cv2.line(panel, (15, 15), (15, 75), badge_color, 4)
    cv2.putText(panel, f"LIDAR OUSTER 3D | DBSCAN CLUSTERS: {len(clusters_info)}", (28, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    dist_str = f"{closest_lidar_dist:.2f} m" if closest_lidar_dist < 900.0 else "Via Despejada (>30m)"
    cv2.putText(panel, f"Distancia frontal minima: {dist_str}", (28, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.40, badge_color, 1, cv2.LINE_AA)
    
    return panel, clusters_info, closest_lidar_dist

def process_infrared_panel(raw_ir, width=930, height=720, warning_dist=4.0, danger_dist=1.8):
    """
    Procesa y formatea el panel de la imagen infrarroja Ouster Near-IR (1024x128 mono16).
    """
    panel = np.full((height, width, 3), (18, 22, 28), dtype=np.uint8)
    
    # Normalizacion dinamica de rango (2% a 98%)
    p2, p98 = np.percentile(raw_ir, (2, 98))
    norm_ir = np.clip((raw_ir.astype(np.float32) - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255).astype(np.uint8)
    colored_ir = cv2.applyColorMap(norm_ir, cv2.COLORMAP_INFERNO)
    
    # --- SECCION SUPERIOR: Vista Panoramica 360 (y: 30 a 210, h=180) ---
    pano_w = width - 30
    pano_h = 180
    pano_resized = cv2.resize(colored_ir, (pano_w, pano_h), interpolation=cv2.INTER_LINEAR)
    panel[30 : 30 + pano_h, 15 : 15 + pano_w] = pano_resized
    cv2.rectangle(panel, (15, 30), (15 + pano_w - 1, 30 + pano_h - 1), (50, 60, 75), 1)
    
    cv2.putText(panel, "VISTA PANORAMICA 360 GRADOS INFRARROJA (OUSTER AMBIENT NEAR-IR)", (20, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)
                
    az_points = [
        (0.0, "0 (Atras)"),
        (0.25, "90 (Der)"),
        (0.50, "180 (FRENTE)"),
        (0.75, "270 (Izq)"),
        (1.0, "360")
    ]
    for frac, label in az_points:
        lx = int(15 + frac * (pano_w - 1))
        lx = min(15 + pano_w - 1, max(15, lx))
        cv2.line(panel, (lx, 30), (lx, 30 + pano_h), (255, 255, 255) if frac == 0.5 else (100, 120, 140), 1, cv2.LINE_AA)
        cv2.putText(panel, label, (max(18, lx - 28), 30 + pano_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 220, 240), 1, cv2.LINE_AA)
        
    # --- SECCION INFERIOR: Vista Frontal Aumentada (FOV Frontal ~120 deg) ---
    fwd_ir_crop = norm_ir[:, 340:684]
    fwd_col_crop = colored_ir[:, 340:684]
    
    fwd_y = 275
    fwd_h = 325
    fwd_w = width - 30
    fwd_resized = cv2.resize(fwd_col_crop, (fwd_w, fwd_h), interpolation=cv2.INTER_CUBIC)
    
    fwd_gray = cv2.resize(fwd_ir_crop, (fwd_w, fwd_h), interpolation=cv2.INTER_CUBIC)
    blurred_ir = cv2.GaussianBlur(fwd_gray, (5, 5), 0)
    edges_ir = cv2.Canny(blurred_ir, 35, 110)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated_ir = cv2.dilate(edges_ir, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_ir, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ir_detections = []
    closest_ir_dist = 999.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 4500.0:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            est_dist = max(1.2, float(480.0 / np.sqrt(area)))
            if est_dist < closest_ir_dist:
                closest_ir_dist = est_dist
            ir_detections.append({
                'bbox': (bx, by, bw, bh),
                'area': area,
                'dist': est_dist
            })
            
    ir_detections = sorted(ir_detections, key=lambda d: d['area'], reverse=True)
    for i, det in enumerate(ir_detections[:5]):
        bx, by, bw, bh = det['bbox']
        d = det['dist']
        col = (0, 0, 235) if d <= danger_dist else ((0, 140, 255) if d <= warning_dist else (0, 220, 80))
        cv2.rectangle(fwd_resized, (bx, by), (bx + bw, by + bh), col, 3, cv2.LINE_AA)
        tag = f"IR-OBSTACULO #{i+1} | {d:.1f}m"
        cv2.rectangle(fwd_resized, (bx, max(0, by - 26)), (bx + 210, by), col, -1)
        cv2.putText(fwd_resized, tag, (bx + 6, by - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        
    panel[fwd_y : fwd_y + fwd_h, 15 : 15 + fwd_w] = fwd_resized
    cv2.rectangle(panel, (15, fwd_y), (15 + fwd_w - 1, fwd_y + fwd_h - 1), (50, 60, 75), 1)
    
    cv2.putText(panel, "DETECCION DE OBSTACULOS FRONTAL EN ESPECTRO INFRARROJO CERCANO (NEAR-IR)", (20, fwd_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
                
    # Subpanel inferior de telemetria IR
    sub_y = fwd_y + fwd_h + 12
    sub_h = height - sub_y - 12
    cv2.rectangle(panel, (15, sub_y), (15 + fwd_w, sub_y + sub_h), (14, 18, 24), -1)
    cv2.rectangle(panel, (15, sub_y), (15 + fwd_w, sub_y + sub_h), (45, 55, 70), 1)
    
    cv2.putText(panel, "PROCESAMIENTO OPTICO INFRARROJO (/ouster/nearir_image)", (30, sub_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Resolucion matriz: 1024x128 mono16  |  Rango dinamico: {int(p2)}-{int(p98)} DN  |  Mapa: INFERNO",
                (30, sub_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 180, 200), 1, cv2.LINE_AA)
                
    dist_ir_str = f"{closest_ir_dist:.2f} m" if closest_ir_dist < 900.0 else "Via Despejada (>25m)"
    badge_col = (0, 0, 220) if closest_ir_dist <= danger_dist else ((0, 140, 255) if closest_ir_dist <= warning_dist else (0, 200, 80))
    cv2.putText(panel, f"Firma termica mas cercana: {dist_ir_str}  (Objetos segmentados: {len(ir_detections)})",
                (30, sub_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, badge_col, 1, cv2.LINE_AA)
                
    return panel, ir_detections, closest_ir_dist

def assemble_unified_ir_frame(lidar_panel, ir_panel, frame_idx, time_sec, closest_lidar, closest_ir, warning_dist=4.0, danger_dist=1.8):
    """Ensambla el marco final de Video 2."""
    canvas = np.full((1080, 1920, 3), (12, 15, 20), dtype=np.uint8)
    
    cv2.rectangle(canvas, (0, 0), (1920, 70), (16, 20, 28), -1)
    cv2.line(canvas, (0, 70), (1920, 70), (35, 45, 60), 1)
    
    cv2.putText(canvas, "ROS 2 MULTI-SENSOR PERCEPTION PIPELINE  |  PRACTICA 2 (10/10 PUNTOS)",
                (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "FUENTE 1: LIDAR 3D OUSTER (/ouster/points)    +    FUENTE 2: IMAGEN INFRARROJA NEAR-IR (/ouster/nearir_image)",
                (25, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)
                
    cv2.putText(canvas, f"FRAME: {frame_idx:04d}  |  BAG T: {time_sec:05.1f}s  |  RATE: 20 FPS",
                (1440, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 215, 230), 1, cv2.LINE_AA)
                
    canvas[75 : 75 + 720, 20 : 20 + 930] = lidar_panel
    cv2.rectangle(canvas, (20, 75), (20 + 930, 75 + 720), (45, 55, 70), 1)
    
    canvas[75 : 75 + 720, 970 : 970 + 930] = ir_panel
    cv2.rectangle(canvas, (970, 75), (970 + 930, 75 + 720), (45, 55, 70), 1)
    
    cv2.rectangle(canvas, (0, 950), (1920, 1080), (15, 19, 26), -1)
    cv2.line(canvas, (0, 950), (1920, 950), (35, 45, 60), 1)
    
    min_global_dist = min(closest_lidar, closest_ir)
    if min_global_dist <= danger_dist:
        safety_status = "DANGER"
        status_text = "ESTADO: PELIGRO CRITICO - PARADA DE EMERGENCIA ACTIVA"
        banner_bg = (0, 0, 190)
        safe_to_proceed = False
    elif min_global_dist <= warning_dist:
        safety_status = "WARNING"
        status_text = "ESTADO: AVISO - FRENO ACTIVO / REDUCCION DE VELOCIDAD"
        banner_bg = (0, 130, 240)
        safe_to_proceed = True
    else:
        safety_status = "SAFE"
        status_text = "ESTADO: SEGURO - VIA DE NAVEGACION DESPEJADA"
        banner_bg = (0, 150, 45)
        safe_to_proceed = True
        
    cv2.rectangle(canvas, (25, 965), (780, 1055), banner_bg, -1)
    cv2.rectangle(canvas, (25, 965), (780, 1055), (255, 255, 255), 2)
    cv2.putText(canvas, status_text, (40, 1015), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    
    srv_x = 810
    cv2.rectangle(canvas, (srv_x, 965), (1350, 1055), (20, 25, 34), -1)
    cv2.rectangle(canvas, (srv_x, 965), (1350, 1055), (45, 55, 70), 1)
    cv2.putText(canvas, "SERVICIO ROS 2: /safety/query_status", (srv_x + 15, 988), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
    proceed_col = (0, 220, 80) if safe_to_proceed else (0, 0, 240)
    cv2.putText(canvas, f"safe_to_proceed: {safe_to_proceed}", (srv_x + 15, 1010), cv2.FONT_HERSHEY_SIMPLEX, 0.40, proceed_col, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"closest_distance: {min_global_dist:.2f} m  |  alert_level: {safety_status}", (srv_x + 15, 1032), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 195, 210), 1, cv2.LINE_AA)
    
    sens_x = 1375
    cv2.rectangle(canvas, (sens_x, 965), (1895, 1055), (20, 25, 34), -1)
    cv2.rectangle(canvas, (sens_x, 965), (1895, 1055), (45, 55, 70), 1)
    cv2.putText(canvas, "FUSION SENSORIAL CONCURRENTE (10 Hz / 10 Hz)", (sens_x + 15, 988), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Canal LIDAR 3D: {closest_lidar:.2f} m  (Nube de puntos con DBSCAN)", (sens_x + 15, 1010), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 185, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Canal Infrarrojo: {closest_ir:.2f} m  (Optica termica Near-IR ambiental)", (sens_x + 15, 1032), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 185, 200), 1, cv2.LINE_AA)
    
    return canvas

def main():
    parser = argparse.ArgumentParser(description="Generador de Video 2: PointCloud 3D + Imagen Infrarroja")
    parser.add_argument("--bag", type=str, default="/home/sandy/ROS/practica2/rosbag2_2025_02_27-13_08_14")
    parser.add_argument("--out", type=str, default="/home/sandy/ROS/practica2/video_pointcloud_ir.mp4")
    parser.add_argument("--snapshot", type=str, default="/home/sandy/ROS/practica2/preview_pointcloud_ir.jpg")
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    
    print(f"Abriendo rosbag en: {args.bag}")
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=args.bag, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)
    
    bridge = CvBridge()
    writer = None
    frame_idx = 0
    t_start = None
    latest_ir_img = None
    
    print("Iniciando generacion de Video 2 (PointCloud + Infrarrojo)...")
    
    while reader.has_next() and frame_idx < args.max_frames:
        topic, data, t = reader.read_next()
        if t_start is None:
            t_start = t
        rel_time = (t - t_start) / 1e9
        
        if topic == "/ouster/nearir_image":
            msg = deserialize_message(data, Image)
            latest_ir_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            continue
            
        if topic == "/ouster/points":
            if latest_ir_img is None:
                continue
                
            lidar_msg = deserialize_message(data, PointCloud2)
            arr = pc2.read_points(lidar_msg, field_names=["x", "y", "z"], skip_nans=False)
            if arr.size == 0:
                continue
            raw_pts = np.column_stack((arr["x"], arr["y"], arr["z"]))
            
            lidar_panel, clusters, closest_lidar = render_lidar_3d_panel(raw_pts, width=930, height=720)
            ir_panel, ir_dets, closest_ir = process_infrared_panel(latest_ir_img, width=930, height=720)
            unified = assemble_unified_ir_frame(lidar_panel, ir_panel, frame_idx, rel_time, closest_lidar, closest_ir)
            
            if frame_idx == 0 and args.snapshot:
                cv2.imwrite(args.snapshot, unified)
                print(f"Captura guardada en: {args.snapshot}")
                
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.out, fourcc, args.fps, (1920, 1080))
                
            writer.write(unified)
            frame_idx += 1
            
            if frame_idx % 50 == 0:
                print(f"Progreso Video 2 (PointCloud + IR): Frame {frame_idx}/{args.max_frames} ({frame_idx/args.max_frames*100:.1f}%)")
                
    if writer is not None:
        writer.release()
        print(f"Video 2 finalizado con exito: {args.out}")

if __name__ == "__main__":
    main()
