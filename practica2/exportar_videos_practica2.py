#!/usr/bin/env python3
"""
=============================================================================
Exportador de Vídeos de Demostración - Práctica 2 ROS 2
=============================================================================
Genera 2 vídeos MP4 independientes de alta fidelidad:
  1. video_lidar_3d.mp4: Vista 3D LookAt orbital idéntica a RViz2 con cuadrícula,
     nube de puntos Ouster por altura, ROI 3D, detección de obstáculos y HUD.
  2. video_camara_yolo.mp4: Stream de la cámara RGB frontal con inferencia
     YOLOv8 en tiempo real, bounding boxes con código de colores según distancia
     y telemetría ADAS en el HUD.
=============================================================================
"""

import os
import sys
import time
import math
import argparse
import sqlite3
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# ROS 2 CDR Deserialization
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

DB_PATH = '/home/jabibi/ROS/practica2/rosbag2_2025_02_27-13_08_14/rosbag2_2025_02_27-13_08_14_0.db3'
YOLO_MODEL_PATH = '/home/jabibi/ROS/practica2/models/yolov8n.pt'
OUTPUT_DIR = '/home/jabibi/ROS/practica2'

def export_lidar(limit=None):
    out_path = os.path.join(OUTPUT_DIR, 'video_lidar_3d.mp4')
    print("\n" + "="*60)
    print(" [1/2] EXPORTANDO VÍDEO LIDAR 3D (RVIZ2 RECONSTRUCTION)")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM messages WHERE topic_id = 5 ORDER BY timestamp ASC")
    msg_ids = [r[0] for r in c.fetchall()]
    total_msgs = len(msg_ids)
    if limit:
        msg_ids = msg_ids[:limit]
    print(f"Total fotogramas LiDAR a procesar: {len(msg_ids)} (de {total_msgs})")
    
    # Resolucion y parametros 3D LookAt
    W, H = 1280, 720
    fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    
    eye = np.array([-7.0, 0.0, 4.5])
    target = np.array([5.0, 0.0, 0.0])
    up_world = np.array([0.0, 0.0, 1.0])
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up_world)
    right = right / np.linalg.norm(right)
    up_cam = np.cross(right, fwd)
    focal = 900.0

    # Parametros ROI y umbrales (mismos que en lidar_obstacle_detector.py)
    roi_x_min, roi_x_max = 0.2, 8.0
    roi_y_min, roi_y_max = -1.5, 1.5
    roi_z_min, roi_z_max = -1.0, 1.5
    danger_dist, warning_dist = 1.5, 3.0

    # Precomputar cuadricula del suelo (Grid de RViz)
    grid_img = np.full((H, W, 3), 30, dtype=np.uint8)
    def project(points):
        p_rel = points - eye
        x_c = p_rel.dot(right)
        y_c = p_rel.dot(up_cam)
        z_c = p_rel.dot(fwd)
        val = z_c > 0.8
        u = np.full(len(points), -1000, dtype=np.int32)
        v = np.full(len(points), -1000, dtype=np.int32)
        u[val] = np.int32(W / 2.0 + focal * (x_c[val] / z_c[val]))
        v[val] = np.int32(H / 2.0 - focal * (y_c[val] / z_c[val]))
        return u, v, val

    # Lineas del suelo
    for gx in np.arange(0, 20.1, 2.0):
        sub_pts = np.linspace(np.array([gx, -8.0, -0.7]), np.array([gx, 8.0, -0.7]), 40)
        gu, gv, gval = project(sub_pts)
        in_s = gval & (gu >= 0) & (gu < W) & (gv >= 0) & (gv < H)
        grid_img[gv[in_s], gu[in_s]] = (52, 52, 56)
    for gy in np.arange(-8.0, 8.1, 2.0):
        sub_pts = np.linspace(np.array([0.0, gy, -0.7]), np.array([20.0, gy, -0.7]), 40)
        gu, gv, gval = project(sub_pts)
        in_s = gval & (gu >= 0) & (gu < W) & (gv >= 0) & (gv < H)
        grid_img[gv[in_s], gu[in_s]] = (52, 52, 56)

    # Esquinas de la caja 3D ROI
    box_corners = np.array([
        [roi_x_min, roi_y_min, roi_z_min],
        [roi_x_max, roi_y_min, roi_z_min],
        [roi_x_max, roi_y_max, roi_z_min],
        [roi_x_min, roi_y_max, roi_z_min],
        [roi_x_min, roi_y_min, roi_z_max],
        [roi_x_max, roi_y_min, roi_z_max],
        [roi_x_max, roi_y_max, roi_z_max],
        [roi_x_min, roi_y_max, roi_z_max],
    ])
    bu, bv, bval = project(box_corners)
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]

    t_start = time.time()
    for idx, mid in enumerate(msg_ids):
        c.execute("SELECT data FROM messages WHERE id = ?", (mid,))
        raw_data = c.fetchone()[0]
        msg = deserialize_message(raw_data, PointCloud2)
        
        struct_arr = pc2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=True)
        pts = np.column_stack((struct_arr['x'], struct_arr['y'], struct_arr['z']))
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        
        # Filtrado ROI
        mask_roi = (
            (pts[:, 0] >= roi_x_min) & (pts[:, 0] <= roi_x_max) &
            (pts[:, 1] >= roi_y_min) & (pts[:, 1] <= roi_y_max) &
            (pts[:, 2] >= roi_z_min) & (pts[:, 2] <= roi_z_max)
        )
        roi_pts = pts[mask_roi]
        
        severity = 'SAFE'
        closest_pt = None
        min_dist = float('inf')
        if len(roi_pts) >= 15:
            dists = np.linalg.norm(roi_pts[:, :2], axis=1)
            min_idx = np.argmin(dists)
            closest_pt = roi_pts[min_idx]
            min_dist = float(dists[min_idx])
            if min_dist <= danger_dist:
                severity = 'DANGER'
            elif min_dist <= warning_dist:
                severity = 'WARNING'
        
        frame = grid_img.copy()
        
        # 1. Puntos 3D coloreados por altura Z
        mask_dist = (pts[:, 0] >= -2) & (pts[:, 0] <= 25) & (np.abs(pts[:, 1]) <= 15)
        pts_view = pts[mask_dist]
        u, v, val = project(pts_view)
        in_screen = val & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u_s, v_s, z_s = u[in_screen], v[in_screen], pts_view[in_screen, 2]
        
        z_norm = np.clip((z_s - (-0.7)) / 3.0, 0.0, 1.0)
        cb = (255 * np.clip(1.5 - np.abs(z_norm - 0.2) * 4, 0, 1)).astype(np.uint8)
        cg = (255 * np.clip(1.5 - np.abs(z_norm - 0.5) * 4, 0, 1)).astype(np.uint8)
        cr = (255 * np.clip(1.5 - np.abs(z_norm - 0.8) * 4, 0, 1)).astype(np.uint8)
        
        frame[v_s, u_s] = np.stack([cb, cg, cr], axis=-1)
        frame[np.clip(v_s+1, 0, H-1), u_s] = np.stack([cb, cg, cr], axis=-1)
        
        # 2. Caja 3D ROI
        box_color = (0, 0, 255) if severity == 'DANGER' else ((0, 190, 255) if severity == 'WARNING' else (0, 220, 0))
        for e1, e2 in edges:
            if bval[e1] and bval[e2]:
                cv2.line(frame, (bu[e1], bv[e1]), (bu[e2], bv[e2]), box_color, 2, cv2.LINE_AA)
                
        # 3. Puntos dentro del ROI en rojo
        if len(roi_pts) > 0:
            ru, rv, rval = project(roi_pts)
            in_r = rval & (ru >= 1) & (ru < W - 1) & (rv >= 1) & (rv < H - 1)
            for px, py in zip(ru[in_r], rv[in_r]):
                cv2.circle(frame, (px, py), 3, (0, 30, 255), -1)
                
        # 4. Marcador de impacto sobre el punto más próximo
        if closest_pt is not None:
            cu, cv, cval = project(np.array([closest_pt]))
            if cval[0] and 0 <= cu[0] < W and 0 <= cv[0] < H:
                px, py = cu[0], cv[0]
                cv2.circle(frame, (px, py), 7, (0, 0, 255), -1)
                cv2.circle(frame, (px, py), 12, (255, 255, 255), 2)
                lbl = f"LIDAR: {min_dist:.2f}m [{severity}]"
                cv2.rectangle(frame, (px + 14, py - 26), (px + 235, py + 2), (20, 20, 20), -1)
                cv2.putText(frame, lbl, (px + 18, py - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(frame, (px, py), (px + 14, py - 12), (255, 255, 255), 1)

        # 5. Barra HUD superior (estilo RViz / ADAS)
        hud_color = (0, 0, 220) if severity == 'DANGER' else ((0, 160, 255) if severity == 'WARNING' else (0, 190, 0))
        cv2.rectangle(frame, (0, 0), (W, 46), (20, 20, 24), -1)
        cv2.line(frame, (0, 46), (W, 46), hud_color, 2)
        
        dist_txt = f"{min_dist:.2f}m" if min_dist < 100 else "Libre"
        coords_txt = f"X={closest_pt[0]:.2f}, Y={closest_pt[1]:.2f}" if closest_pt is not None else "N/A"
        time_sec = idx / fps
        status_str = f"LIDAR Ouster 3D | Estado: {severity} | Dist: {dist_txt} | Pos: ({coords_txt}) | Puntos ROI: {len(roi_pts)} | T={time_sec:.1f}s"
        cv2.putText(frame, status_str, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        
        writer.write(frame)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(msg_ids):
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            rem = (len(msg_ids) - (idx + 1)) / max(1e-3, rate)
            print(f"  -> Progreso LiDAR: {idx+1}/{len(msg_ids)} fotogramas ({(idx+1)/len(msg_ids)*100:.1f}%) | {rate:.1f} FPS | Restante: {rem:.0f}s", flush=True)

    writer.release()
    conn.close()
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    duration_s = len(msg_ids) / fps
    print(f"\n[OK] Vídeo LiDAR exportado con éxito:")
    print(f"  Ruta:     {out_path}")
    print(f"  Duración: {duration_s:.1f} segundos ({len(msg_ids)} frames a {fps:.0f} FPS)")
    print(f"  Tamaño:   {size_mb:.2f} MB")
    return out_path


def export_camera(limit=None, device=0):
    out_path = os.path.join(OUTPUT_DIR, 'video_camara_yolo.mp4')
    print("\n" + "="*60)
    print(" [2/2] EXPORTANDO VÍDEO CÁMARA RGB + YOLOv8 (TRAFFIC DETECTOR)")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM messages WHERE topic_id = 14 ORDER BY timestamp ASC")
    msg_ids = [r[0] for r in c.fetchall()]
    total_msgs = len(msg_ids)
    if limit:
        msg_ids = msg_ids[:limit]
    print(f"Total fotogramas Cámara a procesar: {len(msg_ids)} (de {total_msgs})")
    
    W, H = 1280, 720
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    
    print(f"Cargando modelo YOLOv8 ({YOLO_MODEL_PATH}) en device='{device}'...")
    model = YOLO(YOLO_MODEL_PATH)
    # Warmup
    dummy = np.zeros((H, W, 3), dtype=np.uint8)
    model(dummy, device=device, verbose=False)
    
    danger_dist = 2.0
    warning_dist = 4.5
    
    t_start = time.time()
    for idx, mid in enumerate(msg_ids):
        c.execute("SELECT data FROM messages WHERE id = ?", (mid,))
        raw_data = c.fetchone()[0]
        msg = deserialize_message(raw_data, Image)
        
        # Decodificacion directa a NumPy y conversion de color
        img_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(img_bgr, (W, H))
        
        # Inferencia con YOLOv8 en GPU (clases: personas, bicis, coches, motos, buses, camiones)
        results = model(frame, classes=[0, 1, 2, 3, 5, 7], conf=0.35, device=device, verbose=False)
        
        closest_distance = float('inf')
        global_severity = 'SAFE'
        num_obstacles = 0
        
        for r in results:
            for box in r.boxes:
                num_obstacles += 1
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                area = bw * bh
                
                # Formula de estimacion de distancia por perspectiva de la camara
                est_distance = max(0.5, float(500.0 / math.sqrt(area)))
                if est_distance < closest_distance:
                    closest_distance = est_distance
                    
                if est_distance <= danger_dist:
                    severity = 'DANGER'
                    color = (0, 0, 255)       # Rojo
                elif est_distance <= warning_dist:
                    severity = 'WARNING'
                    color = (0, 165, 255)     # Naranja
                else:
                    severity = 'SAFE'
                    color = (0, 240, 255)     # Amarillo/Cyan
                    
                bx, by = int(x1), int(y1)
                bw_i, bh_i = int(bw), int(bh)
                cv2.rectangle(frame, (bx, by), (bx + bw_i, by + bh_i), color, 2)
                
                # Etiqueta con clase, confianza y distancia estimada
                label = f"{cls_name} {conf:.2f} | {est_distance:.1f}m"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                label_y = max(th + 6, by - 6)
                cv2.rectangle(frame, (bx, label_y - th - 4), (bx + tw + 6, label_y + 4), color, -1)
                cv2.putText(frame, label, (bx + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        if closest_distance <= danger_dist:
            global_severity = 'DANGER'
            hud_color = (0, 0, 220)
        elif closest_distance <= warning_dist:
            global_severity = 'WARNING'
            hud_color = (0, 165, 255)
        else:
            global_severity = 'SAFE'
            hud_color = (0, 200, 0)
            
        # Barra HUD superior (telemetria ADAS)
        cv2.rectangle(frame, (0, 0), (W, 46), (20, 20, 24), -1)
        cv2.line(frame, (0, 46), (W, 46), hud_color, 2)
        
        dist_txt = f"{closest_distance:.1f}m" if closest_distance < 100 else "Libre"
        time_sec = idx / fps
        status_str = f"Camara Frontal (YOLOv8) | Estado: {global_severity} | Vehiculos: {num_obstacles} | Dist. mas cercana: {dist_txt} | T={time_sec:.1f}s"
        cv2.putText(frame, status_str, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        
        writer.write(frame)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(msg_ids):
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            rem = (len(msg_ids) - (idx + 1)) / max(1e-3, rate)
            print(f"  -> Progreso Cámara: {idx+1}/{len(msg_ids)} fotogramas ({(idx+1)/len(msg_ids)*100:.1f}%) | {rate:.1f} FPS | Restante: {rem:.0f}s", flush=True)

    writer.release()
    conn.close()
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    duration_s = len(msg_ids) / fps
    print(f"\n[OK] Vídeo Cámara YOLO exportado con éxito:")
    print(f"  Ruta:     {out_path}")
    print(f"  Duración: {duration_s:.1f} segundos ({len(msg_ids)} frames a {fps:.0f} FPS)")
    print(f"  Tamaño:   {size_mb:.2f} MB")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Exportador de Vídeos Práctica 2")
    parser.add_argument('--limit', type=int, default=None, help="Límite opcional de frames")
    parser.add_argument('--lidar-only', action='store_true', help="Exportar solo vídeo LiDAR")
    parser.add_argument('--cam-only', action='store_true', help="Exportar solo vídeo Cámara")
    parser.add_argument('--device', default=0, help="Dispositivo para YOLO (0 para CUDA, 'cpu' para procesador)")
    args = parser.parse_args()

    print("=================================================================")
    print("   EXPORTADOR OFICIAL DE VÍDEOS - PRÁCTICA 2 (ROS 2 / RVIZ)")
    print("=================================================================")
    
    t0 = time.time()
    if not args.cam_only:
        export_lidar(limit=args.limit)
    if not args.lidar_only:
        export_camera(limit=args.limit, device=args.device)
        
    total_time = time.time() - t0
    print("\n" + "="*60)
    print(f" PROCESO COMPLETADO EN {total_time:.1f} SEGUNDOS")
    print("="*60)

if __name__ == '__main__':
    main()
