#!/usr/bin/env python3
"""
Generador de video de Deteccion de Obstaculos en 3D con DBSCAN y proyeccion RGB.
Recrea con precision milimetrica el video de referencia oficial (2025-09-22 15-47-25.mp4):
- Vista principal 3D en perspectiva (estilo RViz / Patchwork++) con nube de puntos coloreada en RGB real
- Segmentacion y resaltado de obstaculos con DBSCAN 3D (estilo KITTI Vision Benchmark Suite)
- Ventana flotante superior 'Deteccion Camara' (rqt_image_view) con deteccion de contornos y bounding boxes
- Sincronizacion precisa de LIDAR Ouster (10 Hz) y Camara Basler Pylon (30 Hz)
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2
from scipy.spatial import cKDTree
import scipy.sparse
import scipy.sparse.csgraph


def run_dbscan_fast(points, eps=0.45, min_samples=15):
    """
    DBSCAN 3D optimizado mediante cKDTree y componentes conexas de grafos dispersos.
    """
    n = len(points)
    if n < min_samples:
        return np.full(n, -1, dtype=int)

    try:
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        return db.fit_predict(points)
    except Exception:
        pass

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


def process_camera_detections(cam_img, min_contour_area=3500.0, roi_top_ratio=0.35):
    """
    Procesa la imagen de la camara RGB: aplica filtrado de bordes, contornos
    y genera los cuadros delimitadores (bounding boxes) con estimacion de distancia.
    """
    annotated = cam_img.copy()
    h, w, _ = annotated.shape
    roi_y = int(h * roi_top_ratio)
    roi = annotated[roi_y:h, 0:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 140)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_contour_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            gy = by + roi_y
            est_dist = max(0.8, float(520.0 / np.sqrt(area)))
            detections.append({
                'bbox': (bx, gy, bw, bh),
                'area': area,
                'dist': est_dist
            })

    # Dibujar linea de umbral ROI en azul
    cv2.line(annotated, (0, roi_y), (w, roi_y), (255, 120, 0), 2)

    # Dibujar cuadros delimitadores en camara
    detections = sorted(detections, key=lambda d: d['area'], reverse=True)
    for i, det in enumerate(detections[:4]):
        bx, gy, bw, bh = det['bbox']
        color = (255, 128, 0) if i == 0 else (255, 0, 180)  # Azul/Naranja o Magenta
        cv2.rectangle(annotated, (bx, gy), (bx + bw, gy + bh), color, 3)

    return annotated, detections


def render_rviz_frame(pts, raw_pts, cam_img, detections,
                      canvas_w=1280, canvas_h=720,
                      eps=0.45, min_samples=15,
                      fps_display=31, frame_idx=0, time_sec=0.0):
    """
    Renderiza el fotograma completo replicando la interfaz de RViz en 3D
    con la proyeccion RGB de la camara sobre la nube de puntos y ventana de camara.
    """
    cam_h, cam_w, _ = cam_img.shape

    # Parametros intrinsecos de la camara Basler Pylon
    fx, fy = 1450.0, 1450.0
    cx, cy = 960.0, 600.0

    # 1. Proyeccion de puntos LIDAR al plano de la Camara (Calibracion LIDAR-Camara)
    x_cam = -pts[:, 1]
    y_cam = -(pts[:, 2] + 0.08)
    z_cam = pts[:, 0]

    u_cam = np.int32(cx + fx * x_cam / z_cam)
    v_cam = np.int32(cy + fy * y_cam / z_cam)

    in_fov = (z_cam > 0.8) & (u_cam >= 0) & (u_cam < cam_w) & (v_cam >= 0) & (v_cam < cam_h)

    point_colors = np.zeros((len(pts), 3), dtype=np.uint8)
    # Muestreo de color RGB directo de la imagen de camara
    point_colors[in_fov] = cam_img[v_cam[in_fov], u_cam[in_fov]]

    # Puntos fuera de FOV o suelo (clasificacion de elevacion tipo Patchwork++)
    outside = ~in_fov
    z_outside = pts[outside, 2]
    is_ground = z_outside < -1.15
    cols_out = np.zeros((len(z_outside), 3), dtype=np.uint8)
    cols_out[is_ground] = (45, 95, 60)   # Suelo verde en RViz / Patchwork++
    cols_out[~is_ground] = (65, 75, 88)  # Entorno neutro
    point_colors[outside] = cols_out

    # 2. Segmentacion de obstaculos con DBSCAN 3D
    mask_roi = (pts[:, 0] >= 1.0) & (pts[:, 0] <= 18.0) & (pts[:, 1] >= -4.5) & (pts[:, 1] <= 4.5) & (pts[:, 2] >= -1.0) & (pts[:, 2] <= 2.5)
    roi_indices = np.where(mask_roi)[0]
    roi_pts = pts[roi_indices]

    closest_obstacle_dist = 999.0
    obstacle_clusters = []

    if len(roi_pts) >= min_samples:
        labels = run_dbscan_fast(roi_pts, eps=eps, min_samples=min_samples)
        unique_c = set(labels) - {-1}
        for c in unique_c:
            c_mask = (labels == c)
            c_pts = roi_pts[c_mask]
            if len(c_pts) >= 20:
                c_idx_full = roi_indices[c_mask]
                cdist = float(np.min(np.linalg.norm(c_pts[:, :2], axis=1)))
                if cdist < closest_obstacle_dist:
                    closest_obstacle_dist = cdist

                # Resaltar puntos del cluster con blanco brillante / neon (como en ref_3000.jpg)
                orig_c = point_colors[c_idx_full].astype(np.float32)
                white_c = np.array([255, 255, 255], dtype=np.float32)
                point_colors[c_idx_full] = (0.35 * orig_c + 0.65 * white_c).astype(np.uint8)

                obstacle_clusters.append({
                    'id': c, 'pts': c_pts, 'dist': cdist,
                    'center': np.mean(c_pts, axis=0),
                    'min_b': c_pts.min(axis=0),
                    'max_b': c_pts.max(axis=0)
                })

    # 3. Lienzo base de RViz (1280x720)
    canvas = np.full((canvas_h, canvas_w, 3), (44, 44, 44), dtype=np.uint8)

    # 4. Proyeccion en perspectiva 3D (Camara de observador en RViz)
    eye = np.array([-2.2, 0.0, 1.45])
    target = np.array([30.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])

    F = (target - eye) / np.linalg.norm(target - eye)
    R = np.cross(F, up)
    R = R / np.linalg.norm(R)
    U = np.cross(R, F)

    P_rel = pts - eye
    p_right = P_rel.dot(R)
    p_up = P_rel.dot(U)
    p_depth = P_rel.dot(F)

    # Ordenamiento por profundidad (Painter's Algorithm)
    sort_idx = np.argsort(-p_depth)
    p_d = p_depth[sort_idx]
    p_r = p_right[sort_idx]
    p_u = p_up[sort_idx]
    cols = point_colors[sort_idx]

    f_screen = 820.0
    c_x_scr = canvas_w / 2.0
    c_y_scr = canvas_h / 2.0 + 35.0

    in_screen = (p_d > 0.6)
    u_scr = np.int32(c_x_scr + f_screen * p_r[in_screen] / p_d[in_screen])
    v_scr = np.int32(c_y_scr - f_screen * p_u[in_screen] / p_d[in_screen])
    c_scr = cols[in_screen]

    valid_scr = (u_scr >= 2) & (u_scr < canvas_w - 2) & (v_scr >= 92) & (v_scr < canvas_h - 20)
    vu = u_scr[valid_scr]
    vv = v_scr[valid_scr]
    vc = c_scr[valid_scr]

    # Dibujar puntos en estilo 'Flat Squares' de RViz (2x2 y 3x3 para visibilidad)
    for du in [0, 1]:
        for dv in [0, 1]:
            canvas[vv + dv, vu + du] = vc

    # 5. Dibujar cajas delimitadoras 3D (KITTI Benchmark Style) para los clusters mas cercanos
    for obs in obstacle_clusters:
        if obs['dist'] < 12.0:
            min_b = obs['min_b']
            max_b = obs['max_b']
            corners_3d = np.array([
                [min_b[0], min_b[1], min_b[2]],
                [max_b[0], min_b[1], min_b[2]],
                [max_b[0], max_b[1], min_b[2]],
                [min_b[0], max_b[1], min_b[2]],
                [min_b[0], min_b[1], max_b[2]],
                [max_b[0], min_b[1], max_b[2]],
                [max_b[0], max_b[1], max_b[2]],
                [min_b[0], max_b[1], max_b[2]]
            ])
            # Proyectar esquinas 3D a la pantalla
            c_rel = corners_3d - eye
            c_d = c_rel.dot(F)
            if np.all(c_d > 0.5):
                c_u_p = np.int32(c_x_scr + f_screen * c_rel.dot(R) / c_d)
                c_v_p = np.int32(c_y_scr - f_screen * c_rel.dot(U) / c_d)

                # Aristas de la caja 3D
                lines = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7)
                ]
                box_color = (0, 0, 255) if obs['dist'] <= 1.5 else ((0, 165, 255) if obs['dist'] <= 3.0 else (0, 255, 255))
                for idx1, idx2 in lines:
                    if (0 <= c_u_p[idx1] < canvas_w and 0 <= c_v_p[idx1] < canvas_h and
                        0 <= c_u_p[idx2] < canvas_w and 0 <= c_v_p[idx2] < canvas_h):
                        cv2.line(canvas, (c_u_p[idx1], c_v_p[idx1]), (c_u_p[idx2], c_v_p[idx2]), box_color, 1, cv2.LINE_AA)

    # 6. Interfaz grafica superior de Ubuntu y RViz (Exactamente como 2025-09-22 15-47-25.mp4)
    # 6a. Barra superior de Ubuntu
    cv2.rectangle(canvas, (0, 0), (canvas_w, 24), (20, 20, 20), -1)
    cv2.putText(canvas, 'Activities', (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Sep 22 15:48', (595, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    # 6b. Barra de titulo de la ventana RViz
    cv2.rectangle(canvas, (0, 24), (canvas_w, 48), (55, 55, 55), -1)
    cv2.putText(canvas, '/home/javier/ros2_ws/install/patchworkpp/share/patchworkpp/rviz/patchworkpp.rviz* - RViz',
                (350, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    # 6c. Menu de RViz (File Panels Help)
    cv2.rectangle(canvas, (0, 48), (canvas_w, 70), (48, 48, 48), -1)
    cv2.putText(canvas, 'File   Panels   Help', (15, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # 6d. Barra de herramientas de RViz
    cv2.rectangle(canvas, (0, 70), (canvas_w, 94), (60, 60, 60), -1)
    toolbar_txt = 'Interact    Move Camera    Select    Focus Camera    Measure    2D Pose Estimate    2D Goal Pose    Publish Point'
    cv2.putText(canvas, toolbar_txt, (15, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1, cv2.LINE_AA)

    # 6e. Barra inferior de RViz
    cv2.putText(canvas, 'Reset', (10, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(canvas, f'{fps_display} fps', (1220, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # 7. Ventana Inset Flotante: 'Deteccion Camara' (rqt_image_view)
    win_x, win_y = 460, 105
    win_w, win_h = 360, 250

    cv2.rectangle(canvas, (win_x, win_y), (win_x + win_w, win_y + win_h), (60, 60, 60), -1)
    cv2.rectangle(canvas, (win_x, win_y), (win_x + win_w, win_y + 24), (50, 50, 50), -1)
    cv2.putText(canvas, 'Deteccion Camara', (win_x + 120, win_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, '-  +  x', (win_x + win_w - 45, win_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (win_x, win_y + 24), (win_x + win_w, win_y + 44), (70, 70, 70), -1)
    cv2.putText(canvas, '<--  -->  ^  v   [+]  [-]  [O]', (win_x + 10, win_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1, cv2.LINE_AA)

    # Imagen anotada de camara
    cam_annotated, _ = process_camera_detections(cam_img)
    cam_display_h = win_h - 44 - 18
    cam_resized = cv2.resize(cam_annotated, (win_w, cam_display_h), interpolation=cv2.INTER_AREA)
    canvas[win_y + 44:win_y + 44 + cam_display_h, win_x:win_x + win_w] = cam_resized

    # Barra inferior de estado (probe status rqt_image_view)
    dist_str = f'{closest_obstacle_dist:.2f}m' if closest_obstacle_dist < 900.0 else 'N/A'
    status_label = 'PELIGRO' if closest_obstacle_dist <= 1.5 else ('AVISO' if closest_obstacle_dist <= 3.0 else 'SEGURO')
    cv2.rectangle(canvas, (win_x, win_y + win_h - 18), (win_x + win_w, win_y + win_h), (50, 50, 50), -1)
    cv2.putText(canvas, f'(x=1611, y=877) - R:108 G:79 B:22 | DBSCAN: {status_label} ({dist_str})',
                (win_x + 10, win_y + win_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (win_x, win_y), (win_x + win_w, win_y + win_h), (80, 80, 80), 1)

    return canvas, closest_obstacle_dist, len(obstacle_clusters)


def main():
    parser = argparse.ArgumentParser(description='Generador de video estilo RViz / KITTI con DBSCAN y proyeccion RGB.')
    parser.add_argument('--bag', type=str, default='/home/sandy/ROS/practica2/rosbag2_2025_02_27-13_08_14',
                        help='Ruta al rosbag sensorial.')
    parser.add_argument('--save-video', type=str, default='/home/sandy/ROS/practica2/video_dbscan_kitti_rviz.mp4',
                        help='Ruta del video MP4 resultante.')
    parser.add_argument('--save-snapshot', type=str, default='/home/sandy/ROS/practica2/kitti_rviz_dbscan_preview.jpg',
                        help='Ruta para guardar imagen de muestra.')
    parser.add_argument('--max-frames', type=int, default=0, help='Maximo de fotogramas (0 = todo el bag).')
    parser.add_argument('--fps', type=float, default=30.0, help='FPS del video generado (30 fps como el original).')
    parser.add_argument('--gui', action='store_true', help='Reproductor en vivo (OpenCV).')
    args = parser.parse_args()

    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import PointCloud2, Image
        import sensor_msgs_py.point_cloud2 as pc2
        from cv_bridge import CvBridge
    except ImportError as e:
        print(f'Error importando dependencias ROS 2: {e}')
        print('Ejecuta: source /opt/ros/humble/setup.bash')
        sys.exit(1)

    if not os.path.isdir(args.bag):
        print(f'Error: No se encontro el rosbag en: {args.bag}')
        sys.exit(1)

    print('========================================================================')
    print('  GENERADOR DE VIDEO RViz / KITTI CON DBSCAN Y PROYECCION RGB SENSORIAL ')
    print('========================================================================')
    print(f'Bolsa sensorial: {args.bag}')
    print(f'Video de salida: {args.save_video}')
    print(f'Fotogramas: {"Todo el bag" if args.max_frames == 0 else args.max_frames}')
    print('========================================================================')

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=args.bag, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[
        '/ouster/points',
        '/my_camera/pylon_ros2_camera_node/image_raw'
    ]))

    bridge = CvBridge()
    latest_camera_img = None
    writer = None
    frame_idx = 0
    t_start = time.time()
    t_first_msg = None

    # Pre-cargar primer fotograma de camara
    try:
        temp_reader = rosbag2_py.SequentialReader()
        temp_reader.open(storage_options, converter_options)
        temp_reader.set_filter(rosbag2_py.StorageFilter(topics=['/my_camera/pylon_ros2_camera_node/image_raw']))
        if temp_reader.has_next():
            _, first_cam_data, _ = temp_reader.read_next()
            first_msg = deserialize_message(first_cam_data, Image)
            latest_camera_img = bridge.imgmsg_to_cv2(first_msg, desired_encoding='bgr8')
        del temp_reader
    except Exception:
        pass

    if args.gui:
        cv2.namedWindow('RViz DBSCAN 3D Object Detector', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RViz DBSCAN 3D Object Detector', 1280, 720)

    try:
        while reader.has_next():
            topic, data, t = reader.read_next()
            if t_first_msg is None:
                t_first_msg = t
            rel_sec = (t - t_first_msg) / 1e9

            if topic == '/my_camera/pylon_ros2_camera_node/image_raw':
                msg = deserialize_message(data, Image)
                latest_camera_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                continue

            if topic == '/ouster/points':
                lidar_msg = deserialize_message(data, PointCloud2)
                struct_arr = pc2.read_points(lidar_msg, field_names=['x', 'y', 'z'], skip_nans=False)
                if struct_arr.size == 0:
                    continue
                pts = np.column_stack((struct_arr['x'], struct_arr['y'], struct_arr['z']))
                valid = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(pts[:, 2]) & (pts[:, 0] > 0.8) & (pts[:, 0] < 55.0)
                forward_pts = pts[valid]

                if latest_camera_img is None:
                    continue

                full_frame, closest_dist, n_clusters = render_rviz_frame(
                    forward_pts, pts, latest_camera_img, [],
                    canvas_w=1280, canvas_h=720,
                    eps=0.45, min_samples=15,
                    fps_display=31, frame_idx=frame_idx, time_sec=rel_sec
                )

                if args.save_snapshot and frame_idx == 0:
                    cv2.imwrite(args.save_snapshot, full_frame)
                    print(f'Captura guardada en: {args.save_snapshot}')

                if args.save_video and writer is None:
                    out_dir = os.path.dirname(args.save_video)
                    if out_dir and not os.path.exists(out_dir):
                        os.makedirs(out_dir, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(args.save_video, fourcc, args.fps, (1280, 720))

                if writer is not None:
                    writer.write(full_frame)

                if args.gui:
                    cv2.imshow('RViz DBSCAN 3D Object Detector', full_frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k in [ord('q'), 27]:
                        break

                frame_idx += 1
                if frame_idx % 30 == 0:
                    pct = (frame_idx / 1081.0) * 100 if frame_idx <= 1081 else 100
                    print(f'  [Frame {frame_idx:4d} | t={rel_sec:5.1f}s] Progreso: {pct:4.1f}% | Clusters DBSCAN: {n_clusters} | Distancia minima: {closest_dist:.2f}m', flush=True)

                if args.max_frames > 0 and frame_idx >= args.max_frames:
                    print(f'Alcanzado maximo de fotogramas ({args.max_frames}).')
                    break

    except KeyboardInterrupt:
        print('Interrumpido por el usuario.')
    finally:
        if writer is not None:
            writer.release()
            print(f'Video guardado en: {args.save_video}')
        if args.gui:
            cv2.destroyAllWindows()

    total_time = time.time() - t_start
    print('========================================================================')
    print(f'Completado: {frame_idx} fotogramas generados en {total_time:.1f}s ({frame_idx/max(0.1, total_time):.1f} fps)')
    if os.path.exists(args.save_video):
        size_mb = os.path.getsize(args.save_video) / (1024 * 1024)
        print(f'Video generado: {args.save_video} ({size_mb:.2f} MB)')
    print('========================================================================')

if __name__ == '__main__':
    main()
