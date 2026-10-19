#!/usr/bin/env python3
"""
Script de visualizacion y generacion de video para Deteccion de Obstaculos mediante DBSCAN.
Procesa el rosbag sensorial de la Practica 2 (/ouster/points y /my_camera/...),
aplica filtrado ROI y clustering DBSCAN 3D en tiempo real (ultra-optimizado con cKDTree),
generando una vista dual dashboard (Radar BEV + Camara Frontal + Telemetria) en video MP4 o en vivo.
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2
from scipy.spatial import cKDTree

# Paleta de colores vibrantes de alto contraste para clusters (Formato BGR para OpenCV)
CLUSTER_PALETTE = [
    (255, 240, 0),    # Cyan neon
    (0, 165, 255),    # Naranja brillante
    (255, 0, 200),    # Magenta / Rosa neon
    (0, 255, 255),    # Amarillo electrico
    (50, 255, 50),    # Verde lima
    (200, 50, 255),   # Violeta
    (0, 100, 255),    # Rojo-naranja
    (255, 128, 0),    # Azul cielo
    (180, 255, 0),    # Turquesa
    (0, 255, 150),    # Verde primavera
    (255, 180, 100),  # Lavanda
    (100, 220, 255)   # Melocoton
]


def run_dbscan(points, eps=0.45, min_samples=15):
    """
    Implementacion de DBSCAN 3D ultra-rapida mediante cKDTree y componentes conexas de grafos dispersos (SciPy).
    Rendimiento C++ de ~15-20ms para miles de puntos. Retorna IDs de cluster (-1 = ruido).
    """
    n = len(points)
    if n == 0:
        return np.array([], dtype=int)

    # Intentar sklearn si esta disponible
    try:
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        return db.fit_predict(points)
    except Exception:
        pass

    try:
        import scipy.sparse
        import scipy.sparse.csgraph

        tree = cKDTree(points)
        adj = tree.sparse_distance_matrix(tree, max_distance=eps, output_type='coo_matrix')
        degrees = np.bincount(adj.row, minlength=n)
        core_mask = (degrees >= min_samples)

        # Aristas validas: al menos un nodo es punto nuclear (core point)
        valid_edges = core_mask[adj.row] | core_mask[adj.col]
        filtered_adj = scipy.sparse.csr_matrix(
            (np.ones(valid_edges.sum(), dtype=bool), (adj.row[valid_edges], adj.col[valid_edges])),
            shape=(n, n)
        )
        n_components, comp_labels = scipy.sparse.csgraph.connected_components(filtered_adj, directed=False)

        comp_has_core = np.zeros(n_components, dtype=bool)
        np.logical_or.at(comp_has_core, comp_labels, core_mask)

        # Filtrar componentes con al menos un punto core y tamano minimo
        comp_sizes = np.bincount(comp_labels, minlength=n_components)
        valid_components = [c for c in np.where(comp_has_core)[0] if comp_sizes[c] >= min_samples]

        labels = np.full(n, -1, dtype=int)
        for new_id, old_id in enumerate(valid_components):
            labels[comp_labels == old_id] = new_id

        return labels

    except Exception:
        # Fallback a BFS estandar si fallase scipy.sparse
        labels = np.full(n, -1, dtype=int)
        tree = cKDTree(points)
        neighbors_list = tree.query_ball_point(points, r=eps)
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = neighbors_list[i]
            if len(neighbors) < min_samples:
                labels[i] = -1
            else:
                labels[i] = cluster_id
                queue = [idx for idx in neighbors if not visited[idx]]
                for q_idx in queue:
                    visited[q_idx] = True

                head = 0
                while head < len(queue):
                    curr = queue[head]
                    head += 1
                    labels[curr] = cluster_id
                    curr_neighbors = neighbors_list[curr]
                    if len(curr_neighbors) >= min_samples:
                        for n_idx in curr_neighbors:
                            if not visited[n_idx]:
                                visited[n_idx] = True
                                queue.append(n_idx)
                cluster_id += 1

        return labels


def draw_bev_radar(roi_points, labels, canvas_w=960, canvas_h=1080,
                   max_x=12.0, max_y=4.5, warning_dist=3.0, danger_dist=1.5,
                   roi_x_min=0.5, roi_x_max=12.0, roi_y_min=-3.5, roi_y_max=3.5,
                   eps=0.45, min_samples=15):
    """
    Genera el panel izquierdo: Radar Bird's-Eye-View (BEV) con clusters DBSCAN,
    cajas delimitadoras 2D, etiquetas de distancia y anillos de seguridad.
    Totalmente vectorizado para renderizado en ~2ms.
    """
    img = np.full((canvas_h, canvas_w, 3), (16, 20, 26), dtype=np.uint8)

    margin_bottom = 100
    scale = min((canvas_h - margin_bottom - 120) / max_x, (canvas_w / 2 - 50) / max_y)
    origin_u = canvas_w // 2
    origin_v = canvas_h - margin_bottom

    def to_px(x, y):
        u = int(origin_u - y * scale)
        v = int(origin_v - x * scale)
        return u, v

    # 1. Anillos concentricos de distancia
    rings = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0]
    for r in rings:
        if r <= max_x:
            r_px = int(r * scale)
            cv2.ellipse(img, (origin_u, origin_v), (r_px, r_px), 0, 180, 360, (38, 48, 62), 1, cv2.LINE_AA)
            ru, rv = to_px(r, 0)
            cv2.putText(img, f'{int(r)}m', (ru + 8, rv - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (95, 115, 140), 1, cv2.LINE_AA)

    # 2. Lineas radiales guia de angulo
    for deg in [-60, -45, -30, -15, 0, 15, 30, 45, 60]:
        rad = np.radians(deg)
        rx = max_x * np.cos(rad)
        ry = max_x * np.sin(rad)
        pu, pv = to_px(rx, ry)
        line_col = (50, 65, 80) if deg == 0 else (30, 38, 48)
        cv2.line(img, (origin_u, origin_v), (pu, pv), line_col, 1, cv2.LINE_AA)

    # 3. Zonas criticas de seguridad (Peligro rojo, Aviso naranja)
    danger_px = int(danger_dist * scale)
    warning_px = int(warning_dist * scale)
    cv2.ellipse(img, (origin_u, origin_v), (danger_px, danger_px), 0, 180, 360, (0, 0, 200), 2, cv2.LINE_AA)
    cv2.ellipse(img, (origin_u, origin_v), (warning_px, warning_px), 0, 180, 360, (0, 145, 255), 2, cv2.LINE_AA)

    # 4. Rectangulo de Region de Interes (ROI)
    roi_u1, roi_v1 = to_px(roi_x_min, roi_y_max)
    roi_u2, roi_v2 = to_px(roi_x_max, roi_y_min)
    cv2.rectangle(img, (min(roi_u1, roi_u2), min(roi_v1, roi_v2)),
                  (max(roi_u1, roi_u2), max(roi_v1, roi_v2)), (60, 80, 95), 1, cv2.LINE_4)
    cv2.putText(img, 'ZONA ROI', (min(roi_u1, roi_u2) + 6, min(roi_v1, roi_v2) + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 110, 130), 1, cv2.LINE_AA)

    # 5. Vehiculo propio (Ego vehicle)
    veh_w_px = max(18, int(1.2 * scale))
    veh_l_px = max(26, int(1.8 * scale))
    cv2.rectangle(img, (origin_u - veh_w_px//2, origin_v - veh_l_px//2),
                  (origin_u + veh_w_px//2, origin_v + veh_l_px//2), (0, 210, 255), -1)
    arrow_pts = np.array([
        [origin_u, origin_v - veh_l_px//2 - 14],
        [origin_u - 7, origin_v - veh_l_px//2],
        [origin_u + 7, origin_v - veh_l_px//2]
    ], np.int32)
    cv2.fillPoly(img, [arrow_pts], (0, 255, 255), cv2.LINE_AA)
    cv2.putText(img, 'ROBOT', (origin_u - 20, origin_v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (10, 15, 20), 1, cv2.LINE_AA)

    # 6. Proyeccion vectorial de todos los puntos
    if len(roi_points) > 0:
        all_u = np.int32(origin_u - roi_points[:, 1] * scale)
        all_v = np.int32(origin_v - roi_points[:, 0] * scale)

        # 6a. Puntos de Ruido (-1)
        noise_idx = np.where(labels == -1)[0]
        if len(noise_idx) > 0:
            nu = all_u[noise_idx]
            nv = all_v[noise_idx]
            valid_noise = (nu >= 0) & (nu < canvas_w) & (nv >= 0) & (nv < canvas_h)
            img[nv[valid_noise], nu[valid_noise]] = (55, 65, 75)

    # 7. Renderizado de Clusters
    unique_labels = sorted(list(set(labels) - {-1}))
    cluster_summaries = []

    for c_id in unique_labels:
        c_idx = np.where(labels == c_id)[0]
        c_pts = roi_points[c_idx]
        color = CLUSTER_PALETTE[c_id % len(CLUSTER_PALETTE)]

        cu_pts = all_u[c_idx]
        cv_pts = all_v[c_idx]
        valid_c = (cu_pts >= 1) & (cu_pts < canvas_w - 1) & (cv_pts >= 1) & (cv_pts < canvas_h - 1)

        # Splat vectorial 3x3 para puntos mas brillantes y visibles
        vu = cu_pts[valid_c]
        vv = cv_pts[valid_c]
        for du in [-1, 0, 1]:
            for dv in [-1, 0, 1]:
                img[vv + dv, vu + du] = color

        # Dimensiones y Bounding Box
        min_x, max_x_val = c_pts[:, 0].min(), c_pts[:, 0].max()
        min_y, max_y_val = c_pts[:, 1].min(), c_pts[:, 1].max()
        min_z, max_z_val = c_pts[:, 2].min(), c_pts[:, 2].max()
        dim_x = max_x_val - min_x
        dim_y = max_y_val - min_y
        dim_z = max_z_val - min_z

        u1, v1 = to_px(min_x, max_y_val)
        u2, v2 = to_px(max_x_val, min_y)
        bb_u_min = max(0, min(u1, u2) - 4)
        bb_u_max = min(canvas_w - 1, max(u1, u2) + 4)
        bb_v_min = max(0, min(v1, v2) - 4)
        bb_v_max = min(canvas_h - 1, max(v1, v2) + 4)

        # Relleno semi-transparente del cluster
        sub = img[bb_v_min:bb_v_max, bb_u_min:bb_u_max]
        if sub.shape[0] > 0 and sub.shape[1] > 0:
            box_overlay = np.full_like(sub, color, dtype=np.uint8)
            cv2.addWeighted(sub, 0.82, box_overlay, 0.18, 0, sub)
        cv2.rectangle(img, (bb_u_min, bb_v_min), (bb_u_max, bb_v_max), color, 2, cv2.LINE_AA)

        # Centroide
        cx, cy, cz = np.mean(c_pts, axis=0)
        dist = np.linalg.norm([cx, cy])
        cu, cv = to_px(cx, cy)
        cv2.drawMarker(img, (cu, cv), color, cv2.MARKER_CROSS, 11, 2, cv2.LINE_AA)

        # Severidad
        if dist <= danger_dist:
            sev = 'PELIGRO'
        elif dist <= warning_dist:
            sev = 'AVISO'
        else:
            sev = 'SEGURO'

        # Etiqueta visual flotante
        tag = f'C{c_id}: {dist:.1f}m ({len(c_pts)}p)'
        tag_w = len(tag) * 9 + 8
        tag_h = 18
        lbl_y = max(tag_h + 2, bb_v_min - 4)
        cv2.rectangle(img, (bb_u_min, lbl_y - tag_h), (bb_u_min + tag_w, lbl_y), (16, 20, 26), -1)
        cv2.rectangle(img, (bb_u_min, lbl_y - tag_h), (bb_u_min + tag_w, lbl_y), color, 1)
        cv2.putText(img, tag, (bb_u_min + 4, lbl_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

        cluster_summaries.append({
            'id': c_id,
            'dist': dist,
            'pts': len(c_pts),
            'centroid': (cx, cy, cz),
            'dims': (dim_x, dim_y, dim_z),
            'sev': sev,
            'color': color
        })

    # Header del panel izquierdo
    cv2.rectangle(img, (0, 0), (canvas_w, 90), (22, 28, 38), -1)
    cv2.line(img, (0, 90), (canvas_w, 90), (45, 55, 75), 2)
    cv2.putText(img, 'LIDAR BIRD-EYE VIEW (BEV)', (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f'Clustering DBSCAN (eps={eps}m, min_samples={min_samples}) | Clusters: {len(unique_labels)}',
                (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 1, cv2.LINE_AA)

    # Leyenda al pie
    cv2.putText(img, 'Peligro: <=1.5m  |  Aviso: <=3.0m  |  Puntos grises: Ruido DBSCAN',
                (25, canvas_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 150, 175), 1, cv2.LINE_AA)

    return img, cluster_summaries


def draw_right_dashboard(cam_img, cluster_summaries, total_raw_pts, roi_pts_cnt,
                         canvas_w=960, canvas_h=1080, danger_dist=1.5, warning_dist=3.0,
                         fps_val=0.0, frame_idx=0, timestamp_sec=0.0):
    """
    Genera el panel derecho: Video de Camara Frontal sincronizada + Panel de Telemetria
    y lista de clusters con distancias y niveles de severidad.
    """
    panel = np.full((canvas_h, canvas_w, 3), (20, 24, 32), dtype=np.uint8)

    # 1. Camara frontal superior (960x600)
    cam_h = 600
    if cam_img is not None:
        cam_resized = cv2.resize(cam_img, (canvas_w, cam_h), interpolation=cv2.INTER_AREA)
        panel[0:cam_h, 0:canvas_w] = cam_resized
    else:
        cv2.putText(panel, 'CAMARA SIN SENAL', (canvas_w//2 - 120, cam_h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 120, 140), 2, cv2.LINE_AA)

    # Barra de titulo sobre el video de la camara
    overlay_top = panel[0:45, 0:canvas_w].copy()
    cv2.rectangle(overlay_top, (0, 0), (canvas_w, 45), (10, 15, 22), -1)
    cv2.addWeighted(overlay_top, 0.75, panel[0:45, 0:canvas_w], 0.25, 0, panel[0:45, 0:canvas_w])
    cv2.putText(panel, 'CAMARA FRONTAL (Basler Pylon RGB)', (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f'Frame: {frame_idx:04d} | t={timestamp_sec:5.2f}s | {fps_val:4.1f} FPS',
                (canvas_w - 360, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)

    # Linea divisoria camara / telemetria
    cv2.line(panel, (0, cam_h), (canvas_w, cam_h), (50, 65, 85), 3)

    # 2. Panel inferior de telemetria (600 a 1080)
    ty = cam_h + 35
    cv2.putText(panel, 'TELEMETRIA Y FUSION SENSORIAL', (25, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    ty += 35

    # Evaluacion de Estado Global
    min_dist = min([s['dist'] for s in cluster_summaries]) if cluster_summaries else 999.0
    if min_dist <= danger_dist:
        status_text = 'ESTADO GLOBAL: PELIGRO CRITICO (DANGER)'
        status_bg = (0, 0, 200)       # Rojo
        status_fg = (255, 255, 255)
    elif min_dist <= warning_dist:
        status_text = 'ESTADO GLOBAL: PRECAUCION / AVISO (WARNING)'
        status_bg = (0, 140, 255)     # Naranja
        status_fg = (0, 0, 0)
    else:
        status_text = 'ESTADO GLOBAL: VIA DESPEJADA (SAFE)'
        status_bg = (0, 180, 50)      # Verde
        status_fg = (255, 255, 255)

    cv2.rectangle(panel, (25, ty), (canvas_w - 25, ty + 46), status_bg, -1)
    cv2.rectangle(panel, (25, ty), (canvas_w - 25, ty + 46), (255, 255, 255), 1)
    dist_label = f'{min_dist:.2f}m' if min_dist < 900.0 else 'N/A'
    cv2.putText(panel, f'{status_text} | Obstaculo mas proximo: {dist_label}',
                (40, ty + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.56, status_fg, 2, cv2.LINE_AA)
    ty += 65

    # Metricas en tarjetas
    card_w = (canvas_w - 50 - 30) // 4
    cards = [
        ('PUNTOS LIDAR', f'{total_raw_pts:,}'),
        ('PUNTOS EN ROI', f'{roi_pts_cnt:,}'),
        ('CLUSTERS DBSCAN', f'{len(cluster_summaries)}'),
        ('DISTANCIA MIN', f'{dist_label}')
    ]
    for i, (title, val) in enumerate(cards):
        cx = 25 + i * (card_w + 10)
        cv2.rectangle(panel, (cx, ty), (cx + card_w, ty + 50), (30, 36, 48), -1)
        cv2.rectangle(panel, (cx, ty), (cx + card_w, ty + 50), (55, 70, 90), 1)
        cv2.putText(panel, title, (cx + 10, ty + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 175, 200), 1, cv2.LINE_AA)
        cv2.putText(panel, val, (cx + 10, ty + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 240, 255), 2, cv2.LINE_AA)
    ty += 70

    # Tabla de Clusters Detectados
    cv2.putText(panel, 'DETALLE DE CLUSTERS DETECTADOS (DBSCAN)', (25, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 240), 1, cv2.LINE_AA)
    ty += 22

    # Encabezado de la tabla
    cv2.rectangle(panel, (25, ty), (canvas_w - 25, ty + 24), (32, 40, 52), -1)
    header = 'ID      DISTANCIA      POSICION (X, Y, Z)        TAMANO (LxWxH)      PUNTOS     ESTADO'
    cv2.putText(panel, header, (35, ty + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 200, 225), 1, cv2.LINE_AA)
    ty += 30

    sorted_clusters = sorted(cluster_summaries, key=lambda s: s['dist'])
    for s in sorted_clusters[:6]:  # Mostrar los 6 mas cercanos
        cx, cy, cz = s['centroid']
        dx, dy, dz = s['dims']
        row = f'C{s["id"]:02d}     {s["dist"]:5.2f}m        ({cx:4.1f}, {cy:4.1f}, {cz:4.1f})m      {dx:3.1f}x{dy:3.1f}x{dz:3.1f}m     {s["pts"]:5d}p     {s["sev"]}'
        cv2.rectangle(panel, (25, ty - 12), (30, ty + 6), s['color'], -1)
        cv2.putText(panel, row, (38, ty + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, s['color'], 1, cv2.LINE_AA)
        ty += 22

    if not sorted_clusters:
        cv2.putText(panel, 'No se han detectado clusters en la ROI actual.', (40, ty + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 140, 160), 1, cv2.LINE_AA)

    return panel


def main():
    parser = argparse.ArgumentParser(description='Visualizador y Generador de Video de Clusters con DBSCAN.')
    parser.add_argument('--bag', type=str, default='/home/sandy/ROS/practica2/rosbag2_2025_02_27-13_08_14',
                        help='Ruta a la carpeta del rosbag.')
    parser.add_argument('--save-video', type=str, default='/home/sandy/ROS/practica2/video_dbscan_clusters.mp4',
                        help='Ruta del video MP4 de salida.')
    parser.add_argument('--save-snapshot', type=str, default='/home/sandy/ROS/practica2/clusters_dbscan_preview.jpg',
                        help='Ruta para guardar una captura de imagen del primer fotograma analizado.')
    parser.add_argument('--gui', action='store_true', help='Mostrar reproductor interactivo en pantalla (OpenCV).')
    parser.add_argument('--max-frames', type=int, default=0, help='Maximo numero de frames (0 = todo el bag).')
    parser.add_argument('--eps', type=float, default=0.45, help='Parametro eps de DBSCAN en metros.')
    parser.add_argument('--min-samples', type=int, default=15, help='Puntos minimos para formar un cluster en DBSCAN.')
    parser.add_argument('--fps', type=float, default=20.0, help='FPS del video generado.')
    args = parser.parse_args()

    # Importacion de librerias de ROS 2
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import PointCloud2, Image
        import sensor_msgs_py.point_cloud2 as pc2
        from cv_bridge import CvBridge
    except ImportError as e:
        print(f'Error importando dependencias de ROS 2: {e}')
        print('Asegurate de ejecutar: source /opt/ros/humble/setup.bash')
        sys.exit(1)

    if not os.path.isdir(args.bag):
        print(f'Error: No se encontro el rosbag en: {args.bag}')
        sys.exit(1)

    print('=================================================================')
    print('   GENERADOR DE VIDEO Y VISUALIZADOR DE CLUSTERS DBSCAN (LIDAR)  ')
    print('=================================================================')
    print(f'Bolsa sensorial: {args.bag}')
    print(f'Video de salida: {args.save_video}')
    print(f'Parametros DBSCAN: eps={args.eps}m, min_samples={args.min_samples}')
    print(f'Modo interactivo (GUI): {args.gui}')
    print('=================================================================')

    # Apertura de rosbag
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=args.bag, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    # Filtrar solo los topicos de LIDAR y Camara
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

    # Pre-cargar el primer fotograma de camara para que el frame 0 tenga imagen real
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
        cv2.namedWindow('DBSCAN LiDAR Cluster Detector', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('DBSCAN LiDAR Cluster Detector', 1600, 900)

    try:
        while reader.has_next():
            topic, data, t = reader.read_next()
            if t_first_msg is None:
                t_first_msg = t
            rel_sec = (t - t_first_msg) / 1e9

            # Si es imagen de camara, actualizar buffer del fotograma mas reciente
            if topic == '/my_camera/pylon_ros2_camera_node/image_raw':
                msg = deserialize_message(data, Image)
                latest_camera_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                continue

            # Si es PointCloud2 de LIDAR, procesar frame completo
            if topic == '/ouster/points':
                lidar_msg = deserialize_message(data, PointCloud2)
                struct_arr = pc2.read_points(lidar_msg, field_names=['x', 'y', 'z'], skip_nans=False)
                if struct_arr.size == 0:
                    continue
                pts = np.column_stack((struct_arr['x'], struct_arr['y'], struct_arr['z']))
                total_pts = len(pts)

                # Mascara finita y ROI de navegacion
                valid = np.isfinite(pts).all(axis=1)
                pts_finite = pts[valid]

                # Filtrado de suelo (z >= -1.0) y limites laterales y frontales
                mask_roi = (
                    (pts_finite[:, 0] >= 0.5) & (pts_finite[:, 0] <= 12.0) &
                    (pts_finite[:, 1] >= -3.5) & (pts_finite[:, 1] <= 3.5) &
                    (pts_finite[:, 2] >= -1.0) & (pts_finite[:, 2] <= 1.6)
                )
                roi_pts = pts_finite[mask_roi]

                # Clustering con DBSCAN
                t_db_start = time.time()
                labels = run_dbscan(roi_pts, eps=args.eps, min_samples=args.min_samples)
                t_db_elapsed = time.time() - t_db_start

                # Dibujar panel izquierdo (Radar BEV)
                bev_img, summaries = draw_bev_radar(
                    roi_pts, labels, canvas_w=960, canvas_h=1080,
                    max_x=12.0, max_y=4.0, warning_dist=3.0, danger_dist=1.5,
                    eps=args.eps, min_samples=args.min_samples
                )

                # Calcular FPS instantaneo
                fps_val = 1.0 / max(0.001, t_db_elapsed)

                # Dibujar panel derecho (Camara + Telemetria)
                right_panel = draw_right_dashboard(
                    latest_camera_img, summaries, total_pts, len(roi_pts),
                    canvas_w=960, canvas_h=1080, danger_dist=1.5, warning_dist=3.0,
                    fps_val=fps_val, frame_idx=frame_idx, timestamp_sec=rel_sec
                )

                # Composicion en pantalla completa 1920x1080
                full_frame = np.hstack((bev_img, right_panel))

                # Guardar snapshot del primer frame con detecciones
                if args.save_snapshot and frame_idx == 0:
                    cv2.imwrite(args.save_snapshot, full_frame)
                    print(f'Captura del fotograma inicial guardada en: {args.save_snapshot}')

                # Inicializar VideoWriter en el primer frame si esta habilitado
                if args.save_video and writer is None:
                    out_dir = os.path.dirname(args.save_video)
                    if out_dir and not os.path.exists(out_dir):
                        os.makedirs(out_dir, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(args.save_video, fourcc, args.fps, (1920, 1080))
                    if not writer.isOpened():
                        print(f'Aviso: no se pudo abrir con mp4v, probando XVID para {args.save_video}.avi')
                        args.save_video = args.save_video.replace('.mp4', '.avi')
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        writer = cv2.VideoWriter(args.save_video, fourcc, args.fps, (1920, 1080))

                if writer is not None:
                    writer.write(full_frame)

                # Modo interactivo
                if args.gui:
                    cv2.imshow('DBSCAN LiDAR Cluster Detector', full_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('q'), 27]: # Q o Esc
                        print('Reproduccion detenida por el usuario.')
                        break
                    elif key == ord(' '):  # Pausa
                        print('Pausado. Presiona Espacio para continuar o N para avanzar 1 frame.')
                        while True:
                            k2 = cv2.waitKey(0) & 0xFF
                            if k2 in [ord(' '), ord('n'), ord('q'), 27]:
                                break
                        if k2 in [ord('q'), 27]:
                            break

                frame_idx += 1
                if frame_idx % 25 == 0:
                    pct = (frame_idx / 1081.0) * 100 if frame_idx <= 1081 else 100
                    print(f'  [Frame {frame_idx:4d} | t={rel_sec:5.1f}s] Progreso: {pct:4.1f}% | Clusters: {len(summaries)} | Dist min: {min([s["dist"] for s in summaries], default=999.0):.2f}m', flush=True)

                if args.max_frames > 0 and frame_idx >= args.max_frames:
                    print(f'Alcanzado el maximo de fotogramas solicitados ({args.max_frames}).')
                    break

    except KeyboardInterrupt:
        print('Proceso interrumpido con Ctrl+C.')
    finally:
        if writer is not None:
            writer.release()
            print(f'Video guardado exitosamente en: {args.save_video}')
        if args.gui:
            cv2.destroyAllWindows()

    total_time = time.time() - t_start
    print('=================================================================')
    print(f'Proceso completado: {frame_idx} fotogramas generados en {total_time:.1f}s (Media: {frame_idx/max(0.1, total_time):.1f} fps)')
    if os.path.exists(args.save_video):
        size_mb = os.path.getsize(args.save_video) / (1024 * 1024)
        print(f'Archivo de video resultante: {args.save_video} ({size_mb:.2f} MB)')
    print('=================================================================')

if __name__ == '__main__':
    main()
