#!/usr/bin/env python3
"""
Nodo de deteccion de obstaculos mediante LIDAR Ouster (PointCloud2) con Clustering DBSCAN.
Procesa la nube de puntos 3D en tiempo real, aplica filtrado espacial ROI,
agrupa obstaculos mediante DBSCAN (SciPy cKDTree / Sparse Graph), publica alertas
en /obstacles/lidar y genera el stream de video BEV en /obstacles/lidar/cluster_image.
Soporta reconfiguracion dinamica de parametros en tiempo de ejecucion.
"""

import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
from scipy.spatial import cKDTree

from obstacle_detector_interfaces.msg import ObstacleInfo

# Paleta de colores para visualizacion de clusters (BGR)
CLUSTER_PALETTE = [
    (255, 240, 0),    # Cyan
    (0, 165, 255),    # Naranja
    (255, 0, 200),    # Magenta
    (0, 255, 255),    # Amarillo
    (50, 255, 50),    # Lima
    (200, 50, 255),   # Violeta
    (0, 100, 255),    # Rojo-naranja
    (255, 128, 0),    # Azul cielo
    (180, 255, 0),    # Turquesa
    (0, 255, 150)     # Verde primavera
]


class LidarObstacleDetector(Node):
    """
    Nodo que procesa nubes de puntos de LIDAR aplicando DBSCAN para segmentar y detectar obstaculos.
    """

    def __init__(self):
        super().__init__('lidar_obstacle_detector')

        # Declaracion de parametros configurables en tiempo de ejecucion
        self.declare_parameter('roi_x_min', 0.2)
        self.declare_parameter('roi_x_max', 10.0)
        self.declare_parameter('roi_y_min', -2.5)
        self.declare_parameter('roi_y_max', 2.5)
        self.declare_parameter('roi_z_min', -1.0)
        self.declare_parameter('roi_z_max', 1.5)
        self.declare_parameter('warning_distance', 3.0)
        self.declare_parameter('danger_distance', 1.5)
        self.declare_parameter('min_points_cluster', 15)
        self.declare_parameter('dbscan_eps', 0.45)
        self.declare_parameter('publish_cluster_image', True)

        self._load_parameters()
        self.add_on_set_parameters_callback(self._on_parameters_change)

        self.bridge = CvBridge()

        # Suscriptor al topic de LIDAR (/ouster/points)
        self.subscription = self.create_subscription(
            PointCloud2,
            '/ouster/points',
            self._pointcloud_callback,
            10
        )

        # Publicador de informacion estructurada de obstaculos
        self.publisher = self.create_publisher(
            ObstacleInfo,
            '/obstacles/lidar',
            10
        )

        # Publicador de imagen de radar BEV con clusters DBSCAN
        self.image_pub = self.create_publisher(
            Image,
            '/obstacles/lidar/cluster_image',
            10
        )

        self.get_logger().info(
            'LidarObstacleDetector inicializado con Clustering DBSCAN. '
            f'ROI X: [{self.roi_x_min:.1f}, {self.roi_x_max:.1f}]m, '
            f'DBSCAN eps: {self.dbscan_eps:.2f}m, min_samples: {self.min_points_cluster}'
        )

    def _load_parameters(self):
        """Carga y almacena los parametros del nodo."""
        self.roi_x_min = float(self.get_parameter('roi_x_min').value)
        self.roi_x_max = float(self.get_parameter('roi_x_max').value)
        self.roi_y_min = float(self.get_parameter('roi_y_min').value)
        self.roi_y_max = float(self.get_parameter('roi_y_max').value)
        self.roi_z_min = float(self.get_parameter('roi_z_min').value)
        self.roi_z_max = float(self.get_parameter('roi_z_max').value)
        self.warning_distance = float(self.get_parameter('warning_distance').value)
        self.danger_distance = float(self.get_parameter('danger_distance').value)
        self.min_points_cluster = int(self.get_parameter('min_points_cluster').value)
        self.dbscan_eps = float(self.get_parameter('dbscan_eps').value)
        self.publish_cluster_image = bool(self.get_parameter('publish_cluster_image').value)

    def _on_parameters_change(self, params):
        """Callback ejecutado cuando se modifica un parametro dinamicamente con ros2 param set."""
        for param in params:
            self.get_logger().info(f'Actualizando parametro dinamico "{param.name}" -> {param.value}')
            if param.name == 'roi_x_min':
                self.roi_x_min = float(param.value)
            elif param.name == 'roi_x_max':
                self.roi_x_max = float(param.value)
            elif param.name == 'roi_y_min':
                self.roi_y_min = float(param.value)
            elif param.name == 'roi_y_max':
                self.roi_y_max = float(param.value)
            elif param.name == 'roi_z_min':
                self.roi_z_min = float(param.value)
            elif param.name == 'roi_z_max':
                self.roi_z_max = float(param.value)
            elif param.name == 'warning_distance':
                self.warning_distance = float(param.value)
            elif param.name == 'danger_distance':
                self.danger_distance = float(param.value)
            elif param.name == 'min_points_cluster':
                self.min_points_cluster = int(param.value)
            elif param.name == 'dbscan_eps':
                self.dbscan_eps = float(param.value)
            elif param.name == 'publish_cluster_image':
                self.publish_cluster_image = bool(param.value)

        return SetParametersResult(successful=True)

    def _cluster_dbscan(self, points):
        """Aplica DBSCAN utilizando cKDTree y componentes conexas de grafos dispersos."""
        n = len(points)
        if n < self.min_points_cluster:
            return np.full(n, -1, dtype=int)

        try:
            import scipy.sparse
            import scipy.sparse.csgraph

            tree = cKDTree(points)
            adj = tree.sparse_distance_matrix(tree, max_distance=self.dbscan_eps, output_type='coo_matrix')
            degrees = np.bincount(adj.row, minlength=n)
            core_mask = (degrees >= self.min_points_cluster)

            valid_edges = core_mask[adj.row] | core_mask[adj.col]
            filtered_adj = scipy.sparse.csr_matrix(
                (np.ones(valid_edges.sum(), dtype=bool), (adj.row[valid_edges], adj.col[valid_edges])),
                shape=(n, n)
            )
            n_comp, comp_labels = scipy.sparse.csgraph.connected_components(filtered_adj, directed=False)

            comp_has_core = np.zeros(n_comp, dtype=bool)
            np.logical_or.at(comp_has_core, comp_labels, core_mask)
            comp_sizes = np.bincount(comp_labels, minlength=n_comp)
            valid_comp = [c for c in np.where(comp_has_core)[0] if comp_sizes[c] >= self.min_points_cluster]

            labels = np.full(n, -1, dtype=int)
            for new_id, old_id in enumerate(valid_comp):
                labels[comp_labels == old_id] = new_id
            return labels

        except Exception as e:
            self.get_logger().error(f'Error en DBSCAN: {e}', throttle_duration_sec=2.0)
            return np.full(n, -1, dtype=int)

    def _draw_cluster_radar(self, points, labels, closest_dist, severity):
        """Genera una imagen tipo radar Bird's-Eye-View (BEV) con los clusters DBSCAN."""
        h, w = 600, 600
        img = np.full((h, w, 3), (18, 22, 28), dtype=np.uint8)

        max_x = self.roi_x_max + 1.0
        max_y = max(abs(self.roi_y_min), abs(self.roi_y_max)) + 1.0
        scale = min((h - 80) / max_x, (w / 2 - 30) / max_y)
        origin_u = w // 2
        origin_v = h - 60

        # Anillos concentricos
        for r in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
            if r <= max_x:
                r_px = int(r * scale)
                cv2.ellipse(img, (origin_u, origin_v), (r_px, r_px), 0, 180, 360, (40, 50, 65), 1, cv2.LINE_AA)
                cv2.putText(img, f'{int(r)}m', (origin_u + 6, origin_v - r_px - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 110, 130), 1, cv2.LINE_AA)

        # Zonas de peligro y aviso
        cv2.ellipse(img, (origin_u, origin_v), (int(self.danger_distance * scale), int(self.danger_distance * scale)), 0, 180, 360, (0, 0, 200), 2, cv2.LINE_AA)
        cv2.ellipse(img, (origin_u, origin_v), (int(self.warning_distance * scale), int(self.warning_distance * scale)), 0, 180, 360, (0, 145, 255), 2, cv2.LINE_AA)

        # Vehiculo EGO
        veh_w = max(16, int(1.2 * scale))
        veh_l = max(22, int(1.8 * scale))
        cv2.rectangle(img, (origin_u - veh_w//2, origin_v - veh_l//2), (origin_u + veh_w//2, origin_v + veh_l//2), (0, 210, 255), -1)
        cv2.putText(img, 'ROBOT', (origin_u - 16, origin_v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)

        # Proyeccion de puntos
        if len(points) > 0:
            all_u = np.int32(origin_u - points[:, 1] * scale)
            all_v = np.int32(origin_v - points[:, 0] * scale)

            # Ruido
            noise_mask = (labels == -1)
            nu = all_u[noise_mask]
            nv = all_v[noise_mask]
            val_n = (nu >= 0) & (nu < w) & (nv >= 0) & (nv < h)
            img[nv[val_n], nu[val_n]] = (50, 60, 70)

            # Clusters
            unique_labels = sorted(list(set(labels) - {-1}))
            for c_id in unique_labels:
                c_mask = (labels == c_id)
                color = CLUSTER_PALETTE[c_id % len(CLUSTER_PALETTE)]
                cu = all_u[c_mask]
                cv_ = all_v[c_mask]
                val_c = (cu >= 1) & (cu < w - 1) & (cv_ >= 1) & (cv_ < h - 1)
                for du in [-1, 0, 1]:
                    for dv in [-1, 0, 1]:
                        img[cv_[val_c] + dv, cu[val_c] + du] = color

                # Bounding box
                c_pts = points[c_mask]
                cx, cy = np.mean(c_pts[:, :2], axis=0)
                cdist = math.hypot(cx, cy)
                u1 = int(origin_u - c_pts[:, 1].max() * scale)
                u2 = int(origin_u - c_pts[:, 1].min() * scale)
                v1 = int(origin_v - c_pts[:, 0].max() * scale)
                v2 = int(origin_v - c_pts[:, 0].min() * scale)
                cv2.rectangle(img, (max(0, u1 - 3), max(0, v1 - 3)), (min(w - 1, u2 + 3), min(h - 1, v2 + 3)), color, 2, cv2.LINE_AA)
                cv2.putText(img, f'C{c_id}:{cdist:.1f}m', (max(2, u1), max(14, v1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        # Banner de Estado
        sev_color = (0, 0, 220) if severity == 'DANGER' else ((0, 160, 255) if severity == 'WARNING' else (0, 200, 50))
        cv2.rectangle(img, (0, 0), (w, 36), (25, 30, 40), -1)
        cv2.line(img, (0, 36), (w, 36), sev_color, 2)
        cv2.putText(img, f'LIDAR DBSCAN | Estado: {severity} | Min: {closest_dist:.2f}m',
                    (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        return img

    def _pointcloud_callback(self, msg: PointCloud2):
        """Procesa cada mensaje de PointCloud2 del LIDAR con segmentacion DBSCAN."""
        try:
            struct_arr = pc2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=False)
            if struct_arr.size == 0:
                return
            raw = np.column_stack((struct_arr['x'], struct_arr['y'], struct_arr['z']))
            if raw.size == 0:
                return
            # Transformacion de coordenadas os_lidar -> os_sensor (avance frontal x = -x_raw, y = -y_raw)
            pts = np.column_stack((-raw[:, 0], -raw[:, 1], raw[:, 2] + 0.038195))

            valid = np.isfinite(pts).all(axis=1)
            pts_finite = pts[valid]

            mask_roi = (
                (pts_finite[:, 0] >= self.roi_x_min) &
                (pts_finite[:, 0] <= self.roi_x_max) &
                (pts_finite[:, 1] >= self.roi_y_min) &
                (pts_finite[:, 1] <= self.roi_y_max) &
                (pts_finite[:, 2] >= self.roi_z_min) &
                (pts_finite[:, 2] <= self.roi_z_max)
            )

            roi_points = pts_finite[mask_roi]

            out_msg = ObstacleInfo()
            out_msg.header = msg.header
            out_msg.sensor_type = 'lidar'

            # Aplicar clustering DBSCAN
            labels = self._cluster_dbscan(roi_points)
            unique_clusters = set(labels) - {-1}

            closest_dist = float('inf')
            closest_point = np.zeros(3)
            severity = 'SAFE'

            if len(unique_clusters) > 0:
                # Evaluar cada cluster y localizar el mas critico
                for c_id in unique_clusters:
                    c_pts = roi_points[labels == c_id]
                    dists = np.linalg.norm(c_pts[:, :2], axis=1)
                    min_idx = np.argmin(dists)
                    min_c_dist = float(dists[min_idx])
                    if min_c_dist < closest_dist:
                        closest_dist = min_c_dist
                        closest_point = c_pts[min_idx]

                out_msg.obstacle_detected = True
                out_msg.distance = closest_dist
                out_msg.x = float(closest_point[0])
                out_msg.y = float(closest_point[1])
                out_msg.z = float(closest_point[2])

                if closest_dist <= self.danger_distance:
                    severity = 'DANGER'
                    self.get_logger().error(
                        f'[LIDAR-PELIGRO] Obstaculo a {closest_dist:.2f}m (Clusters DBSCAN: {len(unique_clusters)})',
                        throttle_duration_sec=1.0
                    )
                elif closest_dist <= self.warning_distance:
                    severity = 'WARNING'
                    self.get_logger().warn(
                        f'[LIDAR-AVISO] Obstaculo a {closest_dist:.2f}m (Clusters DBSCAN: {len(unique_clusters)})',
                        throttle_duration_sec=1.0
                    )
                else:
                    severity = 'SAFE'
            else:
                out_msg.obstacle_detected = False
                out_msg.distance = float('inf')
                out_msg.x = 0.0
                out_msg.y = 0.0
                out_msg.z = 0.0
                severity = 'SAFE'

            out_msg.severity = severity
            self.publisher.publish(out_msg)

            # Publicar imagen radar de clusters si esta habilitado
            if self.publish_cluster_image:
                radar_img = self._draw_cluster_radar(roi_points, labels, closest_dist if closest_dist < 999.0 else 0.0, severity)
                img_msg = self.bridge.cv2_to_imgmsg(radar_img, encoding='bgr8')
                img_msg.header = msg.header
                self.image_pub.publish(img_msg)

        except Exception as e:
            self.get_logger().error(f'Error procesando PointCloud2: {str(e)}', throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = LidarObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
