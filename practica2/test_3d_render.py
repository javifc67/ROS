import rosbag2_py, cv2, numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial import cKDTree
import scipy.sparse
import scipy.sparse.csgraph

def run_dbscan_fast(points, eps=0.55, min_samples=15):
    n = len(points)
    if n < min_samples:
        return np.full(n, -1, dtype=int)
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

def render_lidar_panel(raw_pts, width=640, height=540):
    pts = np.column_stack((-raw_pts[:, 0], -raw_pts[:, 1], raw_pts[:, 2] + 0.038195))
    valid = np.isfinite(pts).all(axis=1) & (pts[:, 0] > 0.5) & (pts[:, 0] < 35.0) & (np.abs(pts[:, 1]) < 12.0) & (pts[:, 2] > -2.0) & (pts[:, 2] < 3.0)
    pts = pts[valid]
    
    ground_mask = pts[:, 2] < -1.10
    non_ground = pts[~ground_mask]
    
    roi_mask = (non_ground[:, 0] >= 1.0) & (non_ground[:, 0] <= 25.0) & (np.abs(non_ground[:, 1]) <= 6.0)
    roi_pts = non_ground[roi_mask]
    labels = run_dbscan_fast(roi_pts, eps=0.55, min_samples=15)
    
    eye = np.array([-3.5, 0.0, 2.2])
    target = np.array([16.0, 0.0, -0.2])
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
    
    panel = np.full((height, width, 3), (22, 26, 32), dtype=np.uint8)
    
    f_proj = 480.0
    cx = width / 2.0
    cy = height / 2.0 + 40.0
    
    in_front = p_d > 0.5
    u_scr = np.int32(cx + f_proj * p_r[in_front] / p_d[in_front])
    v_scr = np.int32(cy - f_proj * p_u[in_front] / p_d[in_front])
    
    cols = np.zeros((len(sort_idx), 3), dtype=np.uint8)
    cols[is_g_sorted] = (45, 80, 50)
    cols[~is_g_sorted] = (160, 175, 190)
    
    palette = [
        (0, 255, 255),   # Yellow
        (255, 120, 0),   # Blue
        (0, 200, 255),   # Amber
        (255, 0, 255),   # Magenta
        (0, 255, 128),   # Spring green
        (180, 100, 255), # Lilac
        (50, 150, 255),  # Orange
        (255, 200, 0)    # Cyan
    ]
    
    clusters_info = []
    unique_clusters = set(labels) - {-1}
    for c_id in unique_clusters:
        c_pts = roi_pts[labels == c_id]
        if len(c_pts) >= 15:
            min_dist = float(np.min(np.linalg.norm(c_pts[:, :2], axis=1)))
            clusters_info.append({
                'id': c_id,
                'pts': c_pts,
                'min_dist': min_dist,
                'min_b': np.min(c_pts, axis=0),
                'max_b': np.max(c_pts, axis=0)
            })
    
    cols_scr = cols[in_front]
    valid_scr = (u_scr >= 1) & (u_scr < width - 1) & (v_scr >= 1) & (v_scr < height - 1)
    vu = u_scr[valid_scr]
    vv = v_scr[valid_scr]
    vc = cols_scr[valid_scr]
    
    panel[vv, vu] = vc
    panel[vv + 1, vu] = vc
    panel[vv, vu + 1] = vc
    panel[vv + 1, vu + 1] = vc
    
    for c in clusters_info:
        col = palette[c['id'] % len(palette)]
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
            lines = [
                (0,1), (1,2), (2,3), (3,0),
                (4,5), (5,6), (6,7), (7,4),
                (0,4), (1,5), (2,6), (3,7)
            ]
            for p1, p2 in lines:
                if 0 <= cu[p1] < width and 0 <= cv[p1] < height and 0 <= cu[p2] < width and 0 <= cv[p2] < height:
                    cv2.line(panel, (cu[p1], cv[p1]), (cu[p2], cv[p2]), col, 2, cv2.LINE_AA)
            top_pt = np.argmin(cv)
            cv2.putText(panel, f"C{c['id']} {c['min_dist']:.1f}m", 
                        (max(4, cu[top_pt] - 25), max(18, cv[top_pt] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    # Grid / Distance circles on the ground
    for dist in [5.0, 10.0, 15.0, 20.0, 25.0]:
        angles = np.linspace(-np.pi/4, np.pi/4, 40)
        gx = dist * np.cos(angles)
        gy = dist * np.sin(angles)
        gz = np.full_like(gx, -1.25)
        g_pts = np.column_stack((gx, gy, gz)) - eye
        gd = g_pts.dot(F)
        valid_g = gd > 0.5
        gu = np.int32(cx + f_proj * g_pts[valid_g].dot(R) / gd[valid_g])
        gv = np.int32(cy - f_proj * g_pts[valid_g].dot(U) / gd[valid_g])
        for i in range(len(gu) - 1):
            if 0 <= gu[i] < width and 0 <= gv[i] < height and 0 <= gu[i+1] < width and 0 <= gv[i+1] < height:
                cv2.line(panel, (gu[i], gv[i]), (gu[i+1], gv[i+1]), (45, 55, 70), 1, cv2.LINE_AA)
        if len(gu) > 20 and 0 <= gu[20] < width and 0 <= gv[20] < height:
            cv2.putText(panel, f"{int(dist)}m", (gu[20], gv[20]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 95, 115), 1, cv2.LINE_AA)

    return panel, clusters_info

if __name__ == '__main__':
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri='/home/sandy/ROS/practica2/rosbag2_2025_02_27-13_08_14', storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == '/ouster/points':
            msg = deserialize_message(data, PointCloud2)
            arr = pc2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=False)
            raw_pts = np.column_stack((arr['x'], arr['y'], arr['z']))
            panel, clusters = render_lidar_panel(raw_pts, 640, 540)
            cv2.imwrite('/home/sandy/ROS/practica2/test_lidar_panel.jpg', panel)
            print(f"SUCCESS: Rendered lidar panel with {len(clusters)} clusters!")
            break
