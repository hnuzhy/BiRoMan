
import os
import sys
import cv2
import json
import numpy as np

sys.path.insert(0, os.getcwd())
from src.config import cfg_dict_init as cfg_dict

#################################################################
def order_points(pts):
    """
    对四个点进行排序：左上, 右上, 右下, 左下
    Input: pts shape (4, 2)
    Output: sorted_pts shape (4, 2)
    """
    # 初始化排序后的坐标数组 (4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # 1. 寻找左上角和右下角
    # 左上角：x+y 和最小；右下角：x+y 和最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right

    # 2. 寻找右上角和左下角
    # 右上角：x-y 差值最大；左下角：x-y 差值最小
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left

    return rect

def order_points_v2(pts):
    print("detected four points:", pts.tolist())
    rect = np.zeros((4, 2), dtype="float32"); flags = [0,0,0,0]
    px_list = pts[:, 0].tolist(); px_list.sort()  # from small to large
    py_list = pts[:, 1].tolist(); py_list.sort()  # from small to large
    
    for i, (px, py) in enumerate(pts):
        if py_list.index(py) in [0,1] and px_list.index(px) in [0]: rect[0] = pts[i]; flags[0] = 1  # Top-left
        if py_list.index(py) in [0,1] and px_list.index(px) in [3]: rect[1] = pts[i]; flags[1] = 1  # Top-right
        if py_list.index(py) in [2,3] and px_list.index(px) in [3]: rect[2] = pts[i]; flags[2] = 1  # Bottom-right
        if py_list.index(py) in [2,3] and px_list.index(px) in [0]: rect[3] = pts[i]; flags[3] = 1  # Bottom-left
    if sum(flags) != 4:
        print("#####[matching stage +1][warning] double check for some special cases!!!", flags)
        if flags[0] == 0:
            for i, (px, py) in enumerate(pts):
                if py_list.index(py) in [0,1] and px_list.index(px) in [0,1]: rect[0] = pts[i]; flags[0] = 1  # Top-left
        if flags[1] == 0:
            for i, (px, py) in enumerate(pts):
                if py_list.index(py) in [0,1] and px_list.index(px) in [2,3]: rect[1] = pts[i]; flags[1] = 1  # Top-right
        if flags[2] == 0:
            for i, (px, py) in enumerate(pts):
                if py_list.index(py) in [2,3] and px_list.index(px) in [2,3]: rect[2] = pts[i]; flags[2] = 1  # Bottom-right
        if flags[3] == 0:
            for i, (px, py) in enumerate(pts):
                if py_list.index(py) in [2,3] and px_list.index(px) in [0,1]: rect[3] = pts[i]; flags[3] = 1  # Bottom-left
    if sum(flags) != 4:
        print("#####[matching stage +2][warning] double check for some special cases!!!", flags)
        if flags[0] == 0:
            for i, (px, py) in enumerate(pts):
                if py_list.index(py) in [0,1] and px_list.index(px) in [0,1,2]: rect[0] = pts[i]; flags[0] = 1  # Top-left
        if flags[1] == 0:
            for i, (px, py) in enumerate(pts):
                if py_list.index(py) in [0,1] and px_list.index(px) in [1,2,3]: rect[1] = pts[i]; flags[1] = 1  # Top-right
        if flags[2] == 0:
            for i, (px, py) in enumerate(pts):
                if py_list.index(py) in [2,3] and px_list.index(px) in [1,2,3]: rect[2] = pts[i]; flags[2] = 1  # Bottom-right
        if flags[3] == 0:
            for i, (px, py) in enumerate(pts):
                if py_list.index(py) in [2,3] and px_list.index(px) in [0,1,2]: rect[3] = pts[i]; flags[3] = 1  # Bottom-left
    assert sum(flags) == 4, "[Error] You should have to find and match four corner points rightly!"
    return rect

def order_points_v3(pts):
    """
    鲁棒的四点排序函数：左上 -> 右上 -> 右下 -> 左下
    基于重心角度排序，解决分布不均和近似正方形时的排序歧义。
    """
    print("detected ordered points:", pts.tolist())
    pts = np.array(pts, dtype="float32")
    
    # 1. 计算重心 (Centroid)
    # axis=0 表示沿列求平均，得到 (mean_x, mean_y)
    center = np.mean(pts, axis=0)

    # 2. 计算每个点相对于重心的角度
    # atan2(y, x) 返回值的范围是 [-pi, pi]
    # 在图像坐标系中（y轴向下）：
    # TL (左上, x<cx, y<cy) -> 角度约 -135度 (-3pi/4)
    # TR (右上, x>cx, y<cy) -> 角度约 -45度 (-pi/4)
    # BR (右下, x>cx, y>cy) -> 角度约 +45度 (pi/4)
    # BL (左下, x<cx, y>cy) -> 角度约 +135度 (3pi/4)
    diff = pts - center
    angles = np.arctan2(diff[:, 1], diff[:, 0])
    
    # 3. 根据角度进行排序
    # argsort 得到的是从小到大的索引序列
    # 按照上面的分析，排序后的顺序自然就是 TL -> TR -> BR -> BL
    sort_indices = np.argsort(angles)
    sorted_pts = pts[sort_indices]
    
    # 4. 寻找真正的“左上角”并对齐
    # 虽然角度排序通常能让 TL 排在第一位，但如果物体旋转角度较大（例如菱形），
    # -pi 和 pi 的交界处可能会截断序列。
    # 为了双重保险，我们在排序后的数组中，找到距离原点(0,0)最近的点作为真正的 TL。
    
    # 计算到 (0,0) 的欧氏距离平方
    dists = np.sum(sorted_pts**2, axis=1)
    tl_idx = np.argmin(dists)
    
    # 将数组循环移位，使得 tl_idx 变成第 0 个元素
    # roll 的负号表示向左移位
    final_pts = np.roll(sorted_pts, -tl_idx, axis=0)
    
    return final_pts
    
def find_basket_corners(contour):
    """
    从轮廓中找到近似的4个角点
    """
    # 1. 计算凸包 (Convex Hull) 以去除内部凹陷噪声
    hull = cv2.convexHull(contour)
    
    # 2. 多边形逼近 (Polygon Approximation)
    # 我们通过循环调整 epsilon (逼近精度)，试图找到恰好为4个点的拟合
    perimeter = cv2.arcLength(hull, True)
    
    # 初始 epsilon 系数，通常从 0.01 到 0.1 之间尝试
    # epsilon 越大，忽略的细节越多，保留的点越少
    found_quad = False
    approx_curve = None
    
    # 动态搜索策略
    for eps_factor in np.linspace(0.01, 0.1, 20):
        epsilon = eps_factor * perimeter
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == 4:
            approx_curve = approx
            found_quad = True
            break
            
    # 如果上面的简单搜索没找到（比如圆角很大，或者形状很奇怪），
    # 兜底方案：使用最小外接矩形 (Rotated Rectangle) 或者 强制取凸包最远点
    # 这里演示使用 MinAreaRect 作为兜底，但通常上面的循环能解决99%的问题
    if not found_quad:
        print("[Warning]: extract 4 corners failed with approxPolyDP, using MinAreaRect fallback.")
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        # return order_points(box)
        return order_points_v3(box)

    # 将 shape (4, 1, 2) 转换为 (4, 2)
    pts = approx_curve.reshape(4, 2)

    # 3. 对点进行排序
    # sorted_pts = order_points(pts)
    sorted_pts = order_points_v3(pts)
    
    dist_square_p1_p2 = (sorted_pts[0][0] - sorted_pts[1][0])**2 + (sorted_pts[0][1] - sorted_pts[1][1])**2
    dist_square_p2_p3 = (sorted_pts[2][0] - sorted_pts[1][0])**2 + (sorted_pts[2][1] - sorted_pts[1][1])**2
    if dist_square_p1_p2 < dist_square_p2_p3: sorted_pts = np.roll(sorted_pts, -1, axis=0)
        
    
    return sorted_pts

def find_dense_corners(contour, kpts_num=4):
    """
    从轮廓中找到近似的4个角点
    """
    # 1. 计算凸包 (Convex Hull) 以去除内部凹陷噪声
    hull = cv2.convexHull(contour)
    
    # 2. 多边形逼近 (Polygon Approximation)
    # 我们通过循环调整 epsilon (逼近精度)，试图找到恰好为4个点的拟合
    perimeter = cv2.arcLength(hull, True)
    
    # 初始 epsilon 系数，通常从 0.01 到 0.1 之间尝试
    # epsilon 越大，忽略的细节越多，保留的点越少
    found_quad = False
    approx_curve = None
    
    # 动态搜索策略
    # for eps_factor in np.linspace(0.01, 0.1, 20):
    for eps_factor in np.linspace(0.001, 0.1, 200):  # slower yet robuster
        epsilon = eps_factor * perimeter
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == kpts_num:
            approx_curve = approx
            found_quad = True
            break
            
    # 如果上面的简单搜索没找到（比如圆角很大，或者形状很奇怪），
    # 兜底方案：使用最小外接矩形 (Rotated Rectangle) 或者 强制取凸包最远点
    # 这里演示使用 MinAreaRect 作为兜底，但通常上面的循环能解决99%的问题
    if not found_quad and kpts_num == 4:
        print("[Warning]: extract 4 corners failed with approxPolyDP, using MinAreaRect fallback.")
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        # return order_points(box)
        return order_points_v3(box)

    # 将 shape (N, 1, 2) 转换为 (N, 2)
    pts = approx_curve.reshape(kpts_num, 2)

    # 3. 对点进行排序
    # sorted_pts = order_points(pts)
    sorted_pts = order_points_v3(pts)
    
    return sorted_pts
    
#################################################################

def get_closest_index(point, contour):
    """
    找到点 point 在 contour 数组中最近的索引
    """
    # contour shape is (N, 1, 2) usually, squeeze to (N, 2)
    pts = contour.squeeze()
    # 计算所有点到目标点的距离
    dists = np.linalg.norm(pts - point, axis=1)
    return np.argmin(dists)

def extract_segment(contour, idx_start, idx_end):
    """
    从闭合 contour 中提取从 idx_start 到 idx_end 的片段
    处理数组循环的情况
    """
    pts = contour.squeeze()
    if idx_start < idx_end:
        return pts[idx_start : idx_end + 1]
    else:
        # 跨越了数组末尾和开头
        return np.vstack((pts[idx_start:], pts[:idx_end + 1]))

def resample_path_by_length(path, num_segments):
    """
    将一条路径 path (N, 2) 按弧长等分为 num_segments 段，
    返回分割点的坐标 (num_segments + 1, 2)
    """
    if len(path) < 2:
        return np.array([path[0]] * (num_segments + 1))
    
    # 1. 计算每段线段的长度
    diffs = np.diff(path, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    
    # print("seg_lengths:", seg_lengths)
    
    # 2. 计算累计长度
    cum_dist = np.insert(np.cumsum(seg_lengths), 0, 0)
    total_len = cum_dist[-1]

    # 3. 计算目标分割位置
    target_dists = np.linspace(0, total_len, num_segments + 1)
    
    # 4. 插值找到对应的坐标点
    # 对 X 和 Y 分别根据累计距离进行插值
    new_x = np.interp(target_dists, cum_dist, path[:, 0])
    new_y = np.interp(target_dists, cum_dist, path[:, 1])

    return np.column_stack((new_x, new_y))

def line_intersection(p1, p2, p3, p4):
    """
    计算线段 p1-p2 和 p3-p4 的交点
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None  # 平行
    
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return np.array([x, y])

def get_sub_segment(full_segment, idx, total, reverse=False):
    # 将 full segments 按照长度比例切割成 sub-segments
    # 简单根据索引比例切分 (更精确的方法是再次按弧长切分 full_segment)
    # 这里为了演示效果，我们使用 split_indices 逻辑
    if len(full_segment) == 0: return np.array([])
    n = len(full_segment)
    # 寻找切分点的 index
    # 注意：anchors 是重采样出来的坐标，不一定完全等于 full_segment 里的点
    # 所以这里我们还是用 resample 后的直线端点，或者
    # 如果你非常需要保留边缘的锯齿，需要在这里做 Point-to-Curve Matching
    # 下面的实现：如果是边缘，我们用 slice；如果是内部，用直线。
    
    # 简化版：外部边也用直线连接（如果边缘很直）。
    # 增强版（如下）：根据比例从 full_segment 中切一段出来
    start_ratio = idx / total
    end_ratio = (idx + 1) / total
    if reverse:
        start_ratio, end_ratio = 1 - end_ratio, 1 - start_ratio
        
    s_idx = int(start_ratio * (n - 1))
    e_idx = int(end_ratio * (n - 1))
    
    if s_idx > e_idx: s_idx, e_idx = e_idx, s_idx
    return full_segment[s_idx : e_idx + 1]

def split_mask_into_grid(corners, contour, num_rows=3, num_cols=4):
    """
    核心函数：将不规则四边形 mask 分割成 grid cells
    Input:
        corners: (4, 2) array [TL, TR, BR, BL]
        contour: (N, 1, 2) or (N, 2) array
        num_rows: 垂直方向分割数 (短边方向的格子数, 如果短边是竖向的话)
        num_cols: 水平方向分割数
    Output:
        cells: List of polygons, where each polygon is an array of points (M, 2)
    """
    # 1. 找到四个角点在 contour 中的精确索引. 顺序: TL(0), TR(1), BR(2), BL(3)
    indices = [get_closest_index(c, contour) for c in corners]
    
    # 2. 提取四条边缘的完整路径 (保留原始边缘细节)
    edge_top_full = extract_segment(contour, indices[0], indices[1])  # Top: TL -> TR
    edge_right_full = extract_segment(contour, indices[1], indices[2])  # Right: TR -> BR
    # 注意：contour通常是逆时针或顺时针，这里我们提取顺时针方向，后续可能需要翻转。
    # 如果 contour 是顺时针，BR->BL 是自然顺序。如果是逆时针，这段其实是底边。
    # 这里假设我们按照 contour 的自然顺序提取，之后统一逻辑。
    edge_bottom_full = extract_segment(contour, indices[2], indices[3])  # Bottom: BR -> BL 
    edge_left_full = extract_segment(contour, indices[3], indices[0])  # Left: BL -> TL

    # 3. 计算边缘上的关键分割点 (Anchors)
    # Top/Bottom 被 num_cols 分割
    top_anchors = resample_path_by_length(edge_top_full, num_cols)
    # Bottom 需要反转顺序以匹配 Top (从左到右) 因为 contour 提取时是 BR->BL
    # 但为了几何计算交点，我们需要对应顺序：Bottom 应该是 BL -> BR 还是 BR -> BL?
    # 我们定义 grid[0,0] 在 Top-Left。
    # Top anchors: index 0 is TL, index -1 is TR.
    # Bottom anchors needs to be: index 0 is BL, index -1 is BR.
    # 目前 extract 得到的是 BR -> BL (基于contour顺序)。所以我们需要 reverse 后的点
    bottom_anchors = resample_path_by_length(edge_bottom_full, num_cols)[::-1]
    # Left/Right 被 num_rows 分割
    # Left: BL -> TL. We want TL -> BL (index 0 is TL). So reverse.
    left_anchors = resample_path_by_length(edge_left_full, num_rows)[::-1]
    # Right: TR -> BR. This is already Top -> Bottom.
    right_anchors = resample_path_by_length(edge_right_full, num_rows)

    # 4. 生成所有网格节点 (Grid Nodes)
    grid_nodes = np.zeros((num_rows + 1, num_cols + 1, 2))
    
    for r in range(num_rows + 1):
        for c in range(num_cols + 1):
            # 找到对应行列的四个边缘点
            p_top = top_anchors[c]
            p_bottom = bottom_anchors[c]
            p_left = left_anchors[r]
            p_right = right_anchors[r]
            
            # 计算 "竖线" (Top->Bottom) 和 "横线" (Left->Right) 的交点
            # 这种方法比简单的双线性插值更能适应透视畸变
            intersect = line_intersection(p_top, p_bottom, p_left, p_right)
            
            if intersect is None:
                # 极端平行情况兜底，使用简单的均值
                print("[intersect is None!!!]", r, c)
                intersect = (p_top + p_bottom + p_left + p_right) / 4.0
            
            grid_nodes[r, c] = intersect

    # 5. 组装每一个 Grid Cell 的多边形
    cells = []
    
    # 为了能取到原始边缘点，我们需要一种方法将 anchor 映射回 extract_segment 里的点
    # 为简化复杂度且保证速度，对于Cell的内部边，我们使用直线；
    # 对于Cell的外部边（即接触Mask边缘的边），我们尝试使用近似片段或直接使用直线。
    # **需求要求：取自2D mask边界点数组**
    # 下面实现高级逻辑：如果是边缘的格子，使用原始 contour segment；如果是内部，使用直线。
    
    # 预处理边缘片段的切割索引 (为了从 full segment 中取出一段)
    # 这里为了代码健壮性，若边缘非常不规则，精确匹配原始点较难。
    # 策略：如果该边是 Grid 的边界，则使用 contour 上的点；否则使用 grid_nodes 连线。

    for r in range(num_rows):
        for c in range(num_cols):
            # 获取该格子的四个角点 (来自于计算出的交点)
            p_tl = grid_nodes[r, c]
            p_tr = grid_nodes[r, c+1]
            p_br = grid_nodes[r+1, c+1]
            p_bl = grid_nodes[r+1, c]
            
            cell_poly = []
            
            # --- Top Edge ---
            if r == 0: # 如果是第一行，取 Top Contour 的片段
                sub = get_sub_segment(edge_top_full, c, num_cols, reverse=False)
                cell_poly.extend(sub)
            else:
                cell_poly.append(p_tl)
                # cell_poly.append(p_tr) # 下一段处理
            
            # --- Right Edge ---
            if c == num_cols - 1: # 如果是最后一列，取 Right Contour 片段
                sub = get_sub_segment(edge_right_full, r, num_rows, reverse=False)
                # 确保首尾衔接，不重复添加点
                if len(cell_poly) > 0 and len(sub) > 0:
                    if np.allclose(cell_poly[-1], sub[0]): sub = sub[1:]
                cell_poly.extend(sub)
            else:
                cell_poly.append(p_tr)
                
            # --- Bottom Edge ---
            if r == num_rows - 1: # 如果是最后一行，取 Bottom Contour 片段
                # 注意 contour 提取时是 BR->BL, 对应 Grid 顺序是 Right->Left
                sub = get_sub_segment(edge_bottom_full, num_cols - 1 - c, num_cols, reverse=False) # BR->BL direction
                if len(cell_poly) > 0 and len(sub) > 0:
                    if np.allclose(cell_poly[-1], sub[0]): sub = sub[1:]
                cell_poly.extend(sub)
            else:
                cell_poly.append(p_br)
                
            # --- Left Edge ---
            if c == 0: # 如果是第一列，取 Left Contour 片段
                # Contour 是 BL->TL, Grid 需要 BL->TL (Bottom-Left to Top-Left)
                # 顺时针构建多边形的话，这里应该是 p_bl -> p_tl
                sub = get_sub_segment(edge_left_full, num_rows - 1 - r, num_rows, reverse=False)
                if len(cell_poly) > 0 and len(sub) > 0:
                     if np.allclose(cell_poly[-1], sub[0]): sub = sub[1:]
                cell_poly.extend(sub)
            else:
                cell_poly.append(p_bl)
            
            # 转换为 int32 供 opencv 绘图
            cells.append(np.array(cell_poly, dtype=np.int32).reshape((-1, 1, 2)))

    return grid_nodes, cells


#################################################################
def sort_corners_of_towel(corners, mask_arr=None):
    assert len(corners) == 4, "We now only care about [towel] with 4 corners!!!"
    
    dist_square_p1_p2 = (corners[0][0] - corners[1][0])**2 + (corners[0][1] - corners[1][1])**2
    dist_square_p2_p3 = (corners[2][0] - corners[1][0])**2 + (corners[2][1] - corners[1][1])**2
    shortEdge_longEdge_ratio_thre = 1.10
    if dist_square_p1_p2 / dist_square_p2_p3 < shortEdge_longEdge_ratio_thre:
        return corners  # do not change the points' order
    else:
        return np.roll(corners, -1, axis=0)  # start_idx is changed into 1
        
def sort_corners_of_Tshirt(corners, mask_arr=None):
    assert len(corners) == 7, "We now only care about [T-shirt] with 7 corners!!!"
    assert mask_arr is not None, "We also need the mask_arr for sorting the points!!!"
    
    contour = np.array(mask_arr)  # shape is (N,2)
    indices = [get_closest_index(c, contour) for c in corners]
    subedges_dist_var_list = []
    for idx in range(len(corners)):
        sub_edge_arr = extract_segment(contour, indices[idx%len(corners)], indices[(idx+1)%len(corners)])
        sub_edge_dist = np.linalg.norm(sub_edge_arr - contour.mean(0), axis=1)
        subedges_dist_var_list.append(sub_edge_dist.var())
    print("subedges_dist_var_list:", subedges_dist_var_list)
    
    # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    topK = 2  # edges (left-cuff <--> left-hem) and (right-cuff <--> right-hem) are two most convoluted ones
    topK_idxs = np.argpartition(subedges_dist_var_list, -topK)[-topK:]
    topK_vals = np.array(subedges_dist_var_list)[topK_idxs]
    print("subedges_dist_var_list (topK):", topK_idxs, topK_vals)
    
    [idx_1st, idx_2nd] = topK_idxs
    if (idx_1st+3)%len(corners) == idx_2nd: start_idx = (idx_2nd+1)%len(corners)  # the left-cuff
    elif (idx_1st+4)%len(corners) == idx_2nd: start_idx = (idx_1st+1)%len(corners)  # the left-cuff
    else: start_idx = 0  # we do not find the left-cuff, thus using the default order
    
    return np.roll(corners, -start_idx, axis=0)


#################################################################

def find_four_corners_of_a_rectangle_container(mask_arr, numW=3, numL=4):

    trans_mat = cfg_dict["transform_mat_inv"]
    trans_mat_t = cfg_dict["transform_mat"]
    roi_bbox = cfg_dict["detection_roi_bbox"]
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    
    # Please note the order of these points!!!
    obj_mask = np.array(mask_arr[::-1])  # the shape is (N, 2). 
    
    ##### remove the influence of perspective transformation for computing the object centroid
    obj_mask[:, 0] += x1; obj_mask[:, 1] += y1  # remember add back the offsets in x / y
    obj_mask_temp = obj_mask.copy()
    obj_mask_trans = []
    for obj_pt in obj_mask_temp:
        temp_pt = np.dot(trans_mat, np.array([obj_pt[0], obj_pt[1], 1]).T)
        obj_pt_trans = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
        obj_mask_trans.append(obj_pt_trans)
    
    ##### compute the object centroid point for representing the object
    obj_mask_trans = np.array(obj_mask_trans, dtype=np.int32)  # the obj_mask is in contour_points format with shape (N, 2)
    obj_mask_contour = np.expand_dims(obj_mask_trans, axis=1)  # this is very important (N, 2) --> (N, 1, 2)
    M = cv2.moments(obj_mask_contour)  # compute the centroid (https://theailearner.com/tag/cv2-moments/)
    assert M['m00'] != 0, "The area of given [obj_mask] should not be zero!!!"
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
    cpt_project = [cx, cy]
    temp_pt = np.dot(trans_mat_t, np.array([cx, cy, 1]).T)
    cpt_origin = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
    
    corners = find_basket_corners(obj_mask_contour)
    print("Found positions of four corner points (TL, TR, BR, BL):\n", corners.tolist() )
    
    grid_nodes_p, grid_cells_p = split_mask_into_grid(corners.copy(), obj_mask_contour, num_rows=numW, num_cols=numL)
    
    corners_origin = []
    for (cpx, cpy) in corners:
        temp_pt = np.dot(trans_mat_t, np.array([cpx, cpy, 1]).T)
        corners_origin.append([temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]])
    
    cpt_p = [cpt_project[0], cpt_project[1]]
    cpt_o = [cpt_origin[0]-x1, cpt_origin[1]-y1]
    corners_p = [ [cpx, cpy] for (cpx, cpy) in corners ]
    corners_o = [ [cpx-x1, cpy-y1] for (cpx, cpy) in corners_origin ]
    
    return cpt_p, cpt_o, corners_p, corners_o, grid_nodes_p, grid_cells_p

def find_dense_corners_of_a_deformable_cloth(mask_arr, kpts_num, cloth_name, get_mid_anchor=False):
    
    trans_mat = cfg_dict["transform_mat_inv"]
    trans_mat_t = cfg_dict["transform_mat"]
    roi_bbox = cfg_dict["detection_roi_bbox"]
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    
    # Please note the order of these points!!!
    obj_mask = np.array(mask_arr[::-1])  # the shape is (N, 2). 
    
    ##### remove the influence of perspective transformation for computing the object centroid
    obj_mask[:, 0] += x1; obj_mask[:, 1] += y1  # remember add back the offsets in x / y
    obj_mask_temp = obj_mask.copy()
    obj_mask_trans = []
    for obj_pt in obj_mask_temp:
        temp_pt = np.dot(trans_mat, np.array([obj_pt[0], obj_pt[1], 1]).T)
        obj_pt_trans = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
        obj_mask_trans.append(obj_pt_trans)
    
    ##### compute the object centroid point for representing the object
    obj_mask_trans = np.array(obj_mask_trans, dtype=np.int32)  # the obj_mask is in contour_points format with shape (N, 2)
    obj_mask_contour = np.expand_dims(obj_mask_trans, axis=1)  # this is very important (N, 2) --> (N, 1, 2)
    M = cv2.moments(obj_mask_contour)  # compute the centroid (https://theailearner.com/tag/cv2-moments/)
    assert M['m00'] != 0, "The area of given [obj_mask] should not be zero!!!"
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
    cpt_project = [cx, cy]
    temp_pt = np.dot(trans_mat_t, np.array([cx, cy, 1]).T)
    cpt_origin = [temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]]
    
    corners = find_dense_corners(obj_mask_contour, kpts_num)
    print(f"Found positions of {kpts_num} corner points:\n", corners.tolist() )
    
    if cloth_name == "towel" and kpts_num == 4: corners_resort = sort_corners_of_towel(corners)
    # if cloth_name == "pants" and kpts_num == 7: corners_resort = sort_corners_of_pants(corners, mask_arr=obj_mask_contour)
    if cloth_name == "T-shirt" and kpts_num == 7: corners_resort = sort_corners_of_Tshirt(corners, mask_arr=obj_mask_contour)

    #####========================================================================================= 
    corners_p_middle, corners_o_middle = [], []
    if get_mid_anchor:  # compute the middle point in each sub-segment of the edge contour
        indices = [get_closest_index(c, obj_mask_contour) for c in corners_resort]
        for idx in range(kpts_num):
            s_idx, e_idx = indices[idx%len(corners)], indices[(idx+1)%kpts_num]
            sub_edge_arr = extract_segment(obj_mask_contour, s_idx, e_idx)
            quantiles_anchors = resample_path_by_length(sub_edge_arr, 2)  # split the sub_edge into two parts
            [cpx, cpy] = quantiles_anchors[1]  # do not care about the start point and end point
            corners_p_middle.append([cpx, cpy])  
            temp_pt = np.dot(trans_mat_t, np.array([cpx, cpy, 1]).T)
            corners_o_middle.append([temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]])
    #####=========================================================================================
        
    corners_origin = []
    for (cpx, cpy) in corners_resort:
        temp_pt = np.dot(trans_mat_t, np.array([cpx, cpy, 1]).T)
        corners_origin.append([temp_pt[0]/temp_pt[2], temp_pt[1]/temp_pt[2]])
    
    cpt_p = [cpt_project[0], cpt_project[1]]
    cpt_o = [cpt_origin[0], cpt_origin[1]]
    corners_p = [ [cpx, cpy] for (cpx, cpy) in corners_resort ]
    corners_o = [ [cpx, cpy] for (cpx, cpy) in corners_origin ]

    return cpt_p, cpt_o, corners_p, corners_o, corners_p_middle, corners_o_middle

#################################################################

def testing_basket_func():
    
    # img_folder = "./debug/test_mLLM_deformable_objs/test_img_LLM-1_OBJ-basket_ID-01/"
    # img_folder = "./debug/test_mLLM_deformable_objs/test_img_LLM-1_OBJ-basket_ID-02/"
    # img_folder = "./debug/test_mLLM_deformable_objs/test_img_LLM-1_OBJ-basket_ID-03/"
    # img_folder = "./debug/test_mLLM_deformable_objs/test_img_LLM-1_OBJ-basket_ID-04/"
    # img_folder = "./debug/test_mLLM_deformable_objs/test_img_LLM-1_OBJ-basket_ID-05/"
    img_folder = "./debug/test_mLLM_deformable_objs/test_img_LLM-1_OBJ-basket_ID-06/"
    
    trans_mat = cfg_dict["transform_mat_inv"]; trans_mat_t = cfg_dict["transform_mat"]
    rect_l, rect_w = cfg_dict['rect_l'], cfg_dict['rect_w']; colors_list = cfg_dict['colors_list']
    roi_bbox = cfg_dict["detection_roi_bbox"]
    [x1, y1, x2, y2] = roi_bbox  # for fast and stable detection and segmentation
    kfr_height, kfr_width = cfg_dict['kfr_height'], cfg_dict['kfr_width']  # 540, 960
    top, bottom, left, right = y1, kfr_height-y2, x1, kfr_width-x2
    colors_list = [(0,255,128), (255,255,0), (128,0,255), (0,128,128), (0,0,255)]  # lawn green, cyan, light magenta, light yellow, red
    
    file_name_list = os.listdir(img_folder)
    img_names = [i for i in file_name_list if "basket.jpg" in i]; img_names.sort()
    mask_names = [i for i in file_name_list if "basket_mask.jpg" in i]; mask_names.sort()
    json_names = [i for i in file_name_list if "basket_mask.json" in i]; json_names.sort()
    
    for tid, (img_name, mask_name, json_name) in enumerate(zip(img_names, mask_names, json_names)):
        print("\n", tid, img_name)
        
        img_cv2 = cv2.imread(os.path.join(img_folder, img_name))  # shape is (500, 800, 3)
        mask_cv2 = cv2.imread(os.path.join(img_folder, mask_name))  # shape is (500, 800, 1)
        json_dict = json.load(open(os.path.join(img_folder, json_name), "r"))  # {"bbox": ..., "mask": ...}
        
        img_cv2_plot = img_cv2.copy(); color_mask_edge = (0,255,255)  # yellow
        rect_bbox, mask_arr = json_dict["bbox"], json_dict["mask"]
        for [ptx, pty] in mask_arr: cv2.circle(img_cv2_plot, (int(ptx), int(pty)), 2, color_mask_edge, -1, cv2.LINE_AA)
        
        # img_cv2_full = cv2.copyMakeBorder(img_cv2.copy(), top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
        img_cv2_full = cv2.copyMakeBorder(img_cv2_plot.copy(), top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
        img_cv2_project = cv2.warpPerspective(img_cv2_full, trans_mat, (rect_l, rect_w))  # (540, 960) --> sub_area (650, 1000)


        cpt_p, cpt_o, corners_p, corners_o, grid_nodes_p, grid_cells_p = find_four_corners_of_a_rectangle_container(mask_arr)
        
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
        labels = ["TL", "TR", "BR", "BL"]  # blue,green,red,purple --> TL TR BR BL
        
        img_cv2_p_vis = img_cv2_project.copy()
        for i, pt in enumerate(corners_p):
            cv2.line(img_cv2_p_vis, (int(cpt_p[0]), int(cpt_p[1])), (int(pt[0]), int(pt[1])), (192,192,192), 3)
            cv2.circle(img_cv2_p_vis, (int(pt[0]), int(pt[1])), 6, colors[i], -1)
            cv2.putText(img_cv2_p_vis, labels[i], (int(pt[0])+15, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.88, colors[i], 2)
        cv2.circle(img_cv2_p_vis, (int(cpt_p[0]), int(cpt_p[1])), 8, (255,255,255), -1)  # mask center point
        
        img_cv2_o_vis = img_cv2_plot.copy()
        for i, pt in enumerate(corners_o):
            cv2.line(img_cv2_o_vis, (int(cpt_o[0]), int(cpt_o[1])), (int(pt[0]), int(pt[1])), (192,192,192), 3)
            cv2.circle(img_cv2_o_vis, (int(pt[0]), int(pt[1])), 6, colors[i], -1)
            cv2.putText(img_cv2_o_vis, labels[i], (int(pt[0])+12, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.72, colors[i], 2)
        cv2.circle(img_cv2_o_vis, (int(cpt_o[0]), int(cpt_o[1])), 8, (255,255,255), -1)  # mask center point
        
        img_cv2_p_cell = img_cv2_project.copy()
        for i, cell in enumerate(grid_cells_p):
            color = np.random.randint(0, 192, (3,)).tolist()  # a random color
            cv2.polylines(img_cv2_p_cell, [cell], isClosed=True, color=color, thickness=2)  # plot polygon
            M = cv2.moments(cell)  # compute the center point for writing the id number
            assert M["m00"] != 0, "We do not accept a grid cell with zero area!!!"
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.putText(img_cv2_p_cell, str(i), (cX-10, cY+5), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2) 
        for r in range(grid_nodes_p.shape[0]):
            for c in range(grid_nodes_p.shape[1]):
                cv2.circle(img_cv2_p_cell, (int(grid_nodes_p[r,c,0]), int(grid_nodes_p[r,c,1])), 4, (255,255,255), -1)
                
        # img_cv2_project = cv2.resize(img_cv2_project, ( int(rect_l*(y2-y1)/rect_w), (y2-y1) ) )
        # img_cv2_plot_full1 = np.hstack((img_cv2_plot, img_cv2_project))
        
        img_cv2_p_cell = cv2.resize(img_cv2_p_cell, ( int(rect_l*(y2-y1)/rect_w), (y2-y1) ) )
        img_cv2_plot_full1 = np.hstack((img_cv2_plot, img_cv2_p_cell))
        
        img_cv2_p_vis = cv2.resize(img_cv2_p_vis, ( int(rect_l*(y2-y1)/rect_w), (y2-y1) ) )
        img_cv2_plot_full2 = np.hstack((img_cv2_o_vis, img_cv2_p_vis))
        
        img_cv2_plot_full = np.vstack((img_cv2_plot_full1, img_cv2_plot_full2))
        
        cv2.imwrite(os.path.join(img_folder, img_name[:-4]+"_vis.jpg"), img_cv2_plot_full)
        
    
#################################################################

# python src/algs/slots_v1.py

#################################################################
if __name__ == "__main__":

    testing_basket_func()

