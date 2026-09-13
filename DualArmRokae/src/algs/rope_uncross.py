
import os
import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from skimage.morphology import skeletonize
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.getcwd())
from src.algs.slots_v1 import get_closest_index, resample_path_by_length

''' Deformable Linear Object (DLO) Perception '''
class RopeSkeletonExtractor:
    def __init__(self, blur_ksize=9, prune_threshold=20):
        """
        初始化提取器参数
        :param blur_ksize: 预处理高斯模糊核大小 (必须为奇数), 用于平滑边缘
        :param prune_threshold: 剪枝阈值 (像素), 小于此长度的末端分支将被移除
        """
        self.blur_ksize = blur_ksize
        self.prune_threshold = prune_threshold

    def _preprocess_mask(self, mask):
        """
        Step 1: 预处理
        通过闭运算和高斯模糊，消除边缘锯齿，防止骨架化产生大量毛刺。
        """
        # 闭运算填补内部微小空洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # closed = cv2.erode(closed, np.ones((3, 3), np.uint8))    
        
        # 高斯模糊平滑边缘
        blurred = cv2.GaussianBlur(closed, (self.blur_ksize, self.blur_ksize), 0)
        
        # 再次二值化，得到光滑的Mask
        _, smooth_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        return smooth_mask

    def _build_graph(self, skeleton):
        """
        Step 2: 将骨架图像转换为 NetworkX 图
        """
        y_idxs, x_idxs = np.where(skeleton)
        nodes = list(zip(y_idxs, x_idxs))
        
        G = nx.Graph()
        if len(nodes) < 2:
            return G
            
        # 建立快速查找集
        node_set = set(nodes)
        
        # 遍历每个像素构建边 (8邻域)
        for r, c in nodes:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in node_set:
                        # 使用欧氏距离作为权重 (直角=1, 斜角=1.414)
                        dist = math.sqrt(dr**2 + dc**2)
                        G.add_edge((r, c), (nr, nc), weight=dist)
        return G

    def _prune_spurs(self, G):
        """
        Step 3: 递归剪枝
        移除从端点出发、长度小于阈值的短分支。
        """
        clean_G = G.copy()
        changed = True
        
        while changed:
            changed = False
            degrees = dict(clean_G.degree())
            endpoints = [n for n, d in degrees.items() if d == 1]
            nodes_to_remove = set()
            
            for tip in endpoints:
                path = [tip]
                curr = tip
                is_short = True
                
                # 向内搜索，限制深度
                for _ in range(self.prune_threshold + 1):
                    neighbors = list(clean_G.neighbors(curr))
                    next_nodes = [n for n in neighbors if n not in path]
                    
                    if not next_nodes: break # 孤立线段
                    
                    next_node = next_nodes[0]
                    # 如果遇到度>2的节点(交叉点)，停止
                    if clean_G.degree(next_node) > 2:
                        break
                        
                    path.append(next_node)
                    curr = next_node
                else:
                    # 循环正常结束未break，说明超过了阈值仍未遇到主干，保留
                    is_short = False
                
                if is_short:
                    nodes_to_remove.update(path)
                    changed = True
            
            if nodes_to_remove:
                print("nodes_to_remove is not None!!!")
                clean_G.remove_nodes_from(nodes_to_remove)
                clean_G.remove_nodes_from(list(nx.isolates(clean_G)))
                
        return clean_G

    def _calculate_angle(self, p1, center, p2):
        """计算向量夹角 (用于交叉点解耦)"""
        v1 = np.array(p1) - np.array(center)
        v2 = np.array(p2) - np.array(center)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0
        
        # Cosine similarity
        cos_theta = np.clip(np.dot(v1/norm1, v2/norm2), -1.0, 1.0)
        return np.arccos(cos_theta) # 返回弧度 [0, pi]

    def _get_representative_point(self, G, start_node, cluster_nodes, lookahead_steps=10):
        """
        从 start_node 出发，沿着非 cluster 的方向走 lookahead_steps 步，
        返回此时到达的坐标点。这能更好地代表该分支的整体切线方向。
        """
        curr = start_node
        visited = set(cluster_nodes) # 避免走回交叉点中心
        visited.add(curr)
        
        path_nodes = [curr]
        
        for _ in range(lookahead_steps):
            # 获取所有邻居
            neighbors = list(G.neighbors(curr))
            # 筛选出未访问过的邻居 (即向外延伸的方向)
            valid_next = [n for n in neighbors if n not in visited]
            
            if not valid_next:
                # 走到尽头了 (分支长度 < lookahead)，直接返回当前终点
                break
            
            # 骨架化后的线通常是线性的，如果有多个分叉，取第一个即可
            # (在 resolve_crossings 之前我们已经做过 prune_spurs，所以这里通常是单线的)
            next_node = valid_next[0]
            visited.add(next_node)
            path_nodes.append(next_node)
            curr = next_node
            
        return curr

    def _resolve_crossings_v1(self, G, deg_thre=120):
        """
        Step 4: 交叉点解耦
        基于几何动量（直线优先），拆解 'X' 型交叉为两条不相交的线。
        """
        resolved_G = G.copy()
        junctions = [n for n, d in G.degree() if d >= 3]
        
        for j_node in junctions:
            neighbors = list(G.neighbors(j_node))
            if len(neighbors) < 2: continue
            
            pairs = []
            used = set()
            candidates = []
            
            # 计算所有邻居对的夹角
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    n1, n2 = neighbors[i], neighbors[j]
                    angle = self._calculate_angle(n1, j_node, n2)
                    candidates.append((angle, n1, n2))
            
            # 优先连接最接近 180度 (pi) 的对
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            for angle, n1, n2 in candidates:
                if n1 not in used and n2 not in used:
                    # 阈值：大于 120度 (2pi/3) 才认为是同一根绳子
                    if angle > np.pi * (deg_thre / 180):
                        pairs.append((n1, n2))
                        used.add(n1)
                        used.add(n2)
            
            if pairs:
                resolved_G.remove_node(j_node) # 移除中心点
                for n1, n2 in pairs:
                    # 直接连接配对点，权重设为两点间距
                    dist = np.linalg.norm(np.array(n1) - np.array(n2))
                    resolved_G.add_edge(n1, n2, weight=dist)
                
        return resolved_G

    def _resolve_crossings_v2(self, G, deg_thre=120):
        """
        Step 4: 交叉点解耦 (Fix: 支持多对配对)
        基于几何连续性，将 'X' 型 (Degree 4) 或更复杂的交叉点拆解为互不相连的线段。
        """
        resolved_G = G.copy()
        # 找到所有交叉点 (度 >= 3)
        # 注意：这里使用 list() 锁定节点列表，防止在迭代中修改图结构导致报错
        junctions = [n for n, d in G.degree() if d >= 3]
        print(junctions)
        
        for j_node in junctions:
            # 获取当前交叉点的所有邻居
            neighbors = list(G.neighbors(j_node))
            if len(neighbors) < 2: 
                continue
            
            # --- 1. 计算所有可能的配对及其角度 ---
            candidates = []
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    n1 = neighbors[i]
                    n2 = neighbors[j]
                    angle = self._calculate_angle(n1, j_node, n2)
                    candidates.append((angle, n1, n2))
            
            # --- 2. 排序：优先连接最直的 (最接近 180 度 / pi) ---
            candidates.sort(key=lambda x: x[0], reverse=True)
            print(candidates)
            
            # --- 3. 贪婪匹配 ---
            matches = []
            used_neighbors = set()
            
            for idx, (angle, n1, n2) in enumerate(candidates):
                # 如果这一对中任一点已经被配对了，则跳过
                if n1 in used_neighbors or n2 in used_neighbors:
                    continue
                
                # 角度阈值：
                # 只有当两点连线足够“直”才连接。
                # 2pi/3 (120度) 是一个经验值，能容忍一定的弯曲但排除直角转弯
                if angle > np.pi * (deg_thre / 180):
                    print("[Found]", idx, angle)
                    matches.append((n1, n2))
                    used_neighbors.add(n1)
                    used_neighbors.add(n2)
            
            # --- 4. 应用修改 ---
            if matches:
                # 只有找到了至少一对匹配，才移除中心交叉点
                # (如果没找到，说明是个T型死胡同或者角度太小，保留原样或留给剪枝处理)
                resolved_G.remove_node(j_node) 
                
                for n1, n2 in matches:
                    # 计算两点间的欧氏距离作为新边的权重
                    dist = np.linalg.norm(np.array(n1) - np.array(n2))
                    resolved_G.add_edge(n1, n2, weight=dist)

                # 注意：未被匹配的邻居（used_neighbors 之外的）
                # 随着 j_node 被移除，它们自然就与主干断开了。
                # 这对于 T 型交叉（Degree 3）是正确的：主干连通，分叉断开。
                    
        return resolved_G

    def _resolve_crossings_v3(self, G, deg_thre=120):
        """
        Step 4 改进版: 基于聚类的广义交叉点解耦
        解决 'X' 分裂为两个 'T' 以及交叉点丛集的问题。
        """
        resolved_G = G.copy()
        
        # 1. 找到所有高度节点 (Degree >= 3)
        high_degree_nodes = [n for n, d in G.degree() if d >= 3]
        
        if not high_degree_nodes:
            return resolved_G
        
        # 2. 对高度节点进行聚类
        # 构建仅包含高度节点的子图，并寻找连通分量
        # 这样，紧挨着的两个 T 型节点会被归为一个 Component
        junction_subgraph = G.subgraph(high_degree_nodes)
        junction_clusters = list(nx.connected_components(junction_subgraph))
        print(junction_clusters)
        
        for cluster in junction_clusters:
            cluster_nodes = list(cluster)
            
            # 3. 计算该交叉区域的几何重心 (Centroid)
            # cluster_nodes 是 [(y, x), ...]
            ys = [n[0] for n in cluster_nodes]
            xs = [n[1] for n in cluster_nodes]
            center_y = sum(ys) / len(ys)
            center_x = sum(xs) / len(xs)
            centroid = (center_y, center_x)
            
            # 4. 寻找所有“出口” (Exits / Arms)
            # 出口是：连接到 Cluster 内任意节点，但自己不在 Cluster 内的节点
            exits = set()
            for node in cluster_nodes:
                for neighbor in G.neighbors(node):
                    if neighbor not in cluster:
                        exits.add(neighbor)
            
            exits = list(exits)
            
            # 如果出口少于2个，说明是个死胡同，忽略
            if len(exits) < 2:
                continue

            # 5. 在所有出口之间进行贪婪角度匹配
            # 逻辑同之前，但现在的中心是 centroid，候选点是所有 exits
            candidates = []
            for i in range(len(exits)):
                for j in range(i + 1, len(exits)):
                    n1 = exits[i]
                    n2 = exits[j]
                    # 计算 n1 - centroid - n2 的角度
                    angle = self._calculate_angle(n1, centroid, n2)
                    candidates.append((angle, n1, n2))
            
            # 优先连接最直的
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            matches = []
            used_exits = set()
            
            for angle, n1, n2 in candidates:
                if n1 in used_exits or n2 in used_exits:
                    continue
                
                # 角度阈值 (120度)
                if angle > np.pi * (deg_thre / 180.0):
                    matches.append((n1, n2))
                    used_exits.add(n1)
                    used_exits.add(n2)
            
            # 6. 应用拓扑修改
            if matches:
                # 移除整个 Cluster 的所有节点 (核心操作)
                # 这会把原本纠缠在一起的复杂交叉点一次性挖掉
                resolved_G.remove_nodes_from(cluster_nodes)
                
                # 添加新的直连边
                for n1, n2 in matches:
                    dist = np.linalg.norm(np.array(n1) - np.array(n2))
                    resolved_G.add_edge(n1, n2, weight=dist)

                # 未匹配的 exits 自然断开，这正好处理了 T 型交叉中垂直的那一笔
                
        return resolved_G

    def _resolve_crossings_v4(self, G, deg_thre=120, cluster_threshold=20.0):
        """
        Step 4 终极改进版: 基于欧氏距离聚类的广义交叉点解耦
        
        Args:
            G: 输入的骨架图
            cluster_threshold: 聚类阈值(像素)。
                               如果两个交叉点距离小于此值，即使不相连也被视为同一个交叉区域。
                               建议设置为骨架线宽度的 1.5 倍左右。
        """
        resolved_G = G.copy()
        
        # 1. 找到所有高度节点 (Degree >= 3)
        high_degree_nodes = [n for n, d in G.degree() if d >= 3]
        
        if not high_degree_nodes:
            return resolved_G
        
        # 2. 构建“元图”进行空间聚类 (Spatial Clustering)
        # 我们创建一个虚拟图，节点是 high_degree_nodes
        # 如果两点距离 < threshold，则添加一条虚拟边
        meta_G = nx.Graph()
        meta_G.add_nodes_from(high_degree_nodes)
        
        # 两两计算距离 (由于交叉点通常很少，<50个，双重循环开销可忽略)
        for i in range(len(high_degree_nodes)):
            for j in range(i + 1, len(high_degree_nodes)):
                n1 = high_degree_nodes[i]
                n2 = high_degree_nodes[j]
                
                # 欧氏距离
                dist = math.sqrt((n1[0]-n2[0])**2 + (n1[1]-n2[1])**2)
                
                if dist < cluster_threshold:
                    meta_G.add_edge(n1, n2)
        
        # 3. 获取聚类结果
        # 现在，物理上接近的节点都在同一个 component 里了
        junction_clusters = list(nx.connected_components(meta_G))
        print("junction_clusters:", junction_clusters)
        
        for cluster in junction_clusters:
            cluster_nodes = list(cluster)
            
            # --- 以下逻辑与上一版类似，但更稳健 ---
            
            # 4. 计算几何重心
            ys = [n[0] for n in cluster_nodes]
            xs = [n[1] for n in cluster_nodes]
            center_y = sum(ys) / len(ys)
            center_x = sum(xs) / len(xs)
            centroid = (center_y, center_x)
            
            # 5. 寻找所有“有效出口” (Valid Exits)
            # 出口定义：是 Cluster 中某点的邻居，且该邻居不在 Cluster 内部
            exits = set()
            for node in cluster_nodes:
                for neighbor in G.neighbors(node):
                    if neighbor not in cluster:
                        exits.add(neighbor)
            
            # 过滤掉“桥接点”造成的干扰 (可选但推荐)
            # 如果两个 T 点中间隔了 1-2 个普通点，这些普通点会被视为 Exits。
            # 但因为我们移除了 Cluster 节点，这些中间点变成了孤岛，最终会被 longest_path 过滤掉。
            # 所以这里不需要复杂的过滤逻辑，算法天然鲁棒。
            
            exits = list(exits)
            print("len(exits):", len(exits), "\t", exits)
            if len(exits) < 2: continue

            # --- 关键改进: 获取远端代表点 (Representative Points) ---
            # 我们不再直接用 exit 节点计算角度，而是用延伸出去的点
            exit_vectors = []
            for exit_node in exits:
                far_point = self._get_representative_point(G, exit_node, cluster_nodes, lookahead_steps=10)
                exit_vectors.append({
                    'exit_node': exit_node,
                    'far_point': far_point
                })

            # 6. 贪婪角度匹配
            # candidates = []
            # for i in range(len(exits)):
                # for j in range(i + 1, len(exits)):
                    # n1 = exits[i]
                    # n2 = exits[j]
                    # angle = self._calculate_angle(n1, centroid, n2)
                    # candidates.append((angle, n1, n2))

            candidates = []  # (using the far_point), calculate angle (p1_far - centroid - p2_far)
            for i in range(len(exit_vectors)):
                for j in range(i + 1, len(exit_vectors)):
                    e1 = exit_vectors[i]
                    e2 = exit_vectors[j]
                    angle = self._calculate_angle(e1['far_point'], centroid, e2['far_point'])
                    candidates.append((angle, e1['exit_node'], e2['exit_node']))
                    
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            matches = []
            used_exits = set()
            
            # print("candidates:", candidates)
            for angle, n1, n2 in candidates:
                if n1 in used_exits or n2 in used_exits:
                    continue
                
                # 角度阈值 (120度)
                if angle > np.pi * (deg_thre / 180.0):
                    matches.append((n1, n2))
                    used_exits.add(n1)
                    used_exits.add(n2)
            
            if len(cluster_nodes) == 1:  # for example, a rope has the Lasso shape (Loop > Tail)
                continue  # when there are only 1 cluster_nodes, do not remove it
            
            # 7. 应用拓扑修改
            if matches:
                # 移除 Cluster 内的所有节点
                resolved_G.remove_nodes_from(cluster_nodes)
                
                # 连接匹配的出口
                for n1, n2 in matches:
                    dist = np.linalg.norm(np.array(n1) - np.array(n2))
                    resolved_G.add_edge(n1, n2, weight=dist)
                    
        return resolved_G


    def _get_longest_path_v1(self, G):
        """
        Step 5: 提取最长路径
        在解耦后的图中找到最大的连通分量，并计算其直径路径。
        """
        if G.number_of_nodes() == 0: return []
        
        # 1. 取最大的连通分量 (Main Component)
        components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        if not components: return []
        main_comp = max(components, key=lambda g: g.size(weight='weight'))
        
        # 2. 寻找该分量的端点
        degrees = dict(main_comp.degree())
        endpoints = [n for n, d in degrees.items() if d == 1]
        print("endpoints:", endpoints)
        
        # 3. 计算直径 (Diameter)
        if len(endpoints) == 0:
            # 是一个完美的环，任意断开一点作为起点
            start = list(main_comp.nodes())[0]
            # highest_deg, start_node = 0, None
            # for n, d in degrees.items():
                # if d > highest_deg: highest_deg = d; start_node = n
            # start = start_node
        else:
            start = endpoints[0]
            
        # 双次 BFS/Dijkstra 策略寻找最远点对
        # 第一次：从 start 找最远点 u
        dists_1 = nx.single_source_dijkstra_path_length(main_comp, start, weight='weight')
        u = max(dists_1, key=dists_1.get)
        
        # 第二次：从 u 找最远点 v，并获取路径
        dists_2, paths = nx.single_source_dijkstra(main_comp, u, weight='weight')
        v = max(dists_2, key=dists_2.get)
        
        return paths[v]

    def _get_longest_path_v2(self, G):
        """
        Step 5 (Bug Fix): 全局最优路径搜索
        
        修正问题:
        不再局限于 'Largest Connected Component'。
        而是遍历全图所有分量，分别计算每个分量的最长路径，最后取全局最长。
        这能完美应对 Lasso (套索) 被切断导致 '环' 比 '尾巴' 大的情况，
        或者 Lasso 未切断时确保从唯一的 '尾巴尖端' (Degree=1) 出发。
        """
        if G.number_of_nodes() == 0: 
            return []
        
        # 获取所有连通分量
        components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        if not components: 
            return []
        
        best_global_path = []
        max_global_len = -1.0
        
        # 遍历每一个连通分量，寻找各自内部的最长路径
        for idx, comp in enumerate(components):
            # 忽略极小的噪点分量 (可选优化)
            if comp.number_of_nodes() < 2:
                continue
                
            # --- 策略 A: 检查该分量是否有端点 (Degree 1) ---
            degrees = dict(comp.degree())
            # 找到该分量内所有的端点
            endpoints = [n for n, d in degrees.items() if d == 1]
            print(idx, "endpoints:", endpoints)
            
            current_best_path = []
            current_max_len = -1.0
            
            if endpoints:
                # Case 1: 有端点 (线、Lasso、树状)
                # 强制从端点出发，绝不从中间出发
                for start_node in endpoints:
                    # 计算从端点到该分量所有点的距离
                    dists = nx.single_source_dijkstra_path_length(comp, start_node, weight='weight')
                    
                    # 找到最远点
                    target_node = max(dists, key=dists.get)
                    dist = dists[target_node]
                    
                    if dist > current_max_len:
                        current_max_len = dist
                        # 重建路径
                        current_best_path = nx.shortest_path(comp, start_node, target_node, weight='weight')
            
            else:
                # Case 2: 纯环 (Ring) - 无端点
                # 任意取一点作为启发式起点
                start_node = list(comp.nodes())[0]
                
                # 第一次搜索：找最远点 u
                dists_1 = nx.single_source_dijkstra_path_length(comp, start_node, weight='weight')
                u = max(dists_1, key=dists_1.get)
                
                # 第二次搜索：从 u 找最远点 v (直径)
                dists_2, paths = nx.single_source_dijkstra(comp, u, weight='weight')
                v = max(dists_2, key=dists_2.get)
                
                current_max_len = dists_2[v]
                current_best_path = paths[v]
            
            # --- 更新全局最优 ---
            # 比较物理长度 (weight之和)，而不是节点数
            # 计算当前路径的总权重
            if current_best_path:
                # 快速计算路径权重
                # 注意: networkx 的 path_weight 需要原图边权重
                # 这里直接用 current_max_len 即可 (Dijkstra 算出来的就是 path weight)
                if current_max_len > max_global_len:
                    max_global_len = current_max_len
                    best_global_path = current_best_path

        return best_global_path

    def _get_longest_path_v3(self, G):
        """
        Step 5 (Final Version): 基于欧拉路径优先的完整骨架提取
        
        策略:
        1. 提取最大连通分量。
        2. 检查该分量是否满足 '一笔画' (Eulerian Path) 条件:
           - 奇数度节点数量为 0 (完美环) 或 2 (线、Lasso)。
        3. 如果满足，直接计算欧拉路径。这能保证遍历 Lasso 的 Loop 并回到交叉点，
           获取 'Tail + Full Loop' 的完整几何形态。
        4. 如果不满足 (比如有复杂分叉的杂乱结构)，回退到之前的 Dijkstra 直径搜索，
           确保至少能提取出最长的主干。
        """
        if G.number_of_nodes() == 0: 
            return []
        
        # 1. 获取最大连通分量
        # 我们只关心主物体，忽略细碎的噪点线段
        components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        if not components: 
            return []
        
        # 按物理长度(权重和)取最大分量
        main_comp = max(components, key=lambda g: g.size(weight='weight'))
        
        # 2. 分析奇数度节点 (Odd Degree Nodes)
        degrees = dict(main_comp.degree())
        odd_degree_nodes = [n for n, d in degrees.items() if d % 2 == 1]
        n_odd = len(odd_degree_nodes)
        
        best_path = []
        
        # --- 策略 A: 尝试欧拉路径 (一笔画 - 完美覆盖 Lasso/Line/Ring) ---
        # 条件: 奇数点个数必须是 0 或 2
        if n_odd == 0 or n_odd == 2:
            try:
                # 确定起点:
                # - 如果是 Lasso/Line (n_odd=2): 必须从度数为1的点(端点)开始
                # - 如果是 Ring (n_odd=0): 任意点皆可，networkx 会自动处理
                source_node = None
                if n_odd == 2:
                    # 优先选择度数为1的点作为起点 (绳头)，而不是度数为3的交叉点
                    # Lasso 形状中: 绳头度数1，交叉点度数3。我们要从绳头出发。
                    # Line 形状中: 两头都是1。
                    candidates = [n for n in odd_degree_nodes if degrees[n] == 1]
                    if candidates:
                        source_node = candidates[0]
                    else:
                        # 极端情况：比如两个环连在一起 (Figure 8)，两个点都是3度
                        source_node = odd_degree_nodes[0]
                
                # 计算欧拉路径 (返回的是边的列表 [(u,v), (v,w), ...])
                if n_odd == 0:
                    # 闭环使用 circuit
                    edges = list(nx.eulerian_circuit(main_comp, source=source_node))
                else:
                    # 开环使用 path
                    edges = list(nx.eulerian_path(main_comp, source=source_node))
                
                # 将边列表转换为点列表
                if edges:
                    # 取第一条边的起点
                    best_path = [edges[0][0]]
                    # 依次追加每条边的终点
                    for u, v in edges:
                        best_path.append(v)
                    
                    return best_path
                    
            except nx.NetworkXError:
                # 如果因为某种图论边缘情况导致失败，静默回退到策略 B
                pass

        # --- 策略 B: 回退到 Dijkstra 直径搜索 (鲁棒保底) ---
        # 如果图结构复杂 (例如 'Y' 型分叉，3个奇数点)，无法一笔画，
        # 我们退而求其次，寻找图中的“测地直径” (最长的一条路)。
        
        # 找所有端点 (Degree=1)
        endpoints = [n for n, d in degrees.items() if d == 1]
        print("endpoints:", endpoints)
            
        if endpoints:
            # 有端点，穷举端点寻找最长路径
            max_len = -1.0
            for start_node in endpoints:
                dists = nx.single_source_dijkstra_path_length(main_comp, start_node, weight='weight')
                target_node = max(dists, key=dists.get)
                dist = dists[target_node]
                
                if dist > max_len:
                    max_len = dist
                    best_path = nx.shortest_path(main_comp, start_node, target_node, weight='weight')
        else:
            # 纯环且非欧拉 (理论上 n_odd=0 应该进策略 A，这里是双重保底)
            start_node = list(main_comp.nodes())[0]
            dists_1 = nx.single_source_dijkstra_path_length(main_comp, start_node, weight='weight')
            u = max(dists_1, key=dists_1.get)
            dists_2, paths = nx.single_source_dijkstra(main_comp, u, weight='weight')
            v = max(dists_2, key=dists_2.get)
            best_path = paths[v]

        return best_path

    def _get_longest_path_v4(self, G):
        """
        Step 5 (Refined Fix): 基于全图拓扑特征(Global n_odd)的骨架提取
        
        修正点:
        不再盲目选择 '最大连通分量'，而是先分析全图的奇数度节点 (n_odd)。
        - 如果全图 n_odd == 2: 说明存在唯一的线性/套索结构。直接锁定包含这两个奇数点的分量
          作为主骨架 (即使存在比它更大的纯环噪声)。
        - 强制从度数为1的端点 (Tail Tip) 出发，确保覆盖 Lasso 的 Loop。
        """
        if G.number_of_nodes() == 0: 
            return []
        
        # 1. 计算全图的奇数度节点
        degrees = dict(G.degree())
        odd_degree_nodes = [n for n, d in degrees.items() if d % 2 == 1]
        n_odd_global = len(odd_degree_nodes)
        print("n_odd_global:", n_odd_global, "\t", odd_degree_nodes)
        
        target_comp = None
        source_node = None
        
        # --- 策略 A: 黄金锚点锁定 (Lasso / Line) ---
        # 如果全图恰好有 2 个奇数点，这两个点定义了唯一的路径结构
        if n_odd_global == 2:
            node_a, node_b = odd_degree_nodes
            
            # 找到这两个点所在的连通分量
            # 注意: 如果这两个点不在同一个分量 (即两条断开的线), 则此策略失效, 回退到 C
            if nx.has_path(G, node_a, node_b):
                # 获取包含这两个点的分量
                # networkx 的 connected_components 是生成器，我们需要找到包含 node_a 的那个
                # 更快的方法是直接提取子图
                node_connected = nx.node_connected_component(G, node_a)
                target_comp = G.subgraph(node_connected).copy()
                
                # 确定起点: 优先选度数为 1 的点 (Tail Tip)
                # 在 Lasso 中: Tip是1度, Junction是3度. 必须从 Tip 走.
                deg_a = degrees[node_a]
                deg_b = degrees[node_b]
                
                print(node_a, node_b, deg_a, deg_b)  # special cases for the Lasso shape
                # if deg_a == 1 and deg_b == 1:
                    # edges_a = list(nx.eulerian_path(target_comp, source=node_a))
                    # edges_b = list(nx.eulerian_path(target_comp, source=node_b))
                    # edges = edges_a if len(edges_a) > len(edges_b) else edges_b
                    # path = [edges[0][0]]
                    # for u, v in edges: path.append(v)
                    # return path
                    
                if deg_a == 1:
                    source_node = node_a
                elif deg_b == 1:
                    source_node = node_b
                else:
                    # 极其罕见: 两个3度点 (比如两个并列的环), 任选
                    source_node = node_a
                    
                # 执行欧拉路径 (一笔画)
                try:
                    edges = list(nx.eulerian_path(target_comp, source=source_node))
                    # 转换为点路径
                    path = [edges[0][0]]
                    for u, v in edges:
                        path.append(v)
                    return path
                except (nx.NetworkXError, IndexError):
                    pass # 失败则回退
        
        print("Policy A Failed!!!")
        
        # --- 策略 B: 纯闭环结构 (Ring) ---
        # 如果全图没有奇数点 (n_odd == 0), 说明全是环
        if n_odd_global == 0:
            # 取最大分量
            components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
            if components:
                main_comp = max(components, key=lambda g: g.size(weight='weight'))
                # 欧拉回路
                try:
                    # 闭环起点任意
                    edges = list(nx.eulerian_circuit(main_comp))
                    path = [edges[0][0]]
                    for u, v in edges:
                        path.append(v)
                    return path
                except nx.NetworkXError:
                    pass

        print("Policy A and B Failed!!!!!!")

        # --- 策略 C: 复杂/杂乱结构 (保底方案) ---
        # 场景: 
        # 1. n_odd > 2 (分叉多)
        # 2. n_odd == 2 但节点不连通
        # 3. 欧拉算法失败
        # 此时退化为: 遍历所有分量，寻找物理距离最长的 Dijkstra 路径
        
        components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        if not components: return []
        
        best_global_path = []
        max_global_len = -1.0
        
        for idx, comp in enumerate(components):
            if comp.number_of_nodes() < 2: continue
            
            comp_degrees = dict(comp.degree())
            endpoints = [n for n, d in comp_degrees.items() if d == 1]
            print(idx, "endpoints:", endpoints)
        
            current_path = []
            current_len = -1.0
            
            if endpoints:
                # 有端点，穷举所有端点找最长
                for start in endpoints:
                    dists = nx.single_source_dijkstra_path_length(comp, start, weight='weight')
                    target = max(dists, key=dists.get)
                    if dists[target] > current_len:
                        current_len = dists[target]
                        current_path = nx.shortest_path(comp, start, target, weight='weight')
            else:
                # 闭环分量
                start = list(comp.nodes())[0]
                dists = nx.single_source_dijkstra_path_length(comp, start, weight='weight')
                u = max(dists, key=dists.get)
                dists_2, paths = nx.single_source_dijkstra(comp, u, weight='weight')
                v = max(dists_2, key=dists_2.get)
                current_len = dists_2[v]
                current_path = paths[v]
            
            if current_len > max_global_len:
                max_global_len = current_len
                best_global_path = current_path
                
        return best_global_path


    def _postprocess_path(self, path, smooth_sigma=2.0):
        """
        Step 6: 路径后处理 (插值 + 平滑)
        
        解决两个问题:
        1. 填补因交叉点解耦产生的像素空隙 (插值)。
        2. 消除骨架化和直线连接带来的锯齿和折角 (平滑)。
        
        Args:
            path: [(x,y), ...] 原始路径列表
            smooth_sigma: 高斯平滑的 sigma 值。越大越平滑，但也可能导致曲线稍微向内收缩。
                          建议值 1.0 - 3.0。
        """
        if len(path) < 2:
            return path
            
        path = np.array(path)
        
        # --- 1. 线性插值 (Densification) ---
        # 目的: 确保路径上任意相邻两点的距离 <= 1.5 像素 (8-邻域连续)
        
        dense_path = []
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            
            dense_path.append(p1)
            
            dist = np.linalg.norm(p1 - p2)
            
            # 如果距离 > 1.5 (说明有跳变/Gap)
            if dist > 1.5:
                # 计算需要插入多少个点
                num_steps = int(np.ceil(dist))
                # 生成中间点 (不包含起点和终点，因为终点会在下一次循环添加)
                # linspace: 生成 [p1, ..., p2]
                interpolated = np.linspace(p1, p2, num_steps + 1)
                
                # 将中间点加入 (跳过第一个点p1，因为它已经被append了)
                # 我们保留浮点数精度用于平滑，最后再转int
                for pt in interpolated[1:-1]:
                    dense_path.append(pt)
                    
        dense_path.append(path[-1]) # 添加最后一个点
        dense_path = np.array(dense_path)
        
        # --- 2. 高斯平滑 (Gaussian Smoothing) ---
        # 目的: 将直线插值的生硬拐角变圆滑，同时去噪
        
        from scipy.ndimage import gaussian_filter1d
        
        # 分别对 X 和 Y 坐标序列进行一维高斯滤波
        # mode='nearest' 处理边界，防止端点发生剧烈偏移
        smooth_x = gaussian_filter1d(dense_path[:, 0], sigma=smooth_sigma, mode='nearest')
        smooth_y = gaussian_filter1d(dense_path[:, 1], sigma=smooth_sigma, mode='nearest')
        
        # 组合回 (N, 2) 数组
        final_path_float = np.column_stack((smooth_x, smooth_y))
        
        # 转回整数坐标 (用于像素索引)
        # 注意: 如果需要亚像素精度供机器人控制，可以返回 final_path_float
        final_path_int = np.rint(final_path_float).astype(np.int32)
        
        # 去重 (平滑后可能导致相邻点坐标相同)
        # 简单的去重逻辑: 只保留与上一个点不同的点
        unique_path = [tuple(final_path_int[0])]
        for i in range(1, len(final_path_int)):
            pt = tuple(final_path_int[i])
            if pt != unique_path[-1]:
                unique_path.append(pt)
                
        return unique_path


    def _calculate_tangent_at_index(self, path, idx, window=5):
        """
        计算特定索引处的切线向量和角度
        """
        # 处理边界情况：起点和终点没有完整的左右邻居
        # 使用 clip 限制索引范围
        p_start_idx = max(0, idx - window)
        p_end_idx = min(len(path) - 1, idx + window)
        
        # 如果路径太短，直接返回默认
        if p_start_idx == p_end_idx:
            return np.array([1.0, 0.0]), 0.0
            
        p_start = path[p_start_idx]
        p_end = path[p_end_idx]
        
        # 1. 计算向量
        tangent_vec = p_end - p_start
        
        # 2. 归一化
        norm = np.linalg.norm(tangent_vec)
        if norm == 0:
            return np.array([1.0, 0.0]), 0.0
        
        unit_tangent = tangent_vec / norm
        
        # 3. 计算角度 (弧度 -pi 到 pi)
        # 注意: 图像坐标系通常 y 轴向下，计算 grasp 时需要注意坐标系定义
        angle_rad = np.arctan2(unit_tangent[1], unit_tangent[0])
        
        return unit_tangent, angle_rad

    def get_quantiles(self, path_xy, segment_num=4):
        path_xy = np.array(path_xy)
        quantiles_anchors = resample_path_by_length(path_xy, segment_num)  # For example, split the whole rope into 4 segments
        quantiles_indices = [get_closest_index(c, path_xy) for c in quantiles_anchors]  # Find indices of all anchors
        return quantiles_anchors, quantiles_indices
        
    def get_grasp_info(self, path_xy, target_point=None, neighbor_window=5):
        """
        获取抓取方向信息。
        
        Args:
            path_xy: 提取出的完整路径 [(x,y), ...]
            target_point: (x, y) 目标点坐标。如果不填，则返回路径上所有点的切线信息。
            neighbor_window: 计算切线时前后参考的邻居数量 (N)。
                             值越大越平滑，但对于急转弯处可能不准。建议 3-7。
                             
        Returns:
            如果 target_point 为 None:
                returns (tangents, angles)
                - tangents: (N, 2) 每一个点的单位切线向量
                - angles: (N, ) 每一个点的弧度角
                
            如果 target_point 有值:
                returns (nearest_pt, tangent_vec, angle_rad, normal_vec)
                - nearest_pt: 路径上离目标点最近的点坐标
                - tangent_vec: 切线向量
                - angle_rad: 切线角度
                - normal_vec: 法线向量 (垂直于切线，通常是二指夹爪的闭合方向)
        """
        path_arr = np.array(path_xy)
        
        # Case A: 计算特定点的切线 (用户点击或自动选点)
        if target_point is not None:
            # 1. 找到路径上最近的点的索引
            target_arr = np.array(target_point)
            dists = np.linalg.norm(path_arr - target_arr, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_pt = path_arr[nearest_idx]
            
            # 2. 计算切线
            tangent, angle = self._calculate_tangent_at_index(path_arr, nearest_idx, neighbor_window)
            
            # 3. 计算法线 (Normal) - 顺时针旋转 90 度
            # 对于二指夹爪，通常指尖闭合方向垂直于绳子切线
            normal = np.array([-tangent[1], tangent[0]])  # +X --> right; +Y --> down
            
            return nearest_pt, tangent, angle, normal
            
        # Case B: 批量计算所有点的切线
        else:
            tangents = []
            angles = []
            for i in range(len(path_arr)):
                t, a = self._calculate_tangent_at_index(path_arr, i, neighbor_window)
                tangents.append(t)
                angles.append(a)
            
            return np.array(tangents), np.array(angles)

    def extract(self, raw_mask, cluster_thre=20):
        """
        主入口函数
        :param raw_mask: 输入的二值 Mask (0/255)
        :return: ordered_path [(x,y), ...], processed_skeleton_img
        """
        # 1. 预处理
        smooth_mask = self._preprocess_mask(raw_mask)
        
        # 2. 骨架化 (Lee 算法)
        binary = smooth_mask > 0
        skel_ori = skeletonize(binary, method='lee')
        
        # 3. 建图
        G = self._build_graph(skel_ori)
        
        # 4. 剪枝 (去毛刺)
        G_pruned = self._prune_spurs(G)
        
        # 5. 交叉点解耦 (处理自相交)
        # G_uncrossed = self._resolve_crossings_v1(G_pruned, deg_thre=90)
        # G_uncrossed = self._resolve_crossings_v2(G_pruned, deg_thre=90)
        # G_uncrossed = self._resolve_crossings_v3(G_pruned, deg_thre=90)
        G_uncrossed = self._resolve_crossings_v4(G_pruned, deg_thre=90, cluster_threshold=cluster_thre)
        
        # 6. 提取路径
        # path_yx = self._get_longest_path_v1(G_uncrossed)
        # path_yx = self._get_longest_path_v2(G_uncrossed)
        # path_yx = self._get_longest_path_v3(G_uncrossed)
        path_yx = self._get_longest_path_v4(G_uncrossed)
        
        # 坐标转换 (y,x) -> (x,y)
        path_xy_raw = [(p[1], p[0]) for p in path_yx]
       
        # 7. 后处理: 插值补全 + 平滑
        final_path_xy = self._postprocess_path(path_xy_raw, smooth_sigma=2.0)
        print("[Path Length]:", "(before)", len(path_xy_raw), "; (after)", len(final_path_xy))
       
        # 为了可视化，生成最终的骨架图
        skel_vis = np.zeros_like(raw_mask)
        if final_path_xy:
            pts = np.array(final_path_xy)
            h, w = raw_mask.shape
            valid_pts = pts[(pts[:,0] >= 0) & (pts[:,0] < w) & (pts[:,1] >= 0) & (pts[:,1] < h)]
            skel_vis[valid_pts[:,1], valid_pts[:,0]] = 255  # 由于是 (x, y)，对应的图像索引是 [y, x]
            
        return final_path_xy, skel_vis, skel_ori, smooth_mask

##########################################################################################################
def visualize_extracted_rope(path_xy, FPS=30, tangents=None, save_path="rope_path_flow.gif"):
    
    # 为了动画流畅，我们将所有轮廓点展平到一个列表中，并在切换轮廓时插入一个标记
    # 结构: [ (x, y, contour_index, point_index), ... ]
    all_points = []

    # 降采样：如果点太多动画会生成很久，这里每隔 step 取一个点
    # step = 1 表示逐像素 (最精确但最慢)
    step = 5 
    pts = path_xy[::step]
    
    quantiles_anchors = resample_path_by_length(path_xy, 4)  # For example, split the whole rope into 4 segments
    quantiles_indices = [get_closest_index(c, path_xy) for c in quantiles_anchors]  # Find indices of all anchors

    for p_idx, pt in enumerate(pts):
        all_points.append((pt[0], pt[1], p_idx * step))

    # 3. 设置 Matplotlib 动画
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(wspace=0.99, hspace=0.99, left=0.01, right=0.99, top=0.95, bottom=0.01)
    ax.set_title("Extracted Rope Path (Tracing the Centerline)", fontsize=20)
    ax.imshow(mask, cmap='gray')
    ax.axis('off')

    # 动态元素：一个红点代表当前遍历位置
    scat = ax.scatter([], [], c='red', s=75, edgecolors='white', zorder=11)
    # 文本显示当前索引信息
    text = ax.text(10, 50, "", color="yellow", fontsize=16, fontweight='bold')

    scat_anchor = ax.scatter([], [], c='black', s=75, edgecolors='yellow', zorder=10)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        text.set_text("")
        return scat, text

    def update(frame):
        if frame >= len(all_points): return scat, text
        x, y, p_idx = all_points[frame]
        scat.set_offsets([[x, y]])
        text.set_text(f"Point Index: {p_idx}\nPos: ({x}, {y})")
        history_pt_xy_list = []
        for q_idx, pt_xy in zip(quantiles_indices, quantiles_anchors):  
            if p_idx >= q_idx:
                history_pt_xy_list.append(pt_xy)
                if tangents is not None:  # plot the tangent line of each anchor point
                    x_offset, y_offset = tangents[q_idx][0] * 50, tangents[q_idx][1] * 50
                    start_tan = (int(pt_xy[0] - x_offset), int(pt_xy[1] - y_offset))
                    end_tan = (int(pt_xy[0] + x_offset), int(pt_xy[1] + y_offset))
                    ax.plot([start_tan[0], end_tan[0]], [start_tan[1], end_tan[1]],
                        color='cyan', marker='o', linestyle='-', linewidth=1, markersize=3)
        scat_anchor.set_offsets(history_pt_xy_list)  # plotting all quantiles_anchors
        return scat, text

    all_points += [all_points[-1]]*FPS  # Let the last frame linger for one more second.
    
    # 创建动画
    # interval: 帧间隔(ms)
    ani = FuncAnimation(fig, update, frames=len(all_points), init_func=init, blit=True, interval=20)
    
    print(f"正在生成动画 (共 {len(all_points)} 帧)... 请稍候")
    # 保存为 GIF
    writer = PillowWriter(fps=FPS)
    ani.save(save_path, writer=writer)
    print(f"动画已保存至: {save_path}")
    plt.close()
##########################################################################################################


# ==========================================
# 单元测试与可视化 Demo
# ==========================================
if __name__ == "__main__":

    
    # demo_obj_name, cluster_thre = "test_OBJ-cable_ID-01", 40  # a slim white cable (20 ~ 40 are all OK)
    # demo_test_ids = [ "01", "09", "17", "25", "29", "33", "37", "45", "53", "57", "61", "65", "73", "81", "85" ]
    # test_ids_list = ["01", "25", "37", "45", "53", "65", "73", "81", "85" ] # This bad case ("53") can be fixed in _resolve_crossings_v4() without using the far_point
        
    # demo_obj_name, cluster_thre = "test_OBJ-cable_ID-02", 40  # a thick black cable (40 is the best choice)
    # demo_test_ids = [ "01", "09", "17", "21", "25", "33", "41", "45", "53", "61", "65", "73", "77", "81", "89" ]
    # test_ids_list = ["01", "09", "41", "45", "65", "73", "77", "81", "89" ]

    demo_obj_name, cluster_thre = "test_OBJ-rope_ID-01", 40  # a slim red nylon rope (40 is the best choice)
    demo_test_ids = [ "01", "09", "17", "29", "33", "37", "53", "65", "69", "73", "77", "85" ]
    test_ids_list = [ "01", "09", "33", "37", "53", "65", "73", "77", "85" ]
    
    for idx, test_id in enumerate(demo_test_ids):
        demo_mask_path = f"./debug/test_pcdVLM_deformable_cloth/{demo_obj_name}/FRAME{test_id}_mask.jpg"
        demo_img_path = f"./debug/test_pcdVLM_deformable_cloth/{demo_obj_name}/FRAME{test_id}_rgb_L.jpg"
        print("\n", idx, demo_mask_path )
        
        # 1. 创建测试 Mask
        mask = cv2.cvtColor(cv2.imread(demo_mask_path), cv2.COLOR_BGR2GRAY)  
        img = cv2.cvtColor(cv2.convertScaleAbs(cv2.imread(demo_img_path), alpha=1.2, beta=20), cv2.COLOR_BGR2RGB)

        # 2. 实例化并运行
        extractor = RopeSkeletonExtractor(blur_ksize=11, prune_threshold=15)
        path_xy, result_skel, origin_skel, smooth_mask = extractor.extract(mask, cluster_thre=cluster_thre)
        
        print(f"提取完成，路径节点数: {len(path_xy)}")
        if len(path_xy) > 0: print(f"起点: {path_xy[0]}, 终点: {path_xy[-1]}")
        
        tangents, angles = extractor.get_grasp_info(path_xy, neighbor_window=5)

        # 3. 可视化对比
        vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if len(path_xy) > 1:
            # 绘制提取出的平滑路径 (红色)
            pts_arr = np.array(path_xy, dtype=np.int32)
            cv2.polylines(vis_img, [pts_arr], False, (0, 0, 255), 2)
            # 起点(绿) 终点(蓝)
            cv2.circle(vis_img, tuple(path_xy[0]), 6, (0, 255, 0), -1)  # start point (green)
            cv2.circle(vis_img, tuple(path_xy[-1]), 6, (255, 0, 0), -1)  # end point (blue)

        fig = plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 2, 1)
        # plt.title("Original Noisy Mask")
        plt.title("Original Image (Non-Preprocessed)")
        plt.imshow(img, cmap='gray')
        
        plt.subplot(2, 2, 2)
        plt.title("Smoothed Mask (Preprocessed)")
        plt.imshow(smooth_mask, cmap='gray')

        plt.subplot(2, 2, 3)
        plt.title("Skeleton Result (Preprocessed)")
        plt.imshow(origin_skel, cmap='gray')
        # plt.imshow(result_skel, cmap='gray')

        plt.subplot(2, 2, 4)
        plt.title("Final Centerline (Red)")
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        
        plt.tight_layout()
        # plt.show()
        
        # continue

        if test_id in test_ids_list:
            gif_saved_path = f"./debug/test_pcdVLM_rope/RopeFlow_{demo_obj_name}_FRAME{test_id}.gif"
            visualize_extracted_rope(np.array(path_xy, dtype=np.int32), tangents=tangents, save_path=gif_saved_path)
            fig.savefig(gif_saved_path[:-4]+".jpg", dpi=fig.dpi, bbox_inches='tight')


