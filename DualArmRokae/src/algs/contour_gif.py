
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def create_demo_crossing_rope_mask():
    """
    创建一个模拟的自交叉（ figure-8 形状）绳子 Mask
    """
    h, w = 400, 400
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 使用多边形填充来模拟有宽度的绳子
    # 模拟一个 "8" 字结
    
    # 上半环
    pts_top = np.array([[100, 100], [300, 100], [250, 200], [150, 200]], np.int32)
    cv2.polylines(mask, [pts_top.reshape(-1, 1, 2)], isClosed=False, color=255, thickness=40)
    
    # 下半环 (交叉)
    pts_bot = np.array([[150, 200], [100, 300], [300, 300], [250, 200]], np.int32)
    cv2.polylines(mask, [pts_bot.reshape(-1, 1, 2)], isClosed=False, color=255, thickness=40)
    
    # 稍微平滑一下，使其更像真实的分割Mask
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask

def visualize_contour_traversal(mask, save_path="contour_flow.gif"):
    """
    动态可视化 mask 轮廓点的遍历顺序
    """
    # 1. 提取轮廓
    # 使用 RETR_CCOMP 获取两层结构：外轮廓和内孔洞 (这对于交叉绳索形成的环非常重要)
    # 使用 CHAIN_APPROX_NONE 确保我们看到的是逐像素的真实移动，而不是压缩后的顶点
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        print("未找到轮廓！")
        return

    print(f"检测到 {len(contours)} 条轮廓 (包含外边界和内部孔洞)")

    # 2. 准备绘图数据
    vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # 为了动画流畅，我们将所有轮廓点展平到一个列表中，并在切换轮廓时插入一个标记
    # 结构: [ (x, y, contour_index, point_index), ... ]
    all_points = []
    
    for c_idx, contour in enumerate(contours):
        # 降采样：如果点太多动画会生成很久，这里每隔 step 取一个点
        # step = 1 表示逐像素 (最精确但最慢)
        step = 5 
        pts = contour[::step]
        
        for p_idx, pt in enumerate(pts):
            x, y = pt[0]
            all_points.append((x, y, c_idx, p_idx * step))

    # 3. 设置 Matplotlib 动画
    # fig, ax = plt.subplots(figsize=(8, 8))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Contour Traversal Order (Tracing the Boundary)")
    ax.imshow(mask, cmap='gray')
    ax.axis('off')

    # 静态元素：画出所有轮廓的细线作为背景
    for c in contours:
        ax.plot(c[:, 0, 0], c[:, 0, 1], 'c-', linewidth=1, alpha=0.5)

    # 动态元素：一个红点代表当前遍历位置
    scat = ax.scatter([], [], c='red', s=75, edgecolors='white', zorder=10)
    # 文本显示当前索引信息
    text = ax.text(10, 75, "", color="yellow", fontsize=12, fontweight='bold')

    def init():
        scat.set_offsets(np.empty((0, 2)))
        text.set_text("")
        return scat, text

    def update(frame):
        if frame >= len(all_points):
            return scat, text
        
        x, y, c_idx, p_idx = all_points[frame]
        
        # 更新红点位置
        scat.set_offsets([[x, y]])
        
        # 判断是外轮廓还是内孔洞
        # hierarchy[0][i] = [Next, Previous, First_Child, Parent]
        # 如果 Parent != -1，说明是内孔洞 (Inner Hole)
        is_inner = hierarchy[0][c_idx][3] != -1
        c_type = "Inner Hole" if is_inner else "Outer Boundary"
        
        text.set_text(f"Contour ID: {c_idx} ({c_type})\nPoint Index: {p_idx}\nPos: ({x}, {y})")
        
        return scat, text

    # 创建动画
    # interval: 帧间隔(ms)
    ani = FuncAnimation(fig, update, frames=len(all_points), init_func=init, blit=True, interval=20)
    
    print(f"正在生成动画 (共 {len(all_points)} 帧)... 请稍候")
    # 保存为 GIF
    writer = PillowWriter(fps=30)
    ani.save(save_path, writer=writer)
    print(f"动画已保存至: {save_path}")
    plt.close()

# --- 运行脚本 ---
if __name__ == "__main__":
    # 生成测试用例
    rope_mask = create_demo_crossing_rope_mask()
    
    # 1. 创建复杂的测试 Mask
    demo_mask_path = "./debug/test_pcdVLM_deformable_cloth/test_OBJ-cable_ID-01/FRAME01_mask.jpg"
    # demo_mask_path = "./debug/test_pcdVLM_deformable_cloth/test_OBJ-cable_ID-01/FRAME45_mask.jpg"
    # demo_mask_path = "./debug/test_pcdVLM_deformable_cloth/test_OBJ-cable_ID-01/FRAME65_mask.jpg"
    rope_mask = cv2.cvtColor(cv2.imread(demo_mask_path), cv2.COLOR_BGR2GRAY) 
    
    from rope_uncross import RopeSkeletonExtractor
    extractor = RopeSkeletonExtractor(blur_ksize=11, prune_threshold=15)
    rope_mask = extractor._preprocess_mask(rope_mask)  #  smooth mask
    
    
    # 运行可视化
    visualize_contour_traversal(rope_mask)
    
    # 如果你在 Jupyter/Colab 中，可以直接显示生成的 gif
    # from IPython.display import Image
    # display(Image(filename="contour_flow.gif"))