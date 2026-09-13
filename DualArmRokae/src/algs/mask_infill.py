
import cv2
import numpy as np
import matplotlib.pyplot as plt

def complete_basket_mask(broken_mask):
    """
    启发式补全破碎的篮子Mask
    Input: broken_mask (二值图像, 255为前景)
    Output: completed_mask, recovered_corners
    """
    h, w = broken_mask.shape
    
    # --- 1. 形态学闭运算 (Morphological Closing) ---
    # 连接断裂的边缘。Kernel大小取决于断裂的程度，通常取 5x5 到 15x15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed_mask = cv2.morphologyEx(broken_mask, cv2.MORPH_CLOSE, kernel)
    
    # --- 2. 寻找最大轮廓 ---
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: No contour found.")
        return broken_mask, None
    
    # 假设最大的轮廓是篮子（忽略噪点）
    c = max(contours, key=cv2.contourArea)
    
    # --- 3. 计算凸包 (Convex Hull) ---
    # 这一步能填补所有“凹进去”的缺口（比如物体遮挡导致边缘凹陷）
    hull = cv2.convexHull(c)
    
    # --- 4. 几何形状拟合 (根据需求二选一) ---
    
    # 选项 A: 最小外接矩形 (强约束)
    # 适用于：你非常确定篮子投影就是矩形，且想得到最规则的结果
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # 选项 B: 多边形逼近 (软约束)
    # 适用于：篮子可能有透视畸变（梯形），不想强制成矩形
    # epsilon = 0.02 * perimeter
    # box = cv2.approxPolyDP(hull, epsilon, True)
    # if len(box) != 4: 
    #     # 如果逼近出来不是4个点，回退到选项A
    #     rect = cv2.minAreaRect(hull)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    
    # --- 5. 绘制结果 ---
    completed_mask = np.zeros_like(broken_mask)
    # 填充修复后的四边形
    cv2.fillPoly(completed_mask, [box], 255)
    
    return completed_mask, box

# --- 模拟测试 ---

# 1. 创建一个“破碎”的Mask
H, W = 400, 400
true_mask = np.zeros((H, W), dtype=np.uint8)
true_pts = np.array([[100, 100], [300, 100], [300, 300], [100, 300]]) # 完美矩形
cv2.fillPoly(true_mask, [true_pts], 255)

# 模拟遮挡：在矩形中间挖几个大洞，切断一条边
broken_mask = true_mask.copy()
# 挖洞 (模拟内部物体)
cv2.circle(broken_mask, (200, 200), 60, 0, -1) 
# 切断边缘 (模拟边缘遮挡)
cv2.circle(broken_mask, (100, 200), 20, 0, -1) 
cv2.circle(broken_mask, (300, 150), 30, 0, -1)

# 2. 运行补全算法
recovered_mask, corners = complete_basket_mask(broken_mask)

# 3. 可视化对比
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Broken Mask")
plt.imshow(broken_mask, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Recovered Mask (MinAreaRect)")
plt.imshow(recovered_mask, cmap='gray')

# 显示边缘叠加
vis_img = cv2.cvtColor(broken_mask, cv2.COLOR_GRAY2BGR)
cv2.drawContours(vis_img, [corners], -1, (0, 0, 255), 2) # 红色为修复后的框
plt.subplot(1, 3, 3)
plt.title("Overlay Result")
plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))

plt.show()

# 打印修复后的四个角点
print("Recovered Corners:\n", corners)