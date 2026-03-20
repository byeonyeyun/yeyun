import cv2
import numpy as np
import random

# ===== 색 설정 (RGB -> BGR 변환 포함) =====
# 트리 + 글자 색 (RGB: 176,196,222)
tree_text_rgb = (176, 196, 222)
tree_text_bgr = (tree_text_rgb[2], tree_text_rgb[1], tree_text_rgb[0])  # (222,196,176)

# 별 + 장식 색-연한거  
deco_rgb = (255, 250, 212)
deco_bgr = (deco_rgb[2], deco_rgb[1], deco_rgb[0]) 

# 별 + 장식 색-진한거  
deco2_rgb = (255, 225, 43)
deco2_bgr = (deco2_rgb[2], deco2_rgb[1], deco2_rgb[0])  

# ===== 1. 캔버스(배경) 만들기 – 그라데이션 밤 하늘 =====
img_h, img_w = 600, 500

top_color    = np.array([0, 0, 0],   dtype=np.uint8)   # 맨 위 색 (BGR)
bottom_color = np.array([111, 45, 31], dtype=np.uint8)  # 맨 아래 색 (BGR)

# 세로 방향으로 줄마다 색을 조금씩 섞어줌
gradient = np.linspace(top_color, bottom_color, img_h)        # (img_h, 3)
gradient = gradient.reshape(img_h, 1, 3)                      # (img_h, 1, 3)
img = np.tile(gradient, (1, img_w, 1)).astype(np.uint8)       # (img_h, img_w, 3)


# ===== 2. 바닥(눈) 그리기 =====
snow_ground_color = (220, 220, 220)  # 밝은 회색/눈 색
cv2.rectangle(img, (0, img_h - 50), (img_w, img_h), snow_ground_color, -1)

# ===== 3. 크리스마스 트리 위치/크기 설정 =====
cx = img_w // 2          # 트리 중앙 x좌표
base_y = img_h - 80      # 트리 밑둥이 올라가는 위치

# ===== 4. 트리 줄기(나무 몸통) 좌표만 사용 (그리진 않음) =====
trunk_w = 40
trunk_h = 60
trunk_color = (50, 50, 150)  # 약간 갈색 느낌 (BGR)

trunk_x1 = cx - trunk_w // 2
trunk_y1 = base_y - trunk_h
trunk_x2 = cx + trunk_w // 2
trunk_y2 = base_y

# 줄기 사각형은 더 이상 그리지 않음
# cv2.rectangle(img, (trunk_x1, trunk_y1), (trunk_x2, trunk_y2), trunk_color, -1)

# ===== 5. 트리(삼각형 5단) – 크기 키운 버전 =====
tree_color = tree_text_bgr  # (222,196,176)

# (1) 맨 아래 큰 삼각형 (폭, 높이 모두 기존보다 키움)
tier1 = np.array([
    [cx,          trunk_y1 - 10 +5+30],      # 윗점
    [cx - 140,    trunk_y1 + 55 +5+30],      # 왼쪽 아래 (≈ base_y - 5)
    [cx + 140,    trunk_y1 + 55 +5+30],      # 오른쪽 아래
], dtype=np.int32)
cv2.fillConvexPoly(img, tier1, tree_color)

# (2) 두 번째 삼각형
tier2 = np.array([
    [cx,          trunk_y1 - 70 +5+30],
    [cx - 110,    trunk_y1 + 25 +5+30],
    [cx + 110,    trunk_y1 + 25 +5+30],
], dtype=np.int32)
cv2.fillConvexPoly(img, tier2, tree_color)

# (3) 세 번째 삼각형
tier3 = np.array([
    [cx,          trunk_y1 - 130+5+30],
    [cx - 85,     trunk_y1 - 10+5+30],
    [cx + 85,     trunk_y1 - 10+5+30],
], dtype=np.int32)
cv2.fillConvexPoly(img, tier3, tree_color)

# (4) 네 번째 삼각형
tier4 = np.array([
    [cx,          trunk_y1 - 190+30],
    [cx - 60,     trunk_y1 - 50+30],
    [cx + 60,     trunk_y1 - 50+30],
], dtype=np.int32)
cv2.fillConvexPoly(img, tier4, tree_color)

# (5) 가장 위 삼각형 (트리 높이 늘린 부분)
tier5 = np.array([
    [cx,          trunk_y1 - 250-10+30],
    [cx - 45,     trunk_y1 - 90-10+35],
    [cx + 45,     trunk_y1 - 90-10+35],
], dtype=np.int32)
cv2.fillConvexPoly(img, tier5, tree_color)

# ===== 6. 트리 꼭대기 별 =====
tree_top_y = trunk_y1 - 250 - 10 + 30
star_center = (cx, tree_top_y - 10)

# 큰 연한 별
star_color_outer = deco_bgr          # 연한 색
star_radius_outer = 20
star_radius_inner = 8

def draw_star(img, center, outer_radius, inner_radius, color):
    cx, cy = center
    pts = []
    for i in range(10):
        angle_deg = 36 * i
        angle = np.deg2rad(angle_deg - 90)
        r = outer_radius if i % 2 == 0 else inner_radius
        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))
        pts.append([x, y])

    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(img, pts, color)

# 1) 먼저 큰 연한 별 그리기
draw_star(img, star_center, star_radius_outer, star_radius_inner, star_color_outer)

# 2) 그 위에 조금 작고 진한 별 한 번 더 겹쳐서 그리기
star_color_inner = deco2_bgr         # 더 진한 색 사용
star_radius_outer_small = 14         # 바깥 반지름 더 작게
star_radius_inner_small = 5          # 안쪽 반지름 더 작게

draw_star(img, star_center, star_radius_outer_small, star_radius_inner_small, star_color_inner)

# ===== 7. 트리 장식(오너먼트) 공통 위치 =====
ornament_positions = [
    # 1층(맨 아래 삼각형 근처)
    (cx - 130, trunk_y1 + 85),
    (cx - 65,  trunk_y1 + 80),
    (cx,       trunk_y1 + 75),
    (cx + 60,  trunk_y1 + 85),
    (cx + 90,  trunk_y1 + 70),

    # 2층
    (cx - 85,  trunk_y1 + 35),
    (cx - 40,  trunk_y1 + 60),
    (cx + 30,  trunk_y1 + 45),
    (cx + 70,  trunk_y1 + 10),

    # 3층
    (cx - 60,  trunk_y1 + 5),
    (cx - 20,  trunk_y1),
    (cx + 20,  trunk_y1 + 20),
    (cx + 65,  trunk_y1 + 5),

    # 4층
    (cx - 35,  trunk_y1 - 45),
    (cx,       trunk_y1 - 60),
    (cx + 45,  trunk_y1 - 35),

    # 5층(윗부분)
    (cx + 3,   trunk_y1 - 170),
    (cx - 20,  trunk_y1 - 150),
    (cx + 10,  trunk_y1 - 110),
    (cx - 20,  trunk_y1 - 90),
    (cx + 25,  trunk_y1 - 75),
]

# 1) 연하고 큰 장식 먼저 (배경 느낌)
for (ox, oy) in ornament_positions:
    cv2.circle(img, (ox, oy), 2, deco_bgr, -1)

# 2) 그 위에 진하고 작은 장식 (하이라이트)
for (ox, oy) in ornament_positions:
    cv2.circle(img, (ox, oy), 1, deco2_bgr, -1)

# ===== 9. 눈/별 찍기 =====
# (1) 아주 작은 눈 
for _ in range(170):
    sx = random.randint(0, img_w - 1)
    sy = random.randint(0, img_h - 1)  
    cv2.circle(img, (sx, sy), 1, (255, 255, 255), -1)

# (2)  조금 더 큰 눈 (반지름 2~3)
for _ in range(90):
    sx = random.randint(0, img_w - 1)
    sy = random.randint(0, img_h - 1)  # 화면 아래 절반
    r = random.randint(2, 3)
    cv2.circle(img, (sx, sy), r, (255, 255, 255), -1)

# ===== 10. "Merry Christmas" 글자 – tree_text_bgr 색 사용 =====
text = "Merry Christmas"
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
font_scale = 1.0
thickness = 2
text_color = tree_text_bgr

(text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
tx = (img_w - text_w) // 2
ty = 100

cv2.putText(img, text, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)

# ===== 11. 결과 출력 =====
cv2.imshow("Christmas Tree", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
