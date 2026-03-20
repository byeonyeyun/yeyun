import cv2
import numpy as np

# 이미지 선택 & 로드
file_path = './images/mihu.jpg'
img_color = cv2.imread(file_path)

if img_color is None:
    print(f"오류: '{file_path}' 이미지를 로드할 수 없습니다.")
    exit()

#  이미지 전체 스케일 줄이기 
scale = 0.3   
img_color = cv2.resize(img_color, (0, 0), fx=scale, fy=scale)

# BGR -> Gray Scale
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

kernel_size = (5, 5)
blur_mean     = cv2.blur(img_gray, kernel_size)
blur_gaussian = cv2.GaussianBlur(img_gray, kernel_size, 0)
blur_median   = cv2.medianBlur(img_gray, 5)

gray_bgr     = cv2.cvtColor(img_gray,      cv2.COLOR_GRAY2BGR)
mean_bgr     = cv2.cvtColor(blur_mean,     cv2.COLOR_GRAY2BGR)
gaussian_bgr = cv2.cvtColor(blur_gaussian, cv2.COLOR_GRAY2BGR)
median_bgr   = cv2.cvtColor(blur_median,   cv2.COLOR_GRAY2BGR)

combined = cv2.hconcat([img_color, mean_bgr, gaussian_bgr, median_bgr])

h, w = img_color.shape[:2]
font = cv2.FONT_HERSHEY_SIMPLEX
scale_font = 0.7
thickness = 2
color = (128, 128, 128)

cv2.putText(combined, "Original",      (10, 30),         font, scale_font, color, thickness)
cv2.putText(combined, "Mean Blur",     (w + 10, 30),     font, scale_font, color, thickness)
cv2.putText(combined, "Gaussian Blur", (2 * w + 10, 30), font, scale_font, color, thickness)
cv2.putText(combined, "Median Blur",   (3 * w + 10, 30), font, scale_font, color, thickness)

cv2.imshow("Blur Comparison", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

