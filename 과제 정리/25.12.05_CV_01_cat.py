# 2. 맘에 드는 이미지를 출력후 이미지를 소개하는 소개글을 화면 하단에 출력하는 예제를 작성하시오
# opencv의 경우 한글을 출력못함
# 본인이 소개글을 쓰고 => 제미나이를 이용해 번역하기
# Pilow 모듈 -> 한글 출력이 가능함
import cv2
import numpy as np
import os

# 1. 사용할 이미지 파일 경로
file_path = './images/0163.jpg'

# 2. 파일 존재 여부 확인
if not os.path.exists(file_path):
    print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# 3. 이미지 읽기
img = cv2.imread(file_path)
if img is None:
    print("오류: 이미지를 읽어올 수 없습니다.")
    exit()

# ================================
# (추가) 출력용으로 이미지 크기 축소
# ================================
orig_h, orig_w = img.shape[:2]

display_width = 800                      # 원하는 출력 가로 크기 (원하면 숫자 바꿔도 됨)
ratio = display_width / orig_w           # 비율 계산
display_height = int(orig_h * ratio)     # 세로는 비율에 맞게 조정

img = cv2.resize(img, (display_width, display_height))  # 이미지 리사이즈
img_height, img_width = img.shape[:2]                   # 이후는 축소된 크기 기준으로 사용

# 4. 소개글 내용 설정
text = (
    "The cat sleeping in the picture is Mihu. "
    "Mihu is a male born in 2015 and his favorite is Churu"
)

font = cv2.FONT_HERSHEY_TRIPLEX
font_scale = 0.45        # 글자 크기
thickness = 1           # 글자 두께
text_color = (255, 255, 255)  # 흰색 (BGR)

# 5. 텍스트 크기 계산 (가로/세로 폭)
(text_width, text_height), baseline = cv2.getTextSize(
    text, font, font_scale, thickness
)

# 가로 중앙 정렬: (전체 폭 - 글자 폭) / 2
x = (img_width - text_width) // 2
# 세로는 하단에서 10픽셀 위에 baseline 오도록
y = img_height - 10

# 6. 배경 박스 (반투명) – 선택사항
box_top_left  = (x - 10, y - text_height - 10)
box_bottom_right = (x + text_width + 10, y + baseline + 5)

overlay = img.copy()
cv2.rectangle(
    overlay,
    box_top_left,
    box_bottom_right,
    (0, 0, 0),    # 검정색 박스
    -1            # 채우기
)
alpha = 0.5
img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# 7. 텍스트 그리기
cv2.putText(
    img,
    text,
    (x, y),
    font,
    font_scale,
    text_color,
    thickness,
    cv2.LINE_AA
)

# 8. 결과 보기
cv2.imshow("Mihu Info", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

