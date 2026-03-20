# 이미지 상단 중앙에 cv2.FONT_HERSHEY_TRIPLEX 폰트로 빨간색 텍스트
# "OpenCV Text Practice"를 표시합니다.

# 500x500 하늘색 배경 위에 빨간색 텍스트 "OpenCV Text Practice" 그리기 예제
import cv2
import numpy as np

# 1. 500 x 500 크기의 하늘색 배경 이미지 생성 (BGR)
img_height, img_width = 500, 500

sky_blue = (255, 255, 0) # 하늘색(BGR) 예: (255, 255, 0)  
img_color = np.full((img_height, img_width, 3), sky_blue, dtype=np.uint8)
# np.full(...) : shape 모양을 가진 배열을 만들고 그 안을 sky_bule 값으로 전부 채워주는 NumPy 함수
# dtype=np.uint8 : OpenCV는 기본적으로 uint8 타입 이미지를 사용

# 2. 텍스트 설정
text = "OpenCV Text Practice"
font = cv2.FONT_HERSHEY_TRIPLEX
font_scale = 1.0 # 글자 크기 비율
thickness = 2 # 글자 두께
text_color = (0, 0, 255)  # 빨간색 (BGR)

# 3. 텍스트 크기 계산해서 '상단 중앙' 위치 구하기
(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
# cv2.getTextSize(...) : 문자열의 폰트,크기,두께를 입력해 글자박스 크기를 계산해 주는 함수
# (text_width, text_height) : 글자 영역의 폭, 높이(픽셀단위)
#  baseline : 글짜 아래쪽 베이스 라인 기준으로 추가 여유 공간이 몇 픽셀인지
#  가운데 중앙 정렬을 위해 계산함 


# x좌표 가로 중앙 정렬: (전체 폭 - 글자 폭) / 2
x = (img_width - text_width) // 2 
# y 좌표 세로 위치: 상단에서 조금 내려온 위치 (여기선 50 픽셀 정도 여유)
y = (img_width - text_height) // 2  # y는 글자 "밑줄 기준" 좌표 

# 4. 이미지에 텍스트 그리기
cv2.putText(
    img_color,          # 하늘색 배경
    text,               # 텍스트
    (x, y),             # 시작 좌표 (좌측 하단 기준)
    font,               # 폰트
    font_scale,         # 글자 크기 스케일 (글자 크기 배율)
    text_color,         # 색상 (B, G, R)
    thickness,          # 두께
    cv2.LINE_AA         # 안티에일리어싱(부드러운 글자)
)

# 5. 이미지 표시
cv2.imshow("Text Practice", img_color)

# 6. 키 입력 대기
cv2.waitKey(0)

# 7. 모든 창 닫기
cv2.destroyAllWindows()

# 8. 이미지 파일 저장
# cv2.imwrite("text_practice.jpg", img_color)
