import cv2
import numpy as np

# 1. 전역 변수 설정
drawing = False          # 드래그 중인지 여부 처음에는 false로
mode = None              # 어떤 도형 그릴지 : 'circle' 또는 'line'
ix, iy = -1, -1          # 마우스를 누른 “시작점”의 좌표를 저장하는 변수 

# 500x500 크기의 검은색 배경 이미지 생성
img = np.zeros((500, 500, 3), np.uint8) # (높이,너비,컬러)을 가진 배열을 0으로 채움
# dtype=np.uint8 :  이미지에서 흔히 사용하는 0~255 범위의 정수 타입
# 2. 마우스 이벤트 콜백 함수 정의
def draw_shape(event, x, y, flags, param):
    # draw_shape :  마우스 이벤트가 발생할 때마다 자동으로 호출될 함수
    
    # event: 어떤 종류의 마우스 이벤트인지 (버튼 눌림/뗌/이동 등)
    # x, y: 이벤트가 발생한 마우스 포인터의 좌표
    # flags: Shift, Ctrl 등 키 조합이나 마우스 상태에 대한 추가 정보
    # param: setMouseCallback에서 넘긴 추가 인자(여기서는 사용 안 함)
    global ix, iy, drawing, mode, img
    # 함수 안에서 바깥에 정의된 전역 변수들을 수정하기 위해 global로 선언

    # 왼쪽 버튼 눌렀을 때: 원 그리기 모드
    if event == cv2.EVENT_LBUTTONDOWN: # BUTTONDOWN : 버튼 누르기
        drawing = True
        mode = 'circle'
        ix, iy = x, y

    # 오른쪽 버튼 눌렀을 때: 선 그리기 모드
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        mode = 'line'
        ix, iy = x, y

    # 마우스 이동 중일 때: 드래그 중이면 임시 화면에 미리 보기
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_temp = img.copy()

        # 원 미리 보기  
        if mode == 'circle':
            radius = int(((x - ix) ** 2 + (y - iy) ** 2) ** 0.5)
            cv2.circle(img_temp, (ix, iy), radius, (250, 206, 135), 2)
            # 중심: (ix, iy) (마우스를 처음 누른 지점)
            # 반지름: 위에서 계산한 radius
            # 색상: (255, 0, 0) → BGR 기준 → 파란색
            # 두께: 2 픽셀
        # 선 미리 보기  
        elif mode == 'line':
            cv2.line(img_temp, (ix, iy), (x, y), (0, 140, 255), 2)
            #시 작점: (ix, iy) (마우스를 처음 누른 지점)
            # 끝점: (x, y) (현재 마우스 위치)
            # 색상: (0, 0, 255) → BGR 기준 → 빨간색
            # 두께: 2 픽셀
        cv2.imshow('Image', img_temp)
        # 임시 이미지 img_temp를 'Image'라는 이름의 창에 보여줌
        # 드래그 하는 동안 계속 새로운 img_temp가 갱신되면서, 도형이 실시간으로 움직이는 것처럼 보임

    # 왼쪽 버튼을 놓았을 때: 원 최종 확정
    elif event == cv2.EVENT_LBUTTONUP and mode == 'circle':
        # BUTTONDUP : 버튼 떼기
        drawing = False # 드래그 종료, 더 이상 실시간 그리기 안 함
        radius = int(((x - ix) ** 2 + (y - iy) ** 2) ** 0.5)
        # radius : 드래그가 끝난 시점의 마우스 위치 (x,y)까지의 거리를 다시 계산해서 반지름을 구함
        cv2.circle(img, (ix, iy), radius, (255, 0, 0), 2)  # 파란색 원
        mode = None # 모드 초기화
        cv2.imshow('Image', img) # 원이 그려진 img를 창에 다시 표시

    # 오른쪽 버튼을 놓았을 때: 선 최종 확정
    elif event == cv2.EVENT_RBUTTONUP and mode == 'line':
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 2)   # 빨간색 선
        mode = None
        cv2.imshow('Image', img)


# 3. 메인 루프 설정
cv2.namedWindow('Image', cv2.WINDOW_GUI_NORMAL)
# 오른쪽 마우스 클릭시에 확장 GUI(메뉴/툴바/팝업)가 나와서 그냥 기본 GUI로만 창 만듬
# 'Image'라는 이름의 GUI 창을 미리 생성
cv2.setMouseCallback('Image', draw_shape)
# 'Image' 창에 대해 마우스 이벤트가 발생할 때마다 draw_shape 함수를 호출해달라고 OpenCV에 등록
# 이 줄이 있어야 클릭.드래그.이동같은 이벤트가 draw_shape 로 전달됨

# 4. 이미지 표시 및 대기
cv2.imshow('Image', img) #현재 img (검은색 배경만 있는 상태)를 'Image' 창에 처음으로 보여줌
cv2.waitKey(0) # 사용자가 아무 키나 누를 때까지 창이 계속 유지
cv2.destroyAllWindows() # OpenCV가 만든 모든 창을 닫음
