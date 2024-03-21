import glfw
from OpenGL.GL import *
import numpy as np

# 렌더 함수 정의
def render(T):
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    # 좌표 그리기
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([1.,0.]))
    glColor3ub(0, 255, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([0.,1.]))
    glEnd()
    # 삼각형 그리기
    glBegin(GL_TRIANGLES)
    glColor3ub(255, 255, 255)
    glVertex2fv( (T @ np.array([.0,.5,1.]))[:-1] )
    glVertex2fv( (T @ np.array([.0,.0,1.]))[:-1] )
    glVertex2fv( (T @ np.array([.5,.0,1.]))[:-1] )
    glEnd()

# 창 제목 및 크기 설정
window_title = "OpenGL 회전 예제"
window_width, window_height = 480, 480

# GLFW 초기화
if not glfw.init():
    raise RuntimeError("Could not initialize GLFW")

# 창 생성 및 OpenGL 컨텍스트 설정
window = glfw.create_window(window_width, window_height, window_title, None, None)
if not window:
    glfw.terminate()
    raise RuntimeError("Could not create window")

# 현재 창의 컨텍스트를 활성화
glfw.make_context_current(window)

# 사용자가 창을 닫을 때까지 반복
while not glfw.window_should_close(window):
    # 렌더링
    glClear(GL_COLOR_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # 회전 행렬 생성
    angle = glfw.get_time() * 50
    rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                                [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
                                [0, 0, 1]])
    render(rotation_matrix)

    # 전면과 후면 버퍼 교환
    glfw.swap_buffers(window)

    # 이벤트 처리
    glfw.poll_events()

# GLFW 종료
glfw.terminate()
