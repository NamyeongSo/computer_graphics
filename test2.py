import numpy as np
import glfw
from OpenGL.GL import *

# 전역 변수로 primitive type을 저장합니다.
primitive_type = GL_LINE_LOOP

def key_event(window, key, scancode, action, mods):
    global primitive_type
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_1:
            primitive_type = GL_POINTS
        elif key == glfw.KEY_2:
            primitive_type = GL_LINES
        elif key == glfw.KEY_3:
            primitive_type = GL_LINE_STRIP
        elif key == glfw.KEY_4:
            primitive_type = GL_LINE_LOOP
        elif key == glfw.KEY_5:
            primitive_type = GL_TRIANGLES
        elif key == glfw.KEY_6:
            primitive_type = GL_TRIANGLE_STRIP
        elif key == glfw.KEY_7:
            primitive_type = GL_TRIANGLE_FAN
        elif key == glfw.KEY_8:
            primitive_type = GL_QUADS
        elif key == glfw.KEY_9:
            primitive_type = GL_QUAD_STRIP
        elif key == glfw.KEY_0:
            primitive_type = GL_POLYGON

def main():
    if not glfw.init():
        return

    window = glfw.create_window(480, 480, "2017123456-2-1", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_event)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        render()

        glfw.swap_buffers(window)

    glfw.terminate()

def render():
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()

    # 정12각형을 그리기 위한 정점 계산
    num_sides = 12
    radius = 1
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    vertices = np.array([[np.cos(angle) * radius, np.sin(angle) * radius] for angle in angles], dtype=np.float32)

    glBegin(primitive_type)
    for vertex in vertices:
        glVertex2fv(vertex)
    glEnd()

if __name__ == "__main__":
    main()
 