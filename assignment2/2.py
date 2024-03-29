import numpy as np
import glfw
from OpenGL.GL import *

def update_transformation():
    time = glfw.get_time()
    angle = time
    radius = 0.5  
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    transformation_matrix = np.array([[np.cos(angle), -np.sin(angle), x],
                                      [np.sin(angle), np.cos(angle), y],
                                      [0, 0, 1]])
    return transformation_matrix

def render(T):
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    # Draw coordinate lines
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex2fv(np.array([0., 0.]))
    glVertex2fv(np.array([1., 0.]))
    glColor3ub(0, 255, 0)
    glVertex2fv(np.array([0., 0.]))
    glVertex2fv(np.array([0., 1.]))
    glEnd()
    # Draw orbiting triangle
    glBegin(GL_TRIANGLES)
    glColor3ub(255, 255, 255)
    glVertex2fv((T @ np.array([.0, .5, 1.]))[:-1])
    glVertex2fv((T @ np.array([.0, .0, 1.]))[:-1])
    glVertex2fv((T @ np.array([.5, .0, 1.]))[:-1])
    glEnd()

def main():   
    if not glfw.init():
        return
    title = "2022046435-2-2"
    window = glfw.create_window(480, 480, title, None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        T = update_transformation()
        render(T)
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
