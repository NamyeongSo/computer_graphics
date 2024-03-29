import glfw
from OpenGL.GL import *
import numpy as np


## render 조건 설정. Vector이미지이므로 Vertex(정점)을 이용해서 이어줌.
def render(T):
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    # draw cooridnate
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([1.,0.]))
    glColor3ub(0, 255, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([0.,1.]))
    glEnd()
    # draw triangle
    glBegin(GL_TRIANGLES)
    glColor3ub(255, 255, 255)
    glVertex2fv( (T @ np.array([.0,.5,1.]))[:-1] )
    glVertex2fv( (T @ np.array([.0,.0,1.]))[:-1] )
    glVertex2fv( (T @ np.array([.5,.0,1.]))[:-1] )
    glEnd()




def main():
    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(640, 640, "Hello World", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    glfw.swap_interval(1)
    # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # Poll events
        glfw.poll_events()
        t = glfw.get_time()

        s = np.sin(t)
        # T = np.array([[s, 0.],
        #             [0., s*.5]])
        
        #시간이 어짜피 쭈욱 증가하니까 그냥 이놈을 theta로 잡는거임
        th = t
        
        #rotation
        # T = np.array([[np.cos(th), -np.sin(th)],
        #               [np.sin(th), np.cos(th)]])

        #reflection
        # T = np.array([[-1., 0.],
        #               [0. , 1.]])
        
        #shear
        # a = np.sin(t)
        # T = np.array([[1. , a],
        #               [0.,1.]])

        #idendtity matrix
        #T = np.identity(2)

        
        #compare result of these two lines
        th=t
        R=np.array([[np.cos(th),-np.sin(th),0.],[np.sin(th),np.cos(th),0.],[0.,0.,1.]])
        render(R)
        
        # Swap front and back buffers
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
