import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Global variable to store primitive type
primitive_type = GL_POINTS

# Function to draw a regular 12-sided polygon (dodecagon)
def draw_dodecagon():
    sides = 12
    angle = np.linspace(0, 2*np.pi, sides, endpoint=False)
    vertices = np.column_stack((np.cos(angle), np.sin(angle)))
    
    glBegin(primitive_type)
    for vertex in vertices:
        glVertex2fv(vertex)
    glEnd()

# Function to handle keyboard input
def key_callback(window, key, scancode, action, mods):
    global primitive_type
    if action == glfw.PRESS:
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

# Function to render the scene
def render():
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    glColor3ub(255, 255, 255)
    draw_dodecagon()
    glFlush()

# Main function
def main():
    student_id = "2017123456"  # Replace with your student ID
    assignment_num = "2"
    prob_num = "1"
    window_title = student_id + "-" + assignment_num + "-" + prob_num

    # Initialize GLFW
    if not glfw.init():
        return
    
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(480, 480, window_title, None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)

    # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # Render here, e.g. using PyOpenGL
        render()

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
