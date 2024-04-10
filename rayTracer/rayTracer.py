#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

class Color:
    def __init__(self, R, G, B):
        self.color = np.array([R, G, B], dtype=np.float64)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Color(*(self.color * other))
        else:
            raise NotImplementedError("Unsupported type for multiplication with Color.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, Color):
            return Color(*(self.color + other.color))
        else:
            raise NotImplementedError("Unsupported type for addition with Color.")

    def gammaCorrect(self, gamma):
        inverseGamma = 1.0 / gamma
        return Color(*np.power(self.color, inverseGamma))

    def toUINT8(self):
        return (np.clip(self.color, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def from_hex(hex_str):
        h = hex_str.lstrip('#')
        return Color(*(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)))

    def from_decimals(rgb_string):
        """Create Color from a string with decimal RGB values separated by spaces."""
        R, G, B = map(float, rgb_string.split())
        return Color(R, G, B)

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c
        if discriminant > 0:
            dist = (-b - np.sqrt(discriminant)) / (2.0*a)
            if dist > 0:
                return dist
        return None

class Material:
    def __init__(self, diffuse_color):
        self.diffuse_color = diffuse_color

    def shade(self, hit_point, normal, light):
        light_dir = light.position - hit_point
        light_dir /= np.linalg.norm(light_dir)
        diffuse = max(np.dot(normal, light_dir), 0) * self.diffuse_color
        return diffuse
        
class Light:
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity

def render(scene, camera_settings, img_size):
    img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    
    eye_pos = camera_settings['view_point']
    forward = camera_settings['view_dir'] / np.linalg.norm(camera_settings['view_dir'])
    right = np.cross(forward, camera_settings['view_up'])
    up = np.cross(right, forward)
    aspect_ratio = img_size[0] / img_size[1]

    for y in range(img_size[1]):
        for x in range(img_size[0]):
            # Map pixel coordinate to [-1, 1] range
            px = (2 * (x / img_size[0]) - 1) * aspect_ratio * camera_settings['view_width']
            py = (1 - 2 * (y / img_size[1])) * camera_settings['view_height']
            ray_dir = forward + px * right + py * up
            ray_dir /= np.linalg.norm(ray_dir)  # Normalize direction
            
            ray = Ray(eye_pos, ray_dir)
            closest_dist = np.inf
            pixel_color = Color(0, 0, 0)

            for obj in scene['objects']:
                dist = obj.intersect(ray)
                if dist and dist < closest_dist:
                    closest_dist = dist
                    hit_point = ray.origin + ray.direction * dist
                    normal = (hit_point - obj.center) / np.linalg.norm(hit_point - obj.center)
                    pixel_color = obj.material.shade(hit_point, normal, scene['light'])

            img[y, x] = pixel_color.toUINT8()

    return img

def parse_camera(camera_elem):
    view_point = np.array(list(map(float, camera_elem.find('viewPoint').text.split())))
    view_dir = np.array(list(map(float, camera_elem.find('viewDir').text.split())))
    proj_normal = np.array(list(map(float, camera_elem.find('projNormal').text.split())))
    view_up = np.array(list(map(float, camera_elem.find('viewUp').text.split())))
    view_width = float(camera_elem.find('viewWidth').text)
    view_height = float(camera_elem.find('viewHeight').text)
    
    return {
        'view_point': view_point,
        'view_dir': view_dir,
        'proj_normal': proj_normal,
        'view_up': view_up,
        'view_width': view_width,
        'view_height': view_height
    }


def main(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    camera_settings = parse_camera(root.find('camera'))

    shaders = {shader.get('name'): Material(Color.from_decimals(shader.find('diffuseColor').text)) 
               for shader in root.findall('shader')}


    scene = {
        'objects': [],
        'light': Light(np.array([0, 5, 0]), Color(1, 1, 1))  # Assuming a single light for simplicity
    }

    for surface in root.findall('surface'):
        if surface.get('type') == 'Sphere':
            center = np.fromstring(surface.find('center').text, sep=' ')
            radius = float(surface.find('radius').text)
            shader_ref = surface.find('shader').get('ref')
            material = shaders[shader_ref]
            scene['objects'].append(Sphere(center, radius, material))

    img_size = np.array(list(map(int, root.find('image').text.split())))
    img = render(scene, camera_settings, img_size)
    Image.fromarray(img, 'RGB').save(xml_file + '.png')


if __name__ == "__main__":
    main(sys.argv[1])
