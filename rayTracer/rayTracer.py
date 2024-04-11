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
        if isinstance(other, Color):
            # Element-wise multiplication (Hadamard product) with another Color
            return Color(*(self.color * other.color))
        elif isinstance(other, (float, int)):
            # Scalar multiplication as before
            return Color(*(self.color * other))
        else:
            raise NotImplementedError("Unsupported type for multiplication with Color.")

    def __rmul__(self, other):
        # This ensures that scalar * Color and Color * scalar both work
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

def trace(ray, objects, light, depth=0, max_depth=3):
    closest_dist = np.inf
    hit_obj = None
    hit_point = None
    normal = None
    
    for obj in objects:
        dist = obj.intersect(ray)
        if dist and 0 < dist < closest_dist:
            closest_dist = dist
            hit_obj = obj

    if hit_obj is not None:
        hit_point = ray.origin + ray.direction * closest_dist
        normal = (hit_point - hit_obj.center) / np.linalg.norm(hit_point - hit_obj.center)
        view_dir = -ray.direction
        color = hit_obj.material.shade(ray, hit_point, normal, view_dir, light, objects, depth, max_depth)
        return color, hit_obj
    

    return Color(0, 0, 0), None  # No intersection



class Material:
    def __init__(self, type, diffuse_color, specular_color=None, exponent=None):
        self.type = type
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color if specular_color else Color(0, 0, 0)
        self.exponent = exponent if exponent else 1  # Default exponent to 1 for safety

    def shade(self, ray, hit_point, normal, view_dir, light, objects, depth=0, max_depth=3):
        color = Color(0, 0, 0)  # Initialize color

        # Lambertian shading
        if self.type == "Lambertian":
            light_dir = light.position - hit_point
            light_dir /= np.linalg.norm(light_dir)
            diffuse = max(np.dot(normal, light_dir), 0) * self.diffuse_color
            color += diffuse
        
        # Phong shading
        elif self.type == "Phong":
            light_dir = light.position - hit_point
            light_dir /= np.linalg.norm(light_dir)
            diffuse = max(np.dot(normal, light_dir), 0) * self.diffuse_color
            color += diffuse

            # Specular component
            reflect_dir = 2 * normal * np.dot(normal, light_dir) - light_dir
            spec_angle = max(np.dot(view_dir, reflect_dir), 0)
            specular = (self.specular_color * np.power(spec_angle, self.exponent)) if spec_angle > 0 else Color(0, 0, 0)
            color += specular

        # Additional reflection logic here, if applicable

        return color


        
class Light:
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity

def render(scene, camera_settings, img_size):
    img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    
    # Unpack camera settings
    eye_pos = camera_settings['view_point']
    forward = camera_settings['view_dir'] / np.linalg.norm(camera_settings['view_dir'])
    right = np.cross(forward, camera_settings['view_up'])
    up = np.cross(right, forward)
    aspect_ratio = img_size[0] / img_size[1]
    fov_height = camera_settings['view_height']
    fov_width = camera_settings['view_width']
    proj_distance = camera_settings['proj_distance']

    # Iterate over each pixel in the image
    for y in range(img_size[1]):
        for x in range(img_size[0]):
            # Default background color
            pixel_color = Color(0, 0, 0)
            
            # Calculate the direction of the primary ray
            px = (2 * (x / img_size[0]) - 1) * aspect_ratio * fov_width
            py = (1 - 2 * (y / img_size[1])) * fov_height
            ray_dir = forward * proj_distance + px * right + py * up
            ray_dir /= np.linalg.norm(ray_dir)
            
            ray = Ray(eye_pos, ray_dir)
            closest_dist = np.inf
            hit_obj = None

            # Find the closest intersection
            for obj in scene['objects']:
                dist = obj.intersect(ray)
                if dist and 0 < dist < closest_dist:
                    closest_dist = dist
                    hit_obj = obj

            # If there is an intersection, calculate the color at the intersection point
            if hit_obj is not None:
                hit_point = ray.origin + ray.direction * closest_dist
                normal = (hit_point - hit_obj.center) / np.linalg.norm(hit_point - hit_obj.center)
                view_dir = -ray.direction

                # Check for shadow
                is_shadowed = False
                to_light = scene['light'].position - hit_point
                shadow_ray = Ray(hit_point + normal * 0.001, to_light / np.linalg.norm(to_light))
                for obj in scene['objects']:
                    if obj is not hit_obj and obj.intersect(shadow_ray):
                        is_shadowed = True
                        break

                if not is_shadowed:
                    pixel_color = hit_obj.material.shade(ray, hit_point, normal, view_dir, scene['light'], scene['objects'])
                else:
                    # Optionally handle ambient light in shadows
                    pixel_color = Color(0.1, 0.1, 0.1)  # Dim color for shadowed areas

            img[y, x] = pixel_color.toUINT8()

    return img

def parse_shader(shader_elem):
    type = shader_elem.get('type')
    diffuse_color = Color.from_decimals(shader_elem.find('diffuseColor').text)
    specular_color = None
    exponent = None
    
    if type == "Phong":
        specular_color = Color.from_decimals(shader_elem.find('specularColor').text)
        exponent = float(shader_elem.find('exponent').text)
    
    return Material(type, diffuse_color, specular_color, exponent)


def parse_camera(camera_elem):
    view_point = np.array(list(map(float, camera_elem.find('viewPoint').text.split())))
    view_dir = np.array(list(map(float, camera_elem.find('viewDir').text.split())))
    proj_normal = np.array(list(map(float, camera_elem.find('projNormal').text.split())))
    view_up = np.array(list(map(float, camera_elem.find('viewUp').text.split())))
    view_width = float(camera_elem.find('viewWidth').text)
    view_height = float(camera_elem.find('viewHeight').text)
    
    proj_distance_elem = camera_elem.find('projDistance')
    proj_distance = float(proj_distance_elem.text) if proj_distance_elem is not None else 2.0  # Default if not provided
    

    return {
        'view_point': view_point,
        'view_dir': view_dir,
        'view_up': view_up,
        'proj_normal': proj_normal,
        'proj_distance': proj_distance,
        'view_width': view_width,
        'view_height': view_height
    }


def main(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    camera_settings = parse_camera(root.find('camera'))

    shaders = {shader.get('name'): parse_shader(shader) for shader in root.findall('shader')}


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

