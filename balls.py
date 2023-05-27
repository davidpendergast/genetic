import math
import random

import pygame

import rainbowize
import utils.qutils as qutils


TEXTURES = {}
FIBS = {}


def fibonacci_sphere(samples=64):
    if samples not in FIBS:

        points = []
        phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append((x, y, z))

        FIBS[samples] = points

    return FIBS[samples]


IMG_CACHE = {}

class Ball:

    def __init__(self, radius, pos=(0, 0), res=8, texture_id='default', orient=None):
        self.radius = radius
        self.pos = pygame.Vector3(pos[0], pos[1], 0)
        self.texture_id = texture_id
        self.orient = orient or qutils.Qk
        self.outline = ("white", 1)
        self.res = res
        self.mode = 'fib'

        self._cache = {}

    def get_normalized_orient(self):
        if self.res <= 0:
            as_vec, angle = qutils.qdecomp(self.orient)
            return pygame.Vector3(as_vec), angle
        elif self.mode == 'fib':
            as_vec, angle = qutils.qdecomp(self.orient)
            as_vec = pygame.Vector3(as_vec)
            fib_vecs = fibonacci_sphere(samples=self.res)
            new_angle = round((angle % (2 * math.pi)) / (2 * math.pi / self.res)) * (2 * math.pi / self.res)
            to_use = min(fib_vecs, key=lambda v: as_vec.distance_squared_to(v))
            # print(f"{as_vec=} {angle=:.2f} --> {to_use=} {new_angle=:.2f}")
            return pygame.Vector3(to_use), new_angle
        else:
            as_vec, angle = qutils.qdecomp(self.orient)
            as_vec = pygame.Vector3(as_vec)
            (r, theta, phi) = as_vec.as_spherical()
            theta = round(self.res * theta / 180) * 180 // self.res
            phi = round(self.res * phi / 360) * 360 // self.res
            angle = round(self.res * angle / (2 * math.pi)) * (2 * math.pi) // self.res
            norm_vec = pygame.Vector3()
            norm_vec.from_spherical((1, theta, phi))
            return norm_vec, angle

    def get_img(self):
        vec, angle = self.get_normalized_orient()
        key = (tuple(vec), angle, self.outline, self.texture_id, self.radius)
        if key not in IMG_CACHE:
            # print(f"new key: {key}")
            out_color, out_width = self.outline
            res = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            orient = qutils.get_rot_quat(vec, angle)

            surf_center = pygame.Vector2(res.get_width() / 2, res.get_height() / 2)
            texture = TEXTURES[self.texture_id]

            view_dir = pygame.Vector3(0, 0, -1)
            for y in range(res.get_height()):
                for x in range(res.get_width()):
                    if surf_center.distance_to((x + 0.5, y + 0.5)) < self.radius - 0.333:
                        R = self.radius
                        V, X = view_dir, pygame.Vector3(x + 0.5 - surf_center.x, y + 0.5 - surf_center.y, self.radius)
                        X2, V2 = X.magnitude_squared(), V.magnitude_squared()
                        t = (-2 * V.dot(X) - math.sqrt((2 * V.dot(X))**2 - 4 * V2 * (X2 - R*R))) / (2 * V2)

                        pt_on_sphere = X + t * V
                        pt_on_sphere = pygame.Vector3(qutils.rotate_pt_about_quat(pt_on_sphere, orient))
                        (r, theta, phi) = pt_on_sphere.as_spherical()
                        tx_coords = (int(phi % 360 * texture.get_width() / 360),
                                     int(theta / 180 * texture.get_height()))

                        # if (x, y) == (res.get_width() // 2, res.get_height() // 2):
                        #     print(f"{pt_on_sphere=}, {(r, theta, phi)=}, {tx_coords=}")
                        # print(f"{(x,y)=}, {pt_on_sphere}, {r=}, {theta=}, {phi=}, {tx_coords=}")
                        base_color = texture.get_at(tx_coords)
                        shaded_color = pygame.Color(base_color).lerp("black", t / R * 0.666)
                        res.set_at((x, y), shaded_color)

            if out_width > 0:
                pygame.draw.circle(res, out_color, surf_center, self.radius, width=out_width)

            IMG_CACHE[key] = res

        return IMG_CACHE[key]

    def move(self, vec3, rotate=True):
        self.pos += vec3

        if rotate:
            rotation = vec3.magnitude() / (self.radius * 2 * math.pi) * math.pi * 2
            normal = vec3.cross((0, 0, 1))
            q = qutils.get_rot_quat(normal, rotation)
            self.orient = qutils.qmult(self.orient, q)

    def render(self, dest: pygame.Surface, xy):
        surf = self.get_img()
        dest.blit(surf, (xy[0] - surf.get_width() / 2, xy[1] - surf.get_height() / 2))


if __name__ == "__main__":
    pygame.init()
    SIZE = (256, 256)
    screen = rainbowize.make_fancy_scaled_display(SIZE, scale_factor=3, extra_flags=pygame.RESIZABLE)
    clock = pygame.Clock()

    TEXTURES['default'] = pygame.image.load("assets/rainbow_skulls.png")

    grid_res = 8

    balls = []
    for x in range(grid_res):
        for y in range(grid_res):
            b = Ball(8)
            b.pos = pygame.Vector3((x + 0.5) * SIZE[0] // grid_res, (y + 0.5) * SIZE[1] // grid_res, 0)
            rot_axis = pygame.Vector2(1, 0).rotate(x * 360 / grid_res)
            rot_amt = y * 180 / grid_res
            q = qutils.get_rot_quat((rot_axis.x, rot_axis.y, 0), rot_amt / 180 * math.pi)
            b.orient = qutils.qmult(b.orient, q)
            b.res = 0
            balls.append(b)

    movable_balls = [
        Ball(16, pos=(SIZE[0] / 2, SIZE[1] / 2), res=0),
        Ball(16, pos=(SIZE[0] / 2 + 48, SIZE[1] / 2), res=128)
    ]

    speed = 150

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()

        dx = keys[pygame.K_d] - keys[pygame.K_a]
        dy = keys[pygame.K_s] - keys[pygame.K_w]
        if dx != 0 or dy != 0:
            move = pygame.Vector3(dx, dy, 0)
            move.scale_to_length(speed / 60)
            for b in movable_balls:
                b.move(move)

        if keys[pygame.K_r]:
            q = qutils.get_rot_quat((0, 0, 1), 2 * math.pi / 60)
            for b in movable_balls:
                b.orient = qutils.qmult(b.orient, q)

        screen.fill("darkgray")

        for b in balls:
            b.render(screen, (b.pos.x, b.pos.y))

        for b in movable_balls:
            b.render(screen, (b.pos.x, b.pos.y))

        pygame.display.flip()
        clock.tick(60)


