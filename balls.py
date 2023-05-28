import math
import random

import pygame

import rainbowize
import utils.qutils as qutils
import utils.profiling as profiling


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

def quantize(val, max_val, n, and_round=False, wrap=True):
    if wrap:
        quant = round((val % max_val) / (max_val / n)) * (max_val / n)
        return round(quant) if and_round else quant
    else:
        raise NotImplementedError()

class Ball:

    def __init__(self, radius, pos=(0, 0), res=8, texture_id='default', mode='fib', orient=None):
        self.radius = radius
        self.pos = pygame.Vector3(pos[0], pos[1], 0)
        self.texture_id = texture_id
        self.orient = orient or qutils.get_rot_quat((0, 0, 1), 0)
        self.outline = ("white", 1)
        self.res = res
        self.mode = mode

    def get_normalized_orient(self):
        orient = self.orient if self.orient[1] >= 0 else qutils.qscale(self.orient, 1)
        as_vec, angle_rads = qutils.qdecomp(orient)
        as_vec = pygame.Vector3(as_vec)
        angle = angle_rads * 180 / math.pi
        if self.res <= 0:
            return as_vec, angle
        elif self.mode == 'fib':
            fib_vecs = fibonacci_sphere(samples=self.res)
            new_angle = round((angle % 360) / (360 / self.res)) * (360 / self.res)
            to_use = min(fib_vecs, key=lambda v: as_vec.distance_squared_to(v))
            return pygame.Vector3(to_use), new_angle
        else:
            (r, theta, phi) = as_vec.as_spherical()
            theta = round(self.res * theta / 180) * 180 // self.res
            phi = round(self.res * phi / 360) * 360 // self.res
            angle = round((angle % 360) / (360 / self.res)) * (360 / self.res)
            norm_vec = pygame.Vector3()
            norm_vec.from_spherical((1, theta, phi))
            return norm_vec, angle

    def get_img(self):
        if self.mode == 'eul':
            x, y, z = qutils.to_euler2(self.orient)
            quant_x = quantize(x * 180 / math.pi, 360, int(math.sqrt(self.res)), and_round=True)
            quant_y = quantize(y * 180 / math.pi, 360, int(math.sqrt(self.res)), and_round=True)
            rot_z = -z * 180 / math.pi

            key = (quant_x, quant_y, self.texture_id, self.radius)
            if key not in IMG_CACHE:
                orient = qutils.from_euler2(quant_x / 180 * math.pi, quant_y / 180 * math.pi, 0)
                # print(f"({self.orient[0]:.1f}, {self.orient[1]:.1f}, {self.orient[2]:.1f}, {self.orient[3]:.1f})=orient, "
                #       f"({x:.1f}, {y:.1f}, {z:.1f})=xyz, "
                #       f"({quant_x:.1f}, {quant_y:.1f}, {rot_z:.1f})=(quant_x, quant_y, rot_z) -->"
                #       f"({orient[0]:.1f}, {orient[1]:.1f}, {orient[2]:.1f}, {orient[3]:.1f})")
                IMG_CACHE[key] = self._render(orient, ('white', 0))
            img = IMG_CACHE[key]

            img_rot = pygame.transform.rotate(img, rot_z)
            out_color, out_width = self.outline
            if out_width > 0:
                pygame.draw.circle(img_rot, out_color, img_rot.get_rect().center, self.radius, width=out_width)

            return img_rot

        else:
            vec, angle = self.get_normalized_orient()
            key = (tuple(round(x, 3) for x in vec), round(angle, 3), self.outline, self.texture_id, self.radius)
            if key not in IMG_CACHE:
                orient = qutils.get_rot_quat(vec, angle * math.pi / 180)
                img = self._render(orient, self.outline)

                IMG_CACHE[key] = img
            return IMG_CACHE[key]

    def _render(self, orient, outline):
        out_color, out_width = outline
        res = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)

        surf_center = pygame.Vector2(res.get_width() / 2, res.get_height() / 2)
        texture = TEXTURES[self.texture_id]

        view_dir = pygame.Vector3(0, 0, -1)
        for y in range(res.get_height()):
            for x in range(res.get_width()):
                if surf_center.distance_to((x + 0.5, y + 0.5)) < self.radius - 0.333:
                    R = self.radius
                    V, X = view_dir, pygame.Vector3(x + 0.5 - surf_center.x, y + 0.5 - surf_center.y, self.radius)
                    X2, V2 = X.magnitude_squared(), V.magnitude_squared()
                    t = (-2 * V.dot(X) - math.sqrt((2 * V.dot(X)) ** 2 - 4 * V2 * (X2 - R * R))) / (2 * V2)

                    pt_on_sphere = X + t * V
                    pt_on_sphere = pygame.Vector3(qutils.rotate_pt_about_quat(pt_on_sphere, orient))
                    (r, theta, phi) = pt_on_sphere.as_spherical()
                    tx_coords = (int(phi % 360 * texture.get_width() / 360),
                                 int(theta / 180 * texture.get_height()))

                    try:
                        base_color = texture.get_at(tx_coords)
                        shaded_color = pygame.Color(base_color).lerp("black", t / R * 0.666)
                        res.set_at((x, y), shaded_color)
                    except IndexError:
                        pass

        if out_width > 0:
            pygame.draw.circle(res, out_color, surf_center, self.radius, width=out_width)

        return res

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
    dt = 0

    TEXTURES['default'] = pygame.image.load("assets/rainbow_skulls.png")

    grid_res = 8

    balls = []
    for x in range(grid_res):
        for y in range(grid_res):
            b = Ball(8, res=16)
            b.pos = pygame.Vector3((x + 0.5) * SIZE[0] // grid_res, (y + 0.5) * SIZE[1] // grid_res, 0)
            rot_axis = pygame.Vector2(1, 0).rotate(x * 360 / grid_res)
            rot_amt = y * 180 / grid_res
            q = qutils.get_rot_quat((rot_axis.x, rot_axis.y, 0), rot_amt / 180 * math.pi)
            b.orient = qutils.qmult(b.orient, q)
            b.res = 0
            balls.append(b)

    movable_balls = [
        # Ball(16, pos=(SIZE[0] / 2, SIZE[1] / 2), res=0),
        # Ball(16, pos=(SIZE[0] / 2 + 48, SIZE[1] / 2), res=128, mode='fib'),
        Ball(16, pos=(SIZE[0] / 2 + 48 * 2, SIZE[1] / 2), res=256, mode='eul')
    ]

    speed = 150

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_F1:
                    profiling.get_instance().toggle()
        keys = pygame.key.get_pressed()

        dx = keys[pygame.K_d] - keys[pygame.K_a]
        dy = keys[pygame.K_s] - keys[pygame.K_w]
        if dx != 0 or dy != 0:
            move = pygame.Vector3(dx, dy, 0)
            move.scale_to_length(speed * dt)
            for b in movable_balls:
                b.move(move)

        if keys[pygame.K_r]:
            q = qutils.get_rot_quat((0, 0, 1), 2 * math.pi / 60)
            for b in movable_balls:
                b.orient = qutils.qmult(b.orient, q)

        screen.fill("darkgray")

        # for b in balls:
        #     b.render(screen, (b.pos.x, b.pos.y))

        for b in movable_balls:
            b.render(screen, (b.pos.x, b.pos.y))

        pygame.display.flip()
        dt = clock.tick(60) / 1000


