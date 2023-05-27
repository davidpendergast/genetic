import math
import random

import pygame

import rainbowize
import utils.qutils as qutils


class Ball:

    def __init__(self, radius, texture, orient=None):
        self.radius = radius
        self.pos = pygame.Vector3(0, 0, 0)
        self.texture = texture
        self.orient = orient or qutils.Qk
        self.outline = ("white", 1)

        self._cache = {}

    def get_img(self):
        key = (tilt, self.orient, self.outline)
        if key not in self._cache:
            out_color, out_width = self.outline
            res = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            # pygame.draw.rect(res, "red", (0, 0, res.get_width(), res.get_height()), width=1)
            surf_center = pygame.Vector2(res.get_width() / 2, res.get_height() / 2)

            view_dir = pygame.Vector3(0, 0, -1)
            for y in range(res.get_height()):
                for x in range(res.get_width()):
                    if surf_center.distance_to((x + 0.5, y + 0.5)) < self.radius - 0.333:
                        R = self.radius
                        V, X = view_dir, pygame.Vector3(x + 0.5 - surf_center.x, y + 0.5 - surf_center.y, self.radius)
                        X2, V2 = X.magnitude_squared(), V.magnitude_squared()
                        t = (-2 * V.dot(X) - math.sqrt((2 * V.dot(X))**2 - 4 * V2 * (X2 - R*R))) / (2 * V2)

                        pt_on_sphere = X + t * V
                        pt_on_sphere = pygame.Vector3(qutils.rotate_pt_about_quat(pt_on_sphere, self.orient))
                        (r, theta, phi) = pt_on_sphere.as_spherical()
                        tx_coords = (int(phi % 360 * self.texture.get_width() / 360),
                                     int(theta / 180 * self.texture.get_height()))

                        # if (x, y) == (res.get_width() // 2, res.get_height() // 2):
                        #     print(f"{pt_on_sphere=}, {(r, theta, phi)=}, {tx_coords=}")
                        # print(f"{(x,y)=}, {pt_on_sphere}, {r=}, {theta=}, {phi=}, {tx_coords=}")
                        base_color = self.texture.get_at(tx_coords)
                        shaded_color = pygame.Color(base_color).lerp("black", t / R * 0.666)
                        res.set_at((x, y), shaded_color)

            if out_width > 0:
                pygame.draw.circle(res, out_color, surf_center, self.radius, width=out_width)

            self._cache[key] = res

        return self._cache[key]

    def move(self, vec3, rotate=True):
        ball.pos += vec3

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
    screen = rainbowize.make_fancy_scaled_display((256, 128), scale_factor=3, extra_flags=pygame.RESIZABLE)
    clock = pygame.Clock()

    texture = pygame.image.load("assets/rainbow_skulls.png")
    ball = Ball(24, texture)

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
            ball.move(move)

        if keys[pygame.K_r]:
            q = qutils.get_rot_quat((0, 0, 1), 2 * math.pi / 60)
            ball.orient = qutils.qmult(ball.orient, q)

        screen.fill("darkgray")
        ball.render(screen, (screen.get_width() // 2 + ball.pos.x, screen.get_height() // 2 + ball.pos.y))

        pygame.display.flip()
        clock.tick(60)


