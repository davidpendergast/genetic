import math
import random

import pygame

import rainbowize
import utils.qutils as qutils

class Ball:

    def __init__(self, radius, texture, orient=None):
        self.radius = radius
        self.texture = texture
        self.orient = orient or qutils.Qk
        self.outline = ("white", 1)

        self._cache = {}

    def get_img(self, tilt=90):
        key = (tilt, self.orient, self.outline)
        if key not in self._cache:
            out_color, out_width = self.outline
            res = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.rect(res, "red", (0, 0, res.get_width(), res.get_height()), width=1)
            surf_center = pygame.Vector2(res.get_width() / 2, res.get_height() / 2)

            view_dir = pygame.Vector3(0, 0, 1)
            for y in range(res.get_height()):
                for x in range(res.get_width()):
                    if surf_center.distance_to((x + 0.5, y + 0.5)) < self.radius - 0.333:
                        R = self.radius
                        X = pygame.Vector3(x + 0.5 - surf_center.x, y + 0.5 - surf_center.y, -self.radius)
                        V = view_dir
                        X2 = X.magnitude_squared()
                        V2 = V.magnitude_squared()
                        t = (-2 * V.dot(X) - math.sqrt((2 * V.dot(X))**2 - 4 * V2 * (X2 - R*R))) / (2 * V2)

                        pt_on_sphere = X + t * V
                        pt_on_sphere = pygame.Vector3(qutils.rotate_pt_about_quat(pt_on_sphere, self.orient))
                        pt_on_sphere = pygame.Vector3(qutils.rotate_pt_about_vec(pt_on_sphere, (1, 0, 0), -tilt / 180 * math.pi))
                        (r, theta, phi) = pt_on_sphere.as_spherical()

                        tx_coords = (int(phi % 360 / 360 * self.texture.get_width()) % self.texture.get_width(),
                                     int((theta + 90) / 180 * self.texture.get_height()) % self.texture.get_height())
                        # print(f"{(x,y)=}, {pt_on_sphere}, {r=}, {theta=}, {phi=}, {tx_coords=}")
                        base_color = self.texture.get_at(tx_coords)

                        res.set_at((x, y), pygame.Color(base_color).lerp("black", t / R))

            if out_width > 0:
                pygame.draw.circle(res, out_color, surf_center, self.radius, width=out_width)

            self._cache[key] = res

        return self._cache[key]

    def render(self, dest: pygame.Surface, xy, tilt=90):
        # pygame.draw.circle(surface, "red", xy, self.radius, width=0)
        surf = self.get_img(tilt=tilt)
        dest.blit(surf, (xy[0] - surf.get_width() / 2,
                         xy[1] - surf.get_height() / 2 + surf.get_height() * math.cos(tilt * math.pi / 180)))


if __name__ == "__main__":
    pygame.init()
    screen = rainbowize.make_fancy_scaled_display((256, 128), scale_factor=3, extra_flags=pygame.RESIZABLE)
    # screen = pygame.display.set_mode((640, 480))
    clock = pygame.Clock()

    texture = pygame.image.load("assets/rainbow_axis.png")
    ball = Ball(32, texture)

    tilt = 90

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LSHIFT] and keys[pygame.K_DOWN]:
            tilt -= 1
        if keys[pygame.K_LSHIFT] and keys[pygame.K_UP]:
            tilt += 1
        tilt = min(90, max(0, tilt))

        screen.fill("darkgray")
        ball.render(screen, (screen.get_width() // 2, screen.get_height() // 2), tilt=tilt)

        # ball.orient = qutils.qmult(ball.orient, qutils.get_rot_quat((0.3, 1.3, 1), math.pi / 30))

        pygame.display.flip()
        clock.tick(60)


