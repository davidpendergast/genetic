import math
import random
import typing
import threedee
import pygame


Qi = (0, 1, 0, 0)
Qj = (0, 0, 1, 0)
Qk = (0, 0, 0, 1)


def quat(q: typing.Union[float, int, tuple]) -> typing.Tuple:
    if isinstance(q, (float, int)):
        return (q, 0, 0, 0)
    elif len(q) == 4:
        return q
    elif len(q) == 1:
        return (q[0], 0, 0, 0)
    elif len(q) == 3:
        return (0, q[0], q[1], q[2])
    else:
        raise ValueError(f"invalid input: {q}")


def qmult(q, p):
    a1, b1, c1, d1 = quat(q)
    a2, b2, c2, d2 = quat(p)
    return (
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2
    )


def qprod(*qs):
    res = (1, 0, 0, 0)
    for q in qs:
        res = qmult(res, q)
    return res


def qadd(q, p):
    a1, b1, c1, d1 = quat(q)
    a2, b2, c2, d2 = quat(p)
    return (a1 + a2, b1 + b2, c1 + c2, d1 + d2)


def qsum(*qs):
    res = (0, 0, 0, 0)
    for q in qs:
        res = qadd(res, q)
    return res


def qconj(q):
    a, b, c, d = quat(q)
    return (a, -b, -c, -d)


def qnorm(q):
    a, b, c, d = quat(q)
    return math.sqrt(a*a + b*b + c*c + d*d)


def qnormalize(q, to_len=1):
    return qmult(q, to_len / qnorm(q))


def qrecip(q):
    return qmult(qconj(q), 1 / qnorm(q)**2)


def qpow(q, t):
    a, b, c, d = quat(q)
    if b == 0 and c == 0 and d == 0:
        return (a ** t, 0, 0, 0)
    else:
        return qexp(qmult(t, qlog(q)))


def qexp(q):
    a, b, c, d = quat(q)
    ea = math.e ** a
    theta = math.sqrt(b*b + c*c + d*d)
    s = math.sin(theta) / theta
    return (ea * math.cos(theta),
            ea * b * s,
            ea * c * s,
            ea * d * s)


def qlog(q):
    a, b, c, d = quat(q)
    v_norm = math.sqrt(b*b + c*c + d*d)
    q_norm = qnorm(q)
    _ = 1 / v_norm
    val = 1 / v_norm * math.acos(a / q_norm)
    return (math.log(q_norm), val * b, val * c, val * d)


def qlerp(q, p, t):
    a1, b1, c1, d1 = quat(q)
    a2, b2, c2, d2 = quat(p)
    return (a1 + t * (a2 - a1),
            b1 + t * (b2 - b1),
            c1 + t * (c2 - c1),
            d1 + t * (d2 - d1))


def qslerp(q, p, t):
    q_r = qrecip(q)
    return qmult(q, qpow(qmult(q_r, p), t))


def qdecomp(q):
    a, b, c, d = quat(q)
    m = math.sqrt(b*b + c*c + d*d)
    v = (b / m, c / m, d / m) if m != 0 else (0, 0, 1)
    theta = 2*math.atan2(m, a)
    return v, theta


def get_rot_quat(v_3d, theta):
    return qadd(math.cos(theta / 2), qmult(math.sin(theta / 2), qnormalize(v_3d)))


def rotate_pt_about_quat(pt_3d, q):
    pt_rot = qprod(q, pt_3d, qrecip(q))
    return pt_rot[1:4]


def rotate_pt_about_vec(pt_3d, v_3d, theta):
    q = get_rot_quat(v_3d, theta)
    return rotate_pt_about_quat(pt_3d, q)


def lerp(v1, v2, t):
    return tuple(v1[i] + t * (v2[i] - v1[i]) for i in range(max(len(v1), len(v2))))


def sgn(a):
    if a == 0:
        return 0
    else:
        return -1 if a < 0 else 1


def rand_vec(dim):
    vals = [random.gauss(0, 1 / (2 * math.pi)) for _ in range(dim)]
    m = math.sqrt(sum(v*v for v in vals))
    if m == 0:
        return (1,) + (0,) * (dim - 1)
    else:
        return tuple(v / m for v in vals)


class Point3D:
    def __init__(self, pt, color=(255, 255, 255), width=3):
        self.pt = pt
        self.color = color
        self.width = width


class Model3D:

    def __init__(self):
        pass

    def get_pos(self) -> pygame.Vector3:
        return pygame.Vector3(0, 0, 0)

    def get_qrot(self):
        return (1, 0, 0, 0)

    def get_lines(self):
        return []

    def get_pts(self):
        return []

    def get_xformed_lines(self):
        q = self.get_qrot()
        offs = self.get_pos()
        for line in self.get_lines():
            p1_rot = pygame.Vector3(rotate_pt_about_quat(line.p1, q))
            p2_rot = pygame.Vector3(rotate_pt_about_quat(line.p2, q))
            yield threedee.Line3D(offs + p1_rot, offs + p2_rot, color=line.color, width=line.width)

    def get_xformed_pts(self):
        q = self.get_qrot()
        offs = self.get_pos()
        for pt in self.get_pts():
            pt_rot = pygame.Vector3(rotate_pt_about_quat(pt, q))
            yield pt_rot + offs

    def update(self, dt):
        pass


class DiceModel3D(Model3D):

    FACE_ORIENTATIONS = [
        get_rot_quat((1, 0, 0), 0),             # 1
        get_rot_quat((1, 0, 0), math.pi / 2),   # 2
        get_rot_quat((1, 0, 0), -math.pi / 2),  # 3
        get_rot_quat((1, 0, 0), math.pi),       # 4
        get_rot_quat((0, 0, 1), math.pi / 2),   # 5
        get_rot_quat((0, 0, 1), -math.pi / 2),  # 6
    ]

    COLORS = list(map(pygame.Color, ("white", "red", "yellow", "blue", "magenta", "green")))

    def __init__(self, face_idx=None, rest_pos=(0, 0, 0), size=10, width=1):
        super().__init__()
        self.rest_pos = pygame.Vector3(rest_pos)

        self.t = 0
        self.height = 15
        self.extra_rots = 0
        self.extra_rot_axis = None
        self.duration = 3
        self.cur_face = random.randint(0, len(DiceModel3D.FACE_ORIENTATIONS) - 1) if face_idx is None else face_idx
        self.next_face = None

        self.lines = threedee.gen_cube(0, size, (0, 0, 0), DiceModel3D.COLORS[self.cur_face])
        for l in self.lines:
            l.width = width

    def do_roll(self, duration=3., height=10., extra_rots=0, next_idx=None):
        n_faces = len(DiceModel3D.FACE_ORIENTATIONS)
        self.next_face = random.randint(0, n_faces - 1) if next_idx is None else (next_idx % n_faces)

        self.t = 0
        self.height = height
        self.duration = duration

        self.extra_rots = extra_rots
        self.extra_rot_axis = rand_vec(3)

    def roll_prog(self):
        if self.next_face is None:
            return 0
        else:
            return min(1.0, self.t / self.duration)

    def get_pos(self) -> pygame.Vector3:
        prog = self.roll_prog()
        h = 4*self.height*(-prog*prog + prog)
        return self.rest_pos + pygame.Vector3(0, h, 0)

    def get_qrot(self):
        prog = self.roll_prog()
        if prog == 0:
            return DiceModel3D.FACE_ORIENTATIONS[self.cur_face]
        else:
            t = prog

            extra_roll_angle = 2 * math.pi * self.extra_rots * t
            extra_roll_q = get_rot_quat(self.extra_rot_axis, extra_roll_angle)

            primary_roll_q = qslerp(DiceModel3D.FACE_ORIENTATIONS[self.cur_face],
                                    DiceModel3D.FACE_ORIENTATIONS[self.next_face], t)

            return qmult(primary_roll_q, extra_roll_q)

    def get_color(self):
        prog = self.roll_prog()
        if prog == 0:
            return DiceModel3D.COLORS[self.cur_face]
        else:
            c1 = DiceModel3D.COLORS[self.cur_face]
            c2 = DiceModel3D.COLORS[self.next_face]
            return c1.lerp(c2, prog)

    def get_lines(self):
        return self.lines

    def update(self, dt):
        if self.next_face is not None:
            if self.t >= self.duration:
                self.cur_face = self.next_face
                self.next_face = None
                self.t = 0
                print(f"Rolled a {self.cur_face+1}!")
                pygame.display.set_caption(f"Current Roll: {dice_model.cur_face + 1}")
            else:
                self.t += dt

        c = self.get_color()
        for l in self.lines:
            l.color = c


if __name__ == "__main__":
    import pygame
    pygame.init()

    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)

    clock = pygame.time.Clock()
    dt = 0

    camera = threedee.KBControlledCamera3D()
    camera.fov_degrees = 60
    camera.position = pygame.Vector3(0, 10, -70)

    dice_model = DiceModel3D(width=3)

    models = [dice_model]

    running = True
    while running:
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_i:
                    print("camera = " + str(camera))
                elif e.key == pygame.K_SPACE:
                    if dice_model.roll_prog() == 0:
                        dice_model.do_roll(duration=1, height=18, extra_rots=random.randint(1, 2))

        keys_held = pygame.key.get_pressed()
        camera.update(dt, keys_held=keys_held)
        dice_model.update(dt)

        screen.fill((0, 0, 0))

        lines_3d = []
        for m in models:
            lines_3d.extend(m.get_xformed_lines())

        lines_2d = camera.project_to_surface(screen, lines_3d, depth_shading=(20, 200))

        for line in lines_2d:
            pygame.draw.line(screen, line.color, line.p1, line.p2, width=line.width)

        pygame.display.update()
        dt = clock.tick(60) / 1000
