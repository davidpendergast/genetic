import math
import typing
import random

import pygame

Tuple3f = typing.Tuple[float, float, float]
Vector3Like = typing.Union[Tuple3f, pygame.Vector3]
QuaternionType = typing.Tuple[float, float, float, float]
QuaternionLike = typing.Union[float, int, QuaternionType, Vector3Like]

Qi = (0, 1, 0, 0)
Qj = (0, 0, 1, 0)
Qk = (0, 0, 0, 1)


def quat(q: QuaternionLike) -> QuaternionType:
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


def qmult(q: QuaternionLike, p: QuaternionLike) -> QuaternionType:
    a1, b1, c1, d1 = quat(q)
    a2, b2, c2, d2 = quat(p)
    return (
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2
    )

def qscale(q: QuaternionLike, a: float) -> QuaternionType:
    a1, b1, c1, d1 = quat(q)
    return (a1*a, b1*a, c1*a, d1*a)


def qprod(*qs: QuaternionLike) -> QuaternionType:
    res = (1, 0, 0, 0)
    for q in qs:
        res = qmult(res, q)
    return res


def qadd(q: QuaternionLike, p: QuaternionLike) -> QuaternionType:
    a1, b1, c1, d1 = quat(q)
    a2, b2, c2, d2 = quat(p)
    return (a1 + a2, b1 + b2, c1 + c2, d1 + d2)


def qsum(*qs: QuaternionLike) -> QuaternionType:
    res = (0, 0, 0, 0)
    for q in qs:
        res = qadd(res, q)
    return res


def qconj(q: QuaternionLike) -> QuaternionType:
    a, b, c, d = quat(q)
    return (a, -b, -c, -d)


def qnorm(q: QuaternionLike) -> float:
    a, b, c, d = quat(q)
    return math.sqrt(a*a + b*b + c*c + d*d)


def qnormalize(q: QuaternionLike, to_len=1) -> QuaternionType:
    return qmult(q, to_len / qnorm(q))


def qrecip(q: QuaternionLike) -> QuaternionType:
    return qmult(qconj(q), 1 / qnorm(q)**2)


def qpow(q: QuaternionLike, t: float) -> QuaternionType:
    a, b, c, d = quat(q)
    if b == 0 and c == 0 and d == 0:
        return (a ** t, 0, 0, 0)
    else:
        return qexp(qmult(t, qlog(q)))


def qexp(q: QuaternionLike) -> QuaternionType:
    a, b, c, d = quat(q)
    ea = math.e ** a
    theta = math.sqrt(b*b + c*c + d*d)
    s = math.sin(theta) / theta
    return (ea * math.cos(theta),
            ea * b * s,
            ea * c * s,
            ea * d * s)


def qlog(q: QuaternionLike) -> QuaternionType:
    a, b, c, d = quat(q)
    v_norm = math.sqrt(b*b + c*c + d*d)
    q_norm = qnorm(q)
    _ = 1 / v_norm
    val = 1 / v_norm * math.acos(a / q_norm)
    return (math.log(q_norm), val * b, val * c, val * d)


def qlerp(q: QuaternionLike, p: QuaternionLike, t: float) -> QuaternionType:
    a1, b1, c1, d1 = quat(q)
    a2, b2, c2, d2 = quat(p)
    return (a1 + t * (a2 - a1),
            b1 + t * (b2 - b1),
            c1 + t * (c2 - c1),
            d1 + t * (d2 - d1))


def qslerp(q: QuaternionLike, p: QuaternionLike, t: float) -> QuaternionType:
    q_r = qrecip(q)
    return qmult(q, qpow(qmult(q_r, p), t))


def qdecomp(q: QuaternionLike) -> typing.Tuple[Tuple3f, float]:
    a, b, c, d = quat(q)
    m = math.sqrt(b*b + c*c + d*d)
    v = (b / m, c / m, d / m) if m != 0 else (0, 0, 1)
    theta = 2*math.atan2(m, a)
    return v, theta


def get_rot_quat(v_3d: Vector3Like, theta: float) -> QuaternionType:
    return qadd(math.cos(theta / 2), qmult(math.sin(theta / 2), qnormalize(v_3d)))


def rotate_pt_about_quat(pt_3d: Vector3Like, q: QuaternionLike):
    pt_rot = qprod(q, pt_3d, qrecip(q))
    return pt_rot[1:4]


def rotate_pt_about_vec(pt_3d: Vector3Like, v_3d: Vector3Like, theta: float):
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


def to_euler2(q):
    qw, qx, qy, qz = q
    qprime = (-qw, qz, qy, qx)
    zyx = to_euler_321(qprime)
    return zyx[2], zyx[1], zyx[0]


def from_euler2(roll_x, pitch_y, yaw_z):
    zyx = (yaw_z, pitch_y, roll_x)
    qw, qx, qy, qz = from_euler_321(*zyx)
    return (-qw, qz, qy, qx)


def to_euler(q: QuaternionLike) -> Tuple3f:
    """
    :return: roll (x), pitch (y), yaw (z) [123 order]
    """
    return euler_321_to_123(to_euler_321(q))

def from_euler(roll_x: float, pitch_y: float, yaw_z: float) -> QuaternionType:
    return (from_euler_321(*euler_123_to_321((roll_x, pitch_y, yaw_z))))

def to_euler_321(q: QuaternionLike) -> Tuple3f:
    """
    :return: roll (x), pitch (y), yaw (z)
    """
    # roll(x - axis rotation)
    sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
    roll_x = math.atan2(sinr_cosp, cosr_cosp)

    # pitch(y - axis rotation)
    sinp = math.sqrt(1 + 2 * (q[0] * q[2] - q[1] * q[3]))
    cosp = math.sqrt(1 - 2 * (q[0] * q[2] - q[1] * q[3]))
    pitch_y = 2 * math.atan2(sinp, cosp) - math.pi / 2

    # yaw(z - axis rotation)
    siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
    yaw_z = math.atan2(siny_cosp, cosy_cosp)

    return roll_x, pitch_y, yaw_z

def from_euler_321(roll_x: float, pitch_y: float, yaw_z: float) -> QuaternionType:
    cr = math.cos(roll_x * 0.5)
    sr = math.sin(roll_x * 0.5)
    cp = math.cos(pitch_y * 0.5)
    sp = math.sin(pitch_y * 0.5)
    cy = math.cos(yaw_z * 0.5)
    sy = math.sin(yaw_z * 0.5)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)

def euler_321_to_123(xyz):
    c1, c2, c3 = tuple(math.cos(i) for i in xyz)
    s1, s2, s3 = tuple(math.sin(i) for i in xyz)
    m = ((c2*c3, -c2*s3, s2),
         (c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1),
         (s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2))
    a1 = math.atan2(m[1][0], m[0][0])  # z
    a2 = math.asin(-m[2][0])           # y
    a3 = math.atan2(m[2][1], m[2][2])  # x?
    return a3, a2, a1
def euler_123_to_321(xyz):
    c1, c2, c3 = tuple(math.cos(i) for i in xyz)
    s1, s2, s3 = tuple(math.sin(i) for i in xyz)
    m = ((c1*c2, c1*s2*s3-c3*s1, s1*s3 + c1*c3*s2),
         (c2*s1, c1*c3+s1*s2*s3, c3*s1*s2-c1*s3),
         (-s2, c2*s3, c2*c3))
    a1 = math.atan2(-m[1][2], m[2][2])  # x
    a2 = math.asin(m[0][2])             # y
    a3 = math.atan2(-m[0][1], m[0][0])  # z
    return a3, a2, a1


if __name__ == "__main__":
    angles = [(0, 0, 0),
              (math.pi, 1, 2)]
    for a in angles:
        print(f"{a=} -> q={from_euler(*a)} -> e(q)={to_euler(from_euler(*a))}")