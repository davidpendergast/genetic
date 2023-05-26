import math
import typing
import random


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