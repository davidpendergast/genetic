import typing

import pygame
import rainbowize
import random

SIZE = (320, 240)
INSET = 8
N_BALLS = 15
N_GOALS = 5

IMPLUSE_FACTOR = 500
FRICTION = 15

class Problem:
    """Configuration of balls & level geometry."""

    def __init__(self, cue_ball_xy, ball_xys, goal_xys, rect):
        self.cue_ball_xy = cue_ball_xy
        self.ball_xys = ball_xys
        self.goal_xys = goal_xys
        self.rect = rect

class Solution:
    """Sequence of shots."""

    def __init__(self, data):
        self.data = data


BALL_UID_COUNTER = 0
def _next_uid():
    global BALL_UID_COUNTER
    BALL_UID_COUNTER += 1
    return BALL_UID_COUNTER - 1


class Ball:

    def __init__(self, xy, vel, radius=8, color="plum", is_cue=False):
        self.xy = pygame.Vector2(xy)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.is_cue = is_cue
        self.uid = _next_uid()

    def apply_physics(self, dt):
        self.xy += self.vel * dt

        # how does friction work??
        if self.vel.magnitude() < 0.001:
            self.vel *= 0  # just zero it out if it's too slow already
        else:
            opposite_dir = -self.vel.normalize()
            friction_vec = opposite_dir * FRICTION * dt  # slow down X px/sec per second?~
            if friction_vec.magnitude() > self.vel.magnitude():
                self.vel *= 0
            else:
                self.vel += friction_vec

    def draw(self, screen, at_xy):
        pygame.draw.circle(screen, self.color, at_xy, self.radius, width=1)

    def __hash__(self):
        return self.uid

    def __eq__(self, other):
        return self.uid == other.uid

class ProblemSolver:

    def __init__(self, problem, solution):
        # initial conditions
        self.problem = problem
        self.solution = solution
        self.t_limit = 15  # seconds

        # actual state of the simulation
        self.t = 0
        self.goal_xys = [pygame.Vector2(gxy) for gxy in problem.goal_xys]
        self.balls = [Ball(bxy, (0, 0)) for bxy in problem.ball_xys]
        self.cue_ball = Ball(problem.cue_ball_xy, (0, 0), color="snow", is_cue=True)

    def update(self, dt):
        self.t += dt

        if self.is_everything_stationary():
            # either finish or apply next shot
            pass

        # simulate movement
        for b in self.all_balls():
            b.apply_physics(dt)

        self.resolve_collisions(dt)

    def resolve_collisions(self, dt):
        impulses = {}  # map from Ball to list of impulses needed to fix its velocity... hmmm
        rect = self.problem.rect

        for ball in self.all_balls():
            impulses[ball] = list()
            if ball.xy[0] - ball.radius < rect.x:
                impulses[ball].append(pygame.Vector2(abs(ball.xy[0] - ball.radius - rect.x), 0))
            if ball.xy[0] + ball.radius > rect.right:
                impulses[ball].append(pygame.Vector2(-abs(ball.xy[0] + ball.radius - rect.right), 0))
            if ball.xy[1] - ball.radius < rect.y:
                impulses[ball].append(pygame.Vector2(0, abs(ball.xy[1] - ball.radius - rect.y)))
            if ball.xy[1] + ball.radius > rect.bottom:
                impulses[ball].append(pygame.Vector2(0, -abs(ball.xy[1] + ball.radius - rect.bottom)))

            for other in self.all_balls():
                if other != ball:
                    dist = ball.xy.distance_to(other.xy)
                    if dist < ball.radius + other.radius:
                        imp = (ball.xy - other.xy)
                        imp.scale_to_length(ball.radius + other.radius - dist)
                        impulses[ball].append(imp)

        for ball in impulses:
            if len(impulses[ball]) > 0:
                total_impulse = IMPLUSE_FACTOR * sum(impulses[ball], start=pygame.Vector2())
                ball.vel += total_impulse * dt

    def all_balls(self) -> typing.Generator[Ball, None, None]:
        yield self.cue_ball
        for b in self.balls:
            yield b

    def is_everything_stationary(self):
        return all([b.vel.magnitude() <= 0.0001 for b in self.all_balls()])

    def is_done(self):
        return False

    def get_fitness(self, problem, solution):
        if self.is_done():
            return 0
        else:
            raise ValueError("Not finished simulating yet")


class AnimatedProblemSolver(ProblemSolver):

    def __init__(self, problem, solution):
        super().__init__(problem, solution)

    def draw(self, screen):
        screen.fill('black')
        pygame.draw.rect(screen, "gray", self.problem.rect, width=1)

        for gxy in self.goal_xys:
            pygame.draw.circle(screen, "yellow", gxy, 10, width=1)

        self.cue_ball.draw(screen, self.cue_ball.xy)

        for target_ball in self.balls:
            target_ball.draw(screen, target_ball.xy)


def rand_point_in_rect(rect):
    return (random.randint(rect.x, rect.x + rect.width - 1),
            random.randint(rect.y, rect.y + rect.height - 1))


def create_sample_problem():
    playing_rect = pygame.Rect(INSET, INSET, SIZE[0] - INSET * 2, SIZE[1] - INSET * 2)
    spawn_rect = playing_rect.inflate(-8, -8)
    sample_problem = Problem((SIZE[0] // 2, SIZE[1] // 2),
                             [rand_point_in_rect(spawn_rect) for _ in range(N_BALLS)],
                             [rand_point_in_rect(spawn_rect) for _ in range(N_GOALS)],
                             playing_rect)
    sample_solution = Solution([])

    solver = AnimatedProblemSolver(sample_problem, sample_solution)
    solver.cue_ball.vel = pygame.Vector2(160, 10)

    return solver


if __name__ == "__main__":
    pygame.init()

    screen = rainbowize.make_fancy_scaled_display(SIZE, 2, extra_flags=pygame.RESIZABLE)
    clock = pygame.Clock()
    dt = 0

    solver = create_sample_problem()

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    solver = create_sample_problem()

        solver.update(dt)
        solver.draw(screen)

        dt = clock.tick(60) / 1000
        pygame.display.flip()

