import typing
from functools import total_ordering

import pygame
import rainbowize
import random
import math

SIZE = (320, 240)
INSET = 8

N_BALLS = 15
N_GOALS = 5

BALL_RADIUS = 8
IMPLUSE_FACTOR = 100
FRICTION = 20

MAX_SHOT_POWER = 75


class Problem:
    """Configuration of balls & level geometry."""

    def __init__(self, ball_xys, goal_xys, rect):
        self.cue_ball_xy = ball_xys[0]
        self.ball_xys = ball_xys[1:]
        self.goal_xys = goal_xys
        self.rect = rect

@total_ordering
class Solution:
    """Sequence of shots."""

    def __init__(self, data):
        self.data = data
        self.fitness = float('inf')

    def __lt__(self, other):
        return self.fitness < other.fitness

    @classmethod
    def create_random_data(cls, size):
        return [random.random() for _ in range(size)]

class ShotSequenceSolution(Solution):

    def __init__(self, data):
        super().__init__(data)

    def num_shots(self):
        return len(self.data) // 2

    def get_shot(self, idx) -> pygame.Vector2:
        if idx * 2 >= len(self.data):
            return pygame.Vector2()
        else:
            angle = self.data[idx // 2]
            power = self.data[idx // 2 + 1]
            vec = pygame.Vector2()
            vec.from_polar((angle * 360, power * MAX_SHOT_POWER))
            return vec

    @classmethod
    def create_random(cls, num_shots):
        data = Solution.create_random_data(num_shots * 2)
        return ShotSequenceSolution(data)


BALL_UID_COUNTER = 0
def _next_uid():
    global BALL_UID_COUNTER
    BALL_UID_COUNTER += 1
    return BALL_UID_COUNTER - 1


class Ball:

    def __init__(self, xy, vel, color="plum", is_cue=False):
        self.xy = pygame.Vector2(xy)
        self.vel = pygame.Vector2(vel)
        self.radius = BALL_RADIUS
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

    def draw(self, screen, at_xy, color=None):
        pygame.draw.circle(screen, self.color if color is None else color,
                           at_xy, self.radius, width=1)

    def __hash__(self):
        return self.uid

    def __eq__(self, other):
        return self.uid == other.uid

class ProblemSolver:

    def __init__(self, problem: Problem, solution: ShotSequenceSolution):
        # initial conditions
        self.problem = problem
        self.solution = solution
        self.t_limit = 15  # seconds

        # actual state of the simulation
        self.shot_idx = -1
        self.t = 0
        self.goal_xys = [pygame.Vector2(gxy) for gxy in problem.goal_xys]
        self.balls = [Ball(bxy, (0, 0)) for bxy in problem.ball_xys]
        self.cue_ball = Ball(problem.cue_ball_xy, (0, 0), color="snow", is_cue=True)

    def update(self, dt):
        self.t += dt

        if self.shot_idx == -1 or self.is_everything_stationary():
            if self.shot_idx < self.solution.num_shots() - 1:
                self.shot_idx += 1
                next_shot = self.solution.get_shot(self.shot_idx)
                self.cue_ball.vel.x = next_shot.x
                self.cue_ball.vel.y = next_shot.y
            else:
                self.solution.fitness = self.get_fitness()

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

    def get_dist_to_nearest_goal(self, xy):
        best = float('inf')
        for goal_xy in self.goal_xys:
            best = min(best, point_dist(goal_xy, xy))
        return best

    def all_balls(self) -> typing.Generator[Ball, None, None]:
        yield self.cue_ball
        for b in self.balls:
            yield b

    def is_everything_stationary(self):
        return all([b.vel.magnitude() <= 0.0001 for b in self.all_balls()])

    def is_done(self):
        return self.solution.fitness != float('inf')

    def get_fitness(self):
        if self.solution.fitness != float('inf'):
            return self.solution.fitness
        else:
            total = 0
            for ball in self.all_balls():
                if not ball.is_cue:
                    total += self.get_dist_to_nearest_goal(ball.xy)
            return total


class AnimatedProblemSolver(ProblemSolver):

    def __init__(self, problem: Problem, solution: ShotSequenceSolution):
        super().__init__(problem, solution)

    def draw(self, screen):
        screen.fill('black')
        pygame.draw.rect(screen, "gray", self.problem.rect, width=1)

        for gxy in self.goal_xys:
            pygame.draw.circle(screen, "yellow", gxy, 10, width=1)

        self.cue_ball.draw(screen, self.cue_ball.xy)

        for target_ball in self.balls:
            fitness = self.get_dist_to_nearest_goal(target_ball.xy)
            color = pygame.Color("green").lerp("red", min(1, (fitness / (self.problem.rect.width // 3))))
            target_ball.draw(screen, target_ball.xy, color=color)


def rand_point_in_rect(rect):
    return (random.randint(rect.x, rect.x + rect.width - 1),
            random.randint(rect.y, rect.y + rect.height - 1))


def point_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)

def create_sample_problem():
    playing_rect = pygame.Rect(INSET, INSET, SIZE[0] - INSET * 2, SIZE[1] - INSET * 2)
    spawn_rect = playing_rect.inflate(-BALL_RADIUS * 2, -BALL_RADIUS * 2)

    balls = [(SIZE[0] // 2, SIZE[1] // 2)]
    while len(balls) < N_BALLS:
        xy = rand_point_in_rect(spawn_rect)
        not_colliding = True
        for b_xy in balls:
            if point_dist(xy, b_xy) <= BALL_RADIUS * 2:
                not_colliding = False
                break
        if not_colliding:
            balls.append(xy)

    goals = [rand_point_in_rect(spawn_rect) for _ in range(N_GOALS)]

    sample_problem = Problem(balls, goals, playing_rect)
    sample_solution = ShotSequenceSolution.create_random(3)

    return AnimatedProblemSolver(sample_problem, sample_solution)


if __name__ == "__main__":
    pygame.init()

    screen = rainbowize.make_fancy_scaled_display(SIZE, 2, extra_flags=pygame.RESIZABLE)
    clock = pygame.Clock()
    dt = 0

    solver = create_sample_problem()

    font = pygame.Font(None, size=16)

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

        solver.update(1 / 60)  # fixed timestep for consistency's sake
        solver.draw(screen)

        text_y = INSET + 2
        fitness_text = f"Fitness: {solver.get_fitness():.2f}"
        fitness_surf = font.render(fitness_text, True, "snow" if not solver.is_done() else "yellow")
        screen.blit(fitness_surf, (INSET + 2, text_y))
        text_y += fitness_surf.get_height() + 2

        shot_text = f"Shot: {solver.shot_idx + 1}"
        shot_surf = font.render(shot_text, True, "snow")
        screen.blit(shot_surf, (INSET + 2, text_y))
        text_y += shot_surf.get_height() + 2

        dt = clock.tick(60) / 1000
        pygame.display.flip()

