import typing
from functools import total_ordering

import pygame
import rainbowize
import random
import math

SIZE = (320, 240)
INSET = 8

N_BALLS = 4
N_GOALS = 3
N_SHOTS = 1

BALL_RADIUS = 8
IMPLUSE_FACTOR = 500
FRICTION = 40

MAX_SHOT_POWER = 500

SOLUTION_TIME_LIMIT = 15  # seconds
TIME_STEP = 1 / 60

POPULATION_SIZE = 32

MUTATION_CHANCE = 0.25
SWAP_CHANCE = 0.30
LERP_CHANCE = 0.40
NOISE_CHANCE_PER_IDX = 0.25
NOISE_SIGMA = 0.25


class Problem:
    """Configuration of balls & level geometry."""

    def __init__(self, ball_xys, goal_xys, rect):
        self.cue_ball_xy = ball_xys[0]
        self.ball_xys = ball_xys[1:]
        self.goal_xys = goal_xys
        self.rect = rect

        self.cached_solutions = {}

@total_ordering
class Solution:
    """Sequence of shots."""

    def __init__(self, data):
        self.data = tuple(data)
        self.fitness = float('inf')

    def create_new(self, data):
        return Solution(data)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def swap(self, other, weight=0.5):
        new_data = [(d1 if random.random() < weight else d2) for d1, d2 in zip(self.data, other.data)]
        return self.create_new(new_data)

    def lerp(self, other, weight=0.5):
        new_data = [(d1 + weight * (d2 - d1)) for d1, d2 in zip(self.data, other.data)]
        return self.create_new(new_data)

    def jibble(self, amount):
        new_data = [(d + 2 * (random.random() - 0.5) * amount) % 1.0 for d in self.data]
        return self.create_new(new_data)

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def __repr__(self):
        datas = ", ".join(tuple(f'{d:.3f}' for d in self.data))
        return f"S({datas}) = {self.fitness:.1f}"

    @classmethod
    def create_random_data(cls, size):
        return [random.random() for _ in range(size)]

class ShotSequenceSolution(Solution):

    def __init__(self, data):
        super().__init__(data)

    def create_new(self, data):
        return ShotSequenceSolution(data)

    def num_shots(self):
        return len(self.data) // 2

    def get_shot(self, idx) -> pygame.Vector2:
        if idx * 2 >= len(self.data):
            return pygame.Vector2()
        else:
            angle = self.data[idx // 2]
            power = self.data[idx // 2 + 1]
            vec = pygame.Vector2()
            vec.from_polar((power * MAX_SHOT_POWER, angle * 360))
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
        self.problem = problem
        self.solution = solution
        self.t_limit = SOLUTION_TIME_LIMIT

        # actual state of the simulation
        self.done = False
        self.shot_idx = -1
        self.t = 0
        self.goal_xys = [pygame.Vector2(gxy) for gxy in problem.goal_xys]
        self.balls = [Ball(bxy, (0, 0)) for bxy in problem.ball_xys]
        self.cue_ball = Ball(problem.cue_ball_xy, (0, 0), color="snow", is_cue=True)

    def update(self, dt):
        self.t += dt

        if self.t > self.t_limit:
            self.done = True

        if self.shot_idx == -1 or self.is_everything_stationary():
            if self.shot_idx < self.solution.num_shots() - 1:
                self.shot_idx += 1
                next_shot = self.solution.get_shot(self.shot_idx)
                self.cue_ball.vel.x = next_shot.x
                self.cue_ball.vel.y = next_shot.y
            else:
                self.done = True
                if self.solution.fitness == float('inf'):
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
        return self.done

    def get_fitness(self):
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
            fitness = self.get_dist_to_nearest_goal(target_ball.xy) + 1
            color = pygame.Color("green").lerp("red", min(1, (fitness / (self.problem.rect.width // 3))))
            target_ball.draw(screen, target_ball.xy, color=color)


def rand_point_in_rect(rect):
    return (random.randint(rect.x, rect.x + rect.width - 1),
            random.randint(rect.y, rect.y + rect.height - 1))


def point_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)

def create_sample_problem() -> Problem:
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

    return Problem(balls, goals, playing_rect)


class SolutionManager:

    def __init__(self, problem,
                 solution_generator,
                 pop_size=POPULATION_SIZE,
                 mutation_rate=MUTATION_CHANCE,
                 swap_rate=SWAP_CHANCE,
                 preserve_best=False):
        self.problem = problem
        self.solution_generator = solution_generator

        self.size = pop_size
        self.mutation_rate = mutation_rate
        self.swap_rate = swap_rate
        self.preserve_best = preserve_best

        self.generations = []

    def get_current_generation(self):
        return self.generations[-1]

    def get_best_solution(self):
        return self.get_current_generation()[0]

    def evaluate(self, solution) -> Solution:
        if solution in self.problem.cached_solutions:
            solution.fitness = self.problem.cached_solutions[solution]
        else:
            t = 0
            solver = ProblemSolver(self.problem, solution)
            while t < SOLUTION_TIME_LIMIT and not solver.is_done():
                solver.update(TIME_STEP)
                t += TIME_STEP
            solution.fitness = solver.get_fitness()
        return solution

    def create_next_generation(self):
        if len(self.generations) == 0:
            next_gen = [self.evaluate(self.solution_generator()) for _ in range(self.size)]
        else:
            cur_gen = self.get_current_generation()
            weights = [1 / s.fitness for s in cur_gen]

            next_gen = []
            while len(next_gen) < self.size:
                rng = random.random()

                if rng < MUTATION_CHANCE:
                    new_solution = self.solution_generator()
                else:
                    s1, s2 = random.choices(cur_gen, weights, k=2)
                    s3 = self.evaluate(s1.lerp(s2, weight=0.333 + 0.333 * random.random()))
                    if s3.fitness <= min(s1.fitness, s2.fitness):
                        new_solution = s3
                    else:
                        new_solution = min(s1, s2)

                if new_solution in next_gen:
                    new_solution = new_solution.jibble(0.25)

                next_gen.append(self.evaluate(new_solution))

        next_gen.sort()

        fits = [f"{s.fitness:.1f}" for s in next_gen]
        print(f"GEN {len(self.generations) + 1}: {fits}")

        return next_gen

    def step(self) -> Solution:
        self.generations.append(self.create_next_generation())
        return self.get_best_solution()


if __name__ == "__main__":
    pygame.init()

    screen = rainbowize.make_fancy_scaled_display(SIZE, 2, extra_flags=pygame.RESIZABLE)
    clock = pygame.Clock()
    dt = 0

    problem = create_sample_problem()
    solver = SolutionManager(problem, lambda: ShotSequenceSolution.create_random(N_SHOTS))

    animated_solver = None

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
                    problem = create_sample_problem()
                    solver = SolutionManager(problem, lambda: ShotSequenceSolution.create_random(N_SHOTS))
                    animated_solver = None

        if animated_solver is None or animated_solver.is_done():
            last_best = None if animated_solver is None else animated_solver.solution.fitness
            next_solution = solver.step()
            animated_solver = AnimatedProblemSolver(problem, next_solution)
            if last_best is not None and next_solution.fitness >= last_best:
                animated_solver.done = True  # skip it

        animated_solver.update(1 / 60)  # fixed timestep for consistency's sake
        animated_solver.draw(screen)

        text_to_render = [
            (f"Generation: {len(solver.generations)}", "snow"),
            (f"Solution: {solver.get_best_solution()}", "snow"),
            (f"Shot: {animated_solver.shot_idx + 1}, Fitness: {animated_solver.get_fitness():.1f}", "snow")
        ]

        text_y = INSET + 2
        for text, color in text_to_render:
            surf = font.render(text, True, color)
            screen.blit(surf, (INSET + 2, text_y))
            text_y += surf.get_height() + 2

        dt = clock.tick(60) / 1000
        pygame.display.flip()

