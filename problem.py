import typing
from functools import total_ordering

import pygame
import rainbowize
import random
import math

SIZE = (320, 240)
INSET = 8

GRAPH_HEIGHT = 64

N_CUE_BALLS = 2
N_BALLS = 2
N_GOALS = 2

BALL_RADIUS = 8
IMPLUSE_FACTOR = 500
FRICTION = 40

MAX_SHOT_POWER = 250

SOLUTION_TIME_LIMIT = 8  # seconds
TIME_STEP = 1 / 60

POPULATION_SIZE = 32

BEST_ITEM_SELECTION_PROBABILITY = 0.1
MUTATION_CHANCE_PER_IDX = 1 / N_CUE_BALLS
MUTATION_RANGE = 0.1
CROSS_CHANCE = 0.1
LERP_CHANCE = 0.1
DIVERSITY_WEIGHTING = 0.333
FORCE_BEST_TO_STAY = True


class Problem:
    """Configuration of balls & level geometry."""

    def __init__(self, cue_ball_xys, ball_xys, goal_xys, rect, solution_producer):
        self.cue_ball_xys = cue_ball_xys
        self.ball_xys = ball_xys
        self.goal_xys = goal_xys
        self.rect = rect

        self.solution_producer = solution_producer
        self.cached_solutions = {}

    def new_random_solution(self):
        return self.solution_producer()

@total_ordering
class Solution:
    """Sequence of shots."""

    def __init__(self, data, mode='c'):
        self.data = tuple(data)
        self.mode = (mode * (len(self.data) // len(mode)))[:len(self.data)]

        self.fitness = float('inf')

    def create_new(self, data):
        return Solution(data, mode=self.mode)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def swap(self, other, weight=0.5):
        new_data = [(d1 if random.random() < weight else d2) for d1, d2, m in zip(self.data, other.data)]
        return self.create_new(new_data)

    def _lerp_val(self, v1, v2, weight, mode):
        if mode == 'c':
            return min(1, max(0, v1 + (v2 - v1) * weight))
        else:
            v2_l = (v2 - 1)
            v2_u = (v2 + 1)
            if abs(v1 - v2_l) < min(abs(v1 - v2), abs(v1 - v2_u)):
                return (v1 + (v2_l - v1) * weight) % 1.0
            elif abs(v1 - v2) < abs(v1 - v2_u):
                return (v1 + (v2 - v1) * weight) % 1.0
            else:
                return (v1 + (v2_u - v1) * weight) % 1.0

    def lerp(self, other, weight=0.5):
        return self.create_new([self._lerp_val(d1, d2, weight, m) for d1, d2, m in zip(self.data, other.data, self.mode)])

    def mutate(self, p_per_idx, amt):
        raw_vals = [(d + 2 * (random.random() - 0.5) * amt if random.random() < p_per_idx else d) for d in self.data]
        new_data = [min(1, max(0, v)) if m == 'c' else v % 1.0 for v, m in zip(raw_vals, self.mode)]
        return self.create_new(new_data)

    def cross(self, other):
        cross_idx = random.randint(0, len(self.data) - 1)
        new_data_1 = self.data[:cross_idx] + other.data[cross_idx:]
        new_data_2 = other.data[:cross_idx] + self.data[cross_idx:]
        return self.create_new(new_data_1), self.create_new(new_data_2)

    def _val_dist(self, v1, v2, mode):
        if mode == 'c':
            return abs(v1 - v2)
        else:
            return min(abs(v1 - v2), abs(v1 - v2 - 1), abs(v1 - v2 + 1))

    def distance_to(self, other):
        tot_sum = 0
        for d1, d2, m in zip(self.data, other.data, self.mode):
            tot_sum += self._val_dist(d1, d2, m) ** 2
        return math.sqrt(tot_sum)

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
        super().__init__(data, mode="wc")

    def create_new(self, data):
        return ShotSequenceSolution(data)

    def num_shots(self):
        return len(self.data) // 2

    def get_shot(self, idx) -> pygame.Vector2:
        if idx * 2 >= len(self.data):
            return pygame.Vector2()
        else:
            angle = self.data[idx * 2]
            power = self.data[idx * 2 + 1]
            vec = pygame.Vector2()
            vec.from_polar((power * MAX_SHOT_POWER, angle * 360))
            return vec

    def get_shots(self) -> typing.Generator[pygame.Vector2, None, None]:
        for idx in range(self.num_shots()):
            yield self.get_shot(idx)

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
        self.started = False
        self.done = False
        self.t = 0
        self.goal_xys = [pygame.Vector2(gxy) for gxy in problem.goal_xys]
        self.balls = [Ball(bxy, (0, 0)) for bxy in problem.ball_xys]
        self.cue_balls = [Ball(cxy, (0, 0), color="snow", is_cue=True) for cxy in problem.cue_ball_xys]

    def update(self, dt):
        self.t += dt

        if self.t > self.t_limit:
            self.done = True

        if not self.started:
            for idx, cue in enumerate(self.cue_balls):
                shot = self.solution.get_shot(idx)
                cue.vel.x = shot.x
                cue.vel.y = shot.y
            self.started = True
        elif self.is_everything_stationary() or self.t >= SOLUTION_TIME_LIMIT:
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
        for c in self.cue_balls:
            yield c
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

    def __init__(self, problem: Problem, solution: ShotSequenceSolution,
                 all_solutions=[]):
        super().__init__(problem, solution)
        self.all_solutions = all_solutions

    def draw(self, screen):
        screen.fill('black')

        # draw all solutions in current generation
        if not pygame.key.get_pressed()[pygame.K_SPACE]:
            best_fitness = min(s.fitness for s in self.all_solutions)
            worst_fitness = max(s.fitness for s in self.all_solutions)
            best_color = pygame.Color("gray66")
            worst_color = pygame.Color("black")
            for idx, s in enumerate(reversed(self.all_solutions)):
                color_pcnt = (s.fitness - best_fitness) / (worst_fitness - best_fitness) if worst_fitness != best_fitness else 0
                color = best_color.lerp(worst_color, color_pcnt)  # if idx != len(self.all_solutions) - 1 else "gray66"
                for shot_idx, shot in enumerate(s.get_shots()):
                    start = self.problem.cue_ball_xys[shot_idx]
                    end = shot / MAX_SHOT_POWER * 64 + start
                    pygame.draw.line(screen, color, start, end)

        pygame.draw.rect(screen, "gray", self.problem.rect, width=1)

        for gxy in self.goal_xys:
            pygame.draw.circle(screen, "yellow", gxy, 10, width=1)

        for cue in self.cue_balls:
            cue.draw(screen, cue.xy)

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
    playing_rect.y += 48
    playing_rect.height -= 48
    spawn_rect = playing_rect.inflate(-BALL_RADIUS * 2, -BALL_RADIUS * 2)

    balls = [(SIZE[0] // 2, SIZE[1] // 2)]  # always put a cue ball in middle
    while len(balls) < N_BALLS + N_CUE_BALLS:
        xy = rand_point_in_rect(spawn_rect)
        not_colliding = True
        for b_xy in balls:
            if point_dist(xy, b_xy) <= BALL_RADIUS * 2:
                not_colliding = False
                break
        if not_colliding:
            balls.append(xy)

    cue_balls = balls[:N_CUE_BALLS]
    balls = balls[N_CUE_BALLS:]
    goals = [rand_point_in_rect(spawn_rect) for _ in range(N_GOALS)]

    return Problem(cue_balls, balls, goals, playing_rect,
                   lambda: ShotSequenceSolution.create_random(N_CUE_BALLS))


class SolutionManager:

    def __init__(self, problem):
        self.problem = problem
        self.generations = []

        self.best_ever = None
        self.best_solutions = []

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

    def do_rank_selection(self, items, p=BEST_ITEM_SELECTION_PROBABILITY):
        rng = random.random()
        for it in items:  # TODO just solve it
            if rng < p:
                return it
            else:
                rng *= (1 - p)
        return items[-1]

    def create_next_generation(self):
        if len(self.generations) == 0:
            next_gen = [self.evaluate(self.problem.new_random_solution()) for _ in range(POPULATION_SIZE)]
            next_gen.sort()
        else:
            cur_gen = self.get_current_generation()
            cur_gen_by_fitness = list(cur_gen)
            cur_gen_by_fitness.sort()

            # survival
            survivors = []
            while len(survivors) < POPULATION_SIZE:
                if len(survivors) == 0:
                    next_item = self.do_rank_selection(cur_gen_by_fitness)
                else:
                    cur_gen_by_diversity = list(cur_gen_by_fitness)
                    cur_gen_by_diversity.sort(key=lambda item: -sum(item.distance_to(other) for other in survivors))
                    items = {s: idx * idx for idx, s in enumerate(cur_gen_by_fitness)}
                    for idx, s in enumerate(cur_gen_by_diversity):
                        div_idx = idx * DIVERSITY_WEIGHTING
                        items[s] = math.sqrt(items[s] + div_idx * div_idx)
                    items_by_fitdiv = list(cur_gen_by_diversity)
                    random.shuffle(items_by_fitdiv)
                    items_by_fitdiv.sort(key=lambda item: items[item])

                    next_item = self.do_rank_selection(items_by_fitdiv)
                survivors.append(next_item)

            # crossover
            next_gen = []
            while len(survivors) > 0:
                rng = random.random()
                if rng < CROSS_CHANCE + LERP_CHANCE and len(survivors) > 1:
                    s1 = random.choice(survivors)
                    survivors.remove(s1)
                    s2 = random.choice(survivors)
                    survivors.remove(s2)

                    if rng < CROSS_CHANCE:
                        s1, s2 = s1.cross(s2)
                    else:
                        s1, s2 = s1.lerp(s2, 0.333), s1.lerp(s2, 0.666)

                    next_gen.append(s1)
                    next_gen.append(s2)
                else:
                    s = random.choice(survivors)
                    survivors.remove(s)
                    next_gen.append(s)

            # mutation
            next_gen = [s.mutate(MUTATION_CHANCE_PER_IDX, MUTATION_RANGE) for s in next_gen]

            for s in next_gen:
                self.evaluate(s)
            next_gen.sort()

            if FORCE_BEST_TO_STAY and cur_gen[0] not in next_gen:
                next_gen[-1] = cur_gen[0]
                next_gen.sort()

        fits = [f"{s.fitness:.1f}" for s in next_gen]
        print(f"GEN {len(self.generations) + 1}: {fits}")

        return next_gen

    def step(self) -> Solution:
        self.generations.append(self.create_next_generation())

        new_best = self.get_best_solution()
        self.best_solutions.append(new_best)
        if self.best_ever is None or new_best.fitness < self.best_ever.fitness:
            self.best_ever = new_best
        return new_best


if __name__ == "__main__":
    pygame.init()

    screen_size = (SIZE[0], SIZE[1] + INSET + GRAPH_HEIGHT)
    screen = rainbowize.make_fancy_scaled_display(screen_size, 2, extra_flags=pygame.RESIZABLE)
    clock = pygame.Clock()
    dt = 0

    def generate_solution():
        return ShotSequenceSolution.create_random(N_CUE_BALLS)

    problem = create_sample_problem()
    solver = SolutionManager(problem)

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
                    solver = SolutionManager(problem)
                    animated_solver = None

        if animated_solver is None or animated_solver.is_done():
            prev_best_ever = solver.best_ever
            next_solution = solver.step()
            animated_solver = AnimatedProblemSolver(problem, next_solution,
                                                    all_solutions=solver.get_current_generation())
            if prev_best_ever is not None and prev_best_ever == solver.best_ever:
                animated_solver.done = True  # skip it

        animated_solver.update(1 / 60)  # fixed timestep for consistency's sake
        animated_solver.draw(screen)

        text_to_render = [
            (f"Generation: {len(solver.generations)}", "snow"),
            (f"Solution: {solver.get_best_solution()}", "snow"),
            (f"Fitness: {animated_solver.get_fitness():.1f}", "snow")
        ]

        text_y = INSET + 2
        for text, color in text_to_render:
            surf = font.render(text, True, color)
            screen.blit(surf, (INSET + 2, text_y))
            text_y += surf.get_height() + 2

        # draw fitness graph
        graph_rect = pygame.Rect(INSET, screen_size[1] - INSET - GRAPH_HEIGHT,
                                 screen_size[0] - INSET * 2, GRAPH_HEIGHT)
        graph_inner_rect = graph_rect.inflate(-4, -4)

        pygame.draw.rect(screen, "black", graph_rect)
        pygame.draw.rect(screen, "gray", graph_rect, width=1)
        if len(solver.best_solutions) > 1:
            max_y = max(1, max(s.fitness for s in solver.best_solutions))
            max_x = max(25, len(solver.best_solutions))
            pts = []
            best_pts = []
            best = solver.best_solutions[0].fitness
            for idx, s in enumerate(solver.best_solutions):
                pt = (round(graph_inner_rect.x + graph_inner_rect.width * idx / max_x),
                      round(graph_inner_rect.y + graph_inner_rect.height * (1 - s.fitness / max_y)))
                if s.fitness < best:
                    best = s.fitness
                    best_pts.append(pt)
                pts.append(pt)
            pygame.draw.lines(screen, "royalblue", False, pts)
            for pt in best_pts:
                pygame.draw.circle(screen, "snow", pt, 3, width=1)

        dt = clock.tick(60) / 1000
        pygame.display.flip()

