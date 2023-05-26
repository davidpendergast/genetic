import typing

import sys
if sys.version_info < (3, 11):
    typing.Self = typing.Any

from functools import total_ordering

import pygame
import rainbowize
import random
import math

from genetics import Problem, Solution, GeneSpec, SolutionEvaluator, GeneticProblemSolver

SIZE = (320, 240)
INSET = 8

GRAPH_HEIGHT = 64

N_CUE_BALLS = 3
N_BALLS = 8
N_GOALS = 3

BALL_RADIUS = 8
IMPLUSE_FACTOR = 500
FRICTION = 40

MAX_SHOT_POWER = 250


class CueBallProblem(Problem):
    """Configuration of balls & level geometry."""

    def __init__(self, cue_ball_xys, ball_xys, goal_xys, rect):
        super().__init__()
        self.cue_ball_xys = cue_ball_xys
        self.ball_xys = ball_xys
        self.goal_xys = goal_xys
        self.rect = rect


class CueBallSolution(Solution):

    def __init__(self, size=None, data=None, specs=None):
        super().__init__(size=size,
                         data=data,
                         specs=specs if specs is not None else [GeneSpec('c'), GeneSpec('w')] * ((size or len(data)) // 2))

    def create_new(self, data: typing.Optional[typing.Tuple[float]]) -> typing.Self:
        return CueBallSolution(size=len(self.data), data=data, specs=self.specs)

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

class CueBallSolutionEvaluator(SolutionEvaluator):

    def __init__(self, problem: CueBallProblem, solution: CueBallSolution):
        super().__init__(problem, solution)

        # actual state of the simulation
        self.started = False
        self.done = False
        self.t = 0
        self.goal_xys = [pygame.Vector2(gxy) for gxy in problem.goal_xys]
        self.balls = [Ball(bxy, (0, 0)) for bxy in problem.ball_xys]
        self.cue_balls = [Ball(cxy, (0, 0), color="snow", is_cue=True) for cxy in problem.cue_ball_xys]

    @classmethod
    def create(cls, problem: CueBallProblem, solution: Solution) -> typing.Self:
        return CueBallSolutionEvaluator(problem, solution)

    def step(self, dt):
        self.t += dt

        if not self.started:
            for idx, cue in enumerate(self.cue_balls):
                shot = self.solution.get_shot(idx)
                cue.vel.x = shot.x
                cue.vel.y = shot.y
            self.started = True
        elif self.is_everything_stationary():
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


class AnimatedCueBallSolutionEvaluator(CueBallSolutionEvaluator):

    def __init__(self, problem: CueBallProblem, solution: CueBallSolution,
                 all_solutions=()):
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

    return CueBallProblem(cue_balls, balls, goals, playing_rect)


def create_solvers(problem) -> typing.Sequence[GeneticProblemSolver]:
    base_settings = {
        'population': 16,
        'time_limit_secs': 8,
        'time_step_secs': 1 / 60
    }
    proto_solution = CueBallSolution(size=2 * len(problem.cue_ball_xys))
    initial_solutions = [proto_solution.create_new(None) for _ in range(base_settings['population'])]
    return [
        # GeneticProblemSolver(
        #     problem,
        #     proto_solution,
        #     CueBallSolutionEvaluator,
        #     settings=dict(base_settings, **{'force_best_to_stay_chance': 0}),
        #     userdata={'name': "Normal", 'color': 'royalblue'},
        #     first_generation=initial_solutions
        # ),
        # GeneticProblemSolver(
        #     problem,
        #     proto_solution,
        #     CueBallSolutionEvaluator,
        #     settings=dict(base_settings, **{
        #         'force_best_to_stay_chance': 1.0,
        #         'rank_selection_diversity_weighting': 0.333,
        #         'rank_selection_pratio': 1.05
        #     }),
        #     userdata={'name': "33% Div, P=5%", 'color': 'green'},
        #     first_generation=initial_solutions
        # ),
        GeneticProblemSolver(
            problem,
            proto_solution,
            CueBallSolutionEvaluator,
            settings=dict(base_settings, **{
                'rank_selection_pratio': (1.15, 1.333, 10),
                'rank_selection_diversity_weighting': 0.333,
                'cross_chance': (0.333, 0.1, 10),
                'lerp_chance': (0.333, 0.1, 10),
                'avg_mutations_per_offspring': (3, 1, 10),
                'mutation_max_pcnt': (0.333, 0.1, 10),
                'max_attempts_per_solution': 1,
                'force_best_to_stay_chance': 1
            }),
            userdata={'name': "Hybrid", 'color': 'yellow'},
            first_generation=initial_solutions
        ),
        GeneticProblemSolver(
            problem,
            proto_solution,
            CueBallSolutionEvaluator,
            settings=dict(base_settings, **{
                'rank_selection_pratio': 1.15,
                'rank_selection_diversity_weighting': 0.333,
                'cross_chance': 0.33,
                'lerp_chance': 0.33,
                'avg_mutations_per_offspring': 3,
                'mutation_max_pcnt': 0.333,
                'max_attempts_per_solution': 1,
                'force_best_to_stay_chance': 1
            }),
            userdata={'name': "Hot", 'color': 'red'},
            first_generation=initial_solutions
        ),
        GeneticProblemSolver(
            problem,
            proto_solution,
            CueBallSolutionEvaluator,
            settings=dict(base_settings, **{
                'rank_selection_pratio': 1.333,
                'rank_selection_diversity_weighting': 0.333,
                'cross_chance': 0.1,
                'lerp_chance': 0.1,
                'avg_mutations_per_offspring': 1,
                'mutation_max_pcnt': 0.1,
                'max_attempts_per_solution': 1,
                'force_best_to_stay_chance': 1
            }),
            userdata={'name': "Cold", 'color': 'royalblue'},
            first_generation=initial_solutions
        ),
        # GeneticProblemSolver(
        #     problem,
        #     proto_solution,
        #     CueBallSolutionEvaluator,
        #     settings=dict(base_settings, **{
        #         'force_best_to_stay_chance': 1.0,
        #         'rank_selection_diversity_weighting': 0.333,
        #         'rank_selection_pratio': 1.4,
        #     }),
        #     userdata={'name': "33% Div, P=40%", 'color': 'cyan'},
        #     first_generation=initial_solutions
        # ),
        # GeneticProblemSolver(
        #     problem,
        #     proto_solution,
        #     CueBallSolutionEvaluator,
        #     settings=dict(base_settings, **{
        #         'force_best_to_stay_chance': 1.0,
        #         'rank_selection_diversity_weighting': 0.333,
        #         'rank_selection_pratio': 1.50,
        #     }),
        #     userdata={'name': "33% Div, P=50%", 'color': 'orange'},
        #     first_generation=initial_solutions
        # )
        # GeneticProblemSolver(
        #     problem,
        #     proto_solution,
        #     CueBallSolutionEvaluator,
        #     settings=dict(base_settings, **{
        #         'force_best_to_stay_chance': 1.0,
        #         'rank_selection_diversity_weighting': 0.666
        #     }),
        #     userdata={'name': "66% Diversity", 'color': 'yellow'},
        #     first_generation=initial_solutions
        # ),
        # GeneticProblemSolver(
        #     problem,
        #     proto_solution,
        #     CueBallSolutionEvaluator,
        #     settings=dict(base_settings, **{
        #         'force_best_to_stay_chance': 1.0,
        #         'rank_selection_diversity_weighting': 1.0
        #     }),
        #     userdata={'name': "100% Diverisity", 'color': 'red'},
        #     first_generation=initial_solutions
        # ),
        GeneticProblemSolver(
            problem,
            proto_solution,
            CueBallSolutionEvaluator,
            settings=dict(base_settings, **{
                'force_best_to_stay_chance': 0,
                'avg_mutations_per_offspring': 1000,
                'mutation_max_pcnt': 1,
            }),
            userdata={'name': "Random", 'color': 'orange'},
            first_generation=initial_solutions
        )
    ]


if __name__ == "__main__":
    pygame.init()

    screen_size = (SIZE[0], SIZE[1] + INSET + GRAPH_HEIGHT)
    screen = rainbowize.make_fancy_scaled_display(screen_size, 2, extra_flags=pygame.RESIZABLE)
    clock = pygame.Clock()
    dt = 0

    solvers = create_solvers(create_sample_problem())
    best_solver = None
    animated_solver = None
    seen_solutions = set()

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
                    solvers = create_solvers(create_sample_problem())
                    best_solver = None
                    animated_solver = None
                    seen_solutions.clear()

        if pygame.key.get_pressed()[pygame.K_RIGHT]:
            animated_solver = None

        if animated_solver is None or animated_solver.is_done():
            for solver in solvers:
                solver.step()
            best_solver = min(solvers, key=lambda solver: solver.best_ever)
            current_best = best_solver.get_best_solution()
            animated_solver = AnimatedCueBallSolutionEvaluator(best_solver.problem, current_best,
                                                               all_solutions=best_solver.get_current_generation())
            if best_solver.best_ever != current_best or best_solver.best_ever in seen_solutions:
                animated_solver.done = True  # skip it
            seen_solutions.add(current_best)

        animated_solver.step(1 / 60)  # fixed timestep for consistency's sake
        animated_solver.draw(screen)

        text_to_render = [
            (f"Solver: {best_solver.userdata['name']} (Generation {len(best_solver.generations)})", best_solver.userdata['color']),
            (f"Solution: {best_solver.get_best_solution()}", 'snow'),
            (f"Fitness: {animated_solver.get_fitness():.1f}", 'snow')
        ]

        text_y = INSET + 2
        for text, color in text_to_render:
            surf = font.render(text, True, color)
            screen.blit(surf, (INSET + 2, text_y))
            text_y += surf.get_height() + 2

        # draw fitness graph
        sorted_solvers = sorted(solvers, key=lambda s: s.best_ever.fitness)
        solver_labels = [font.render(solver.userdata['name'] + f"[{solver.elapsed_time:.1f}s]", True, solver.userdata['color']) for solver in sorted_solvers]
        offs = max(label.get_width() for label in solver_labels)

        graph_rect = pygame.Rect(INSET * 2 + offs, screen_size[1] - INSET - GRAPH_HEIGHT,
                                 screen_size[0] - INSET * 2 - (offs + INSET), GRAPH_HEIGHT)
        graph_inner_rect = graph_rect.inflate(-4, -4)

        # draw graph legend
        for idx, label in enumerate(solver_labels):
            screen.blit(label, (graph_rect[0] - label.get_width() - INSET, graph_rect[1] + idx * label.get_height()))

        pygame.draw.rect(screen, "black", graph_rect)
        pygame.draw.rect(screen, "gray", graph_rect, width=1)

        raw_lines = []  # list of (color, pts)
        raw_pts = []  # list of (color, pts)

        for idx, solver in enumerate(solvers):
            if len(best_solver.best_solutions) > 1:
                best_pts = []
                overall_best_pts = []
                avg_pts = []

                best = solver.best_solutions[0].fitness
                for idx, (s, avg) in enumerate(zip(solver.best_solutions, solver.avg_fitnesses)):
                    if s.fitness < best:
                        best = s.fitness
                        overall_best_pts.append((idx, best))
                    best_pts.append((idx, s.fitness))
                    avg_pts.append((idx, avg))

                color = solver.userdata['color']
                raw_lines.append((color, best_pts))
                raw_pts.append((pygame.Color(color).lerp('white', 0.666), overall_best_pts))

        max_y = 1
        max_x = 5
        for _, xy_list in (raw_lines + raw_pts):
            for xy in xy_list:
                max_x = max(max_x, xy[0])
                max_y = max(max_y, xy[1])

        for idx, (color, xy_line) in enumerate(raw_lines):
            xformed_line = [
                (round(graph_inner_rect.x + graph_inner_rect.width * x / max_x),
                 round(graph_inner_rect.y + graph_inner_rect.height * (1 - y / max_y))) for (x, y) in xy_line
            ]
            pygame.draw.lines(screen, color, False, xformed_line)
        for idx, (color, xy_line) in enumerate(raw_pts):
            xformed_pts = [
                (round(graph_inner_rect.x + graph_inner_rect.width * x / max_x),
                 round(graph_inner_rect.y + graph_inner_rect.height * (1 - y / max_y))) for (x, y) in xy_line
            ]
            for xy in xformed_pts:
                pygame.draw.circle(screen, color, xy, 3, width=1)

        dt = clock.tick(60) / 1000
        pygame.display.flip()

