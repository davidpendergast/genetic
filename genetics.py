import typing

import sys
if sys.version_info < (3, 11):
    typing.Self = typing.Any

from functools import total_ordering

import random
import math


class Problem:

    def __init__(self):
        self.cached_solutions = {}


class GeneSpec:

    def __init__(self, wrapmode='c', upper=1, lower=0, integer=False):
        if wrapmode != 'c' and wrapmode != 'w':
            raise ValueError("wrapmode must be 'c' (clamp) or 'w' (wrap)")
        self.wrapmode = wrapmode
        if lower > upper:
            raise ValueError(f"lower must be less than upper: {lower} > {upper}")
        self.integer = integer
        self.upper = int(upper) if integer else upper
        self.lower = int(lower) if integer else lower

    def new_value(self) -> float:
        return self.lower + (self.upper - self.lower) * random.random()

    def choose(self, v1, v2, weight=0.5):
        return v1 if random.random() < weight else v2

    def lerp(self, v1, v2, t=0.5) -> float:
        v2 = self._unwrap(v1, v2) if self.wrapmode == 'w' else v2
        return self._normalize(v1 + t * (v2 - v1))

    def add(self, v, amt) -> float:
        return self._normalize(v + amt)

    def mutate(self, v, max_pct) -> float:
        max_change = max_pct * (self.upper - self.lower)
        return self.add(v, 2 * (random.random() - 0.5) * max_change)

    def distance(self, v1, v2) -> float:
        if self.wrapmode == 'c':
            return abs(v1 - v2)
        else:
            return abs(v1 - self._unwrap(v1, v2))

    def _clamp(self, val) -> float:
        return max(self.lower, min(self.upper, val))

    def _wrap(self, val) -> float:
        return self.lower + (val - self.lower) % self.upper

    def _round(self, val) -> float:
        if val == int(val):
            return int(val)
        else:
            upper = math.ceil(val)
            return int(math.floor(val) if random.random() < (upper - val) else upper)

    def _normalize(self, val) -> float:
        if self.integer:
            val = self._round(val)
        if self.wrapmode == 'c':
            val = self._clamp(val)
        else:
            val = self._wrap(val)
        return val

    def _unwrap(self, v1, v2) -> float:
        unwrapped_v2 = v2
        if abs(v1 - (v2 - (self.upper - self.lower))) < abs(v1 - unwrapped_v2):
            unwrapped_v2 = v2 - (self.upper - self.lower)
        elif abs(v1 - (v2 + (self.upper - self.lower))) < abs(v1 - unwrapped_v2):
            unwrapped_v2 = v2 + (self.upper - self.lower)
        return unwrapped_v2


@total_ordering
class Solution:
    """Sequence of shots."""

    def __init__(self, size: typing.Optional[int] = None,
                 data: typing.Optional[typing.Tuple[float]] = None,
                 specs: typing.Optional[typing.Sequence[GeneSpec]] = None):

        if data is not None and not isinstance(data, tuple):
            raise ValueError("data must be tuple")

        self.specs: typing.Tuple[GeneSpec] = specs if specs is not None else tuple(GeneSpec() for _ in range(size))
        self.data: typing.Tuple[float] = data if data is not None else tuple(spec.new_value() for spec in self.specs)
        self.fitness = float('inf')

    def create_new(self, data: typing.Optional[typing.Tuple[float]]) -> typing.Self:
        return Solution(size=len(self.data), data=data, specs=self.specs)

    def __lt__(self, other: typing.Self):
        return self.fitness < other.fitness

    def swap(self, other: typing.Self, weight=0.5) -> typing.Self:
        new_data = tuple((d1 if random.random() < weight else d2) for d1, d2, m in zip(self.data, other.data))
        return self.create_new(new_data)

    def lerp(self, other: typing.Self, t=0.5) -> typing.Self:
        new_data = tuple(spec.lerp(d1, d2, t=t) for d1, d2, spec in zip(self.data, other.data, self.specs))
        return self.create_new(new_data)

    def mutate(self, p_per_idx, max_pct) -> typing.Self:
        new_data = tuple((spec.mutate(v, max_pct) if random.random() < p_per_idx else v)
                         for v, spec in zip(self.data, self.specs))
        return self.create_new(new_data)

    def cross(self, other: typing.Self) -> typing.Tuple[typing.Self, typing.Self]:
        cross_idx = random.randint(0, len(self.data) - 1)
        new_data_1 = self.data[:cross_idx] + other.data[cross_idx:]
        new_data_2 = other.data[:cross_idx] + self.data[cross_idx:]
        return self.create_new(new_data_1), self.create_new(new_data_2)

    def distance_to(self, other: typing.Self) -> float:
        tot_sum = 0
        for d1, d2, spec in zip(self.data, other.data, self.specs):
            tot_sum += spec.distance(d1, d2) ** 2
        return math.sqrt(tot_sum)

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def __repr__(self):
        datas = ", ".join(tuple(f'{d:.3f}' for d in self.data))
        return f"S({datas}) = {self.fitness:.1f}"


class SolutionEvaluator:

    def __init__(self, problem: Problem, solution: Solution):
        self.problem = problem
        self.solution = solution

    def step(self, dt):
        raise NotImplementedError()

    def steps(self, dt, n):
        while not self.is_done() and n > 0:
            self.step(dt)
            n -= 1

    def is_done(self):
        raise NotImplementedError

    def get_fitness(self):
        raise NotImplementedError()

    @classmethod
    def create(cls, problem: Problem, solution: Solution) -> typing.Self:
        raise NotImplementedError()

class GeneticProblemSolver:

    DEFAULT_SETTINGS = {
        'population': 16,
        'time_limit_secs': float('inf'),
        'time_step_secs': 0.1,
        'rank_selection_pratio': 1.1,
        'rank_selection_diversity_weighting': 0.333,
        'cross_chance': 0.1,
        'lerp_chance': 0.1,
        'avg_mutations_per_offspring': 1,
        'mutation_max_pcnt': 0.1,
        'max_attempts_per_solution': 1,
        'force_best_to_stay_chance': 0.0
    }

    def __init__(self,
                 problem: Problem,
                 proto_solution: Solution,
                 evaluator_cls: typing.Type[SolutionEvaluator],
                 settings=None,
                 userdata=None,
                 first_generation=()):
        self.problem = problem
        self.proto_solution = proto_solution
        self.evaluator_cls = evaluator_cls

        self.settings = dict(GeneticProblemSolver.DEFAULT_SETTINGS, **(settings or {}))
        self._first_gen_override = first_generation

        self.generations = []

        self.best_ever = None
        self.best_solutions = []
        self.avg_fitnesses = []

        self.cached_solutions = {}  # Solution -> list of fitnesses
        self.userdata = userdata or {}

    def get_current_generation(self):
        return self.generations[-1]

    def get_best_solution(self, ever=False):
        if ever and self.best_ever is not None:
            return self.best_ever
        else:
            return self.get_current_generation()[0]

    def evaluate(self, solution) -> Solution:
        if solution not in self.cached_solutions or len(self.cached_solutions[solution]) < self.settings['max_attempts_per_solution']:
            if solution.fitness == float('inf'):
                time_limit = self.settings['time_limit_secs']
                dt = self.settings['time_step_secs']
                t = 0
                evaluator = self.evaluator_cls.create(self.problem, solution)
                while t < time_limit and not evaluator.is_done():
                    evaluator.step(dt)
                    t += dt
                solution.fitness = evaluator.get_fitness()

            if solution not in self.cached_solutions:
                self.cached_solutions[solution] = [solution.fitness]
            else:
                self.cached_solutions[solution].append(solution.fitness)
        else:
            solution.fitness = sum(self.cached_solutions[solution]) / len(self.cached_solutions[solution])

        return solution

    # def do_rank_selection(self, items, weights):
    #     rng = random.random()
    #     return random.choices(items, weights, k=1)[0]
    #     p = self.settings['rank_selection_pmax']
    #     for it in items:  # TODO just solve it
    #         if rng < p:
    #             return it
    #         else:
    #             rng *= (1 - p)
    #     return items[-1]

    def create_next_generation(self):
        if len(self.generations) == 0:
            next_gen = []
            for i in range(self.settings['population']):
                if i < len(self._first_gen_override):
                    next_gen.append(self.evaluate(self._first_gen_override[i]))
                else:
                    next_gen.append(self.evaluate(self.proto_solution.create_new(None)))
            next_gen.sort()
        else:
            cur_gen = self.get_current_generation()
            cur_gen_by_fitness = list(cur_gen)
            cur_gen_by_fitness.sort()

            rank_selection_weights = list(reversed([self.settings['rank_selection_pratio'] ** x for x in range(len(cur_gen))]))

            # survival
            survivors = []
            while len(survivors) < self.settings['population']:
                if len(survivors) == 0:
                    next_item = random.choices(cur_gen_by_fitness, weights=rank_selection_weights, k=1)[0]
                else:
                    cur_gen_by_diversity = list(cur_gen_by_fitness)
                    cur_gen_by_diversity.sort(key=lambda item: -sum(item.distance_to(other) for other in survivors))
                    items = {s: idx * idx for idx, s in enumerate(cur_gen_by_fitness)}
                    for idx, s in enumerate(cur_gen_by_diversity):
                        div_idx = idx * self.settings['rank_selection_diversity_weighting']
                        items[s] = math.sqrt(items[s] + div_idx * div_idx)
                    items_by_fitdiv = list(cur_gen_by_diversity)
                    random.shuffle(items_by_fitdiv)
                    items_by_fitdiv.sort(key=lambda item: items[item])

                    next_item = random.choices(items_by_fitdiv, weights=rank_selection_weights, k=1)[0]
                survivors.append(next_item)

            # crossover
            next_gen = []
            while len(survivors) > 0:
                rng = random.random()
                if rng < self.settings['cross_chance'] + self.settings['lerp_chance'] and len(survivors) > 1:
                    s1 = random.choice(survivors)
                    survivors.remove(s1)
                    s2 = random.choice(survivors)
                    survivors.remove(s2)

                    if rng < self.settings['cross_chance']:
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
            mutation_chance_per_idx = self.settings['avg_mutations_per_offspring'] / len(self.proto_solution.data)
            next_gen = [s.mutate(mutation_chance_per_idx, self.settings['mutation_max_pcnt']) for s in next_gen]

            for s in next_gen:
                self.evaluate(s)
            next_gen.sort()

            if random.random() < self.settings['force_best_to_stay_chance'] and cur_gen[0] not in next_gen:
                next_gen[-1] = cur_gen[0]
                next_gen.sort()

        fits = [f"{s.fitness:.1f}" for s in next_gen]
        print(f"GEN {len(self.generations) + 1}: {fits}")

        return next_gen

    def step(self) -> Solution:
        next_gen = self.create_next_generation()
        self.generations.append(next_gen)

        new_best = min(next_gen, key=lambda s: s.fitness)
        self.best_solutions.append(new_best)
        if self.best_ever is None or new_best.fitness < self.best_ever.fitness:
            self.best_ever = new_best
        self.avg_fitnesses.append(sum(s.fitness for s in next_gen) / len(next_gen))

        return new_best