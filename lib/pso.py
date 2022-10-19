from typing import Callable, List, Tuple
from enum import Enum
import random
from math import inf
from numpy import multiply, array, clip
from copy import deepcopy


class parameter:
    name: str = None
    lower: float = None
    upper: float = None

    def rand(self):
        scale = self.upper - self.lower
        return random.random()*scale + self.lower

    def rand_velocity(self):
        mag = self.rand()
        direction = 1 if random.random() > 0.5 else -1

        return mag*direction

    def __init__(self, name, lower_bound, upper_bound):
        self.name = name
        self.lower = lower_bound
        self.upper = upper_bound


class parameter_space:
    parameters: List[parameter] = None
    lower = None
    upper = None
    constraints = None

    def __init__(self, constraints):
        self.parameters = []
        self.constraints = constraints

    def set_lower_upper(self):
        self.lower = array([p.lower for p in self.parameters])
        self.upper = array([p.upper for p in self.parameters])

    def rand_param(self):
        r = []
        for p in self.parameters:
            r.append(p.rand())

        return r

    def rand_velocity(self):
        r = []
        for p in self.parameters:
            r.append(p.rand_velocity())

        return r

    def constrain_position(self, position):
        new_position = position
        for c in self.constraints:
            new_position = c(new_position)
        return new_position


class pso_particle:
    position: List[float] = None
    velocity: Tuple[float, List[float]] = None
    position_best: List[float] = None
    cost_best: List[float] = None
    cost: List[float] = None


class PSO_MODE(Enum):
    MIN = 0
    MAX = 1


class PSO:
    save_n_snapshots = 10
    n_iter: int = None
    particles: List[pso_particle] = None
    position_best = None
    cost_best: float = inf
    parameter_space: parameter_space = None
    c0: float = None
    c1: float = None
    c2: float = None
    cost_fn: Callable[[pso_particle], float] = None
    snapshots: List[List[pso_particle]] = None
    iter: int = 0
    mode: PSO_MODE = None

    def velocity(self, particle: pso_particle, x_best, r1, r2):
        t1 = self.c0*array(particle.velocity)
        t2 = self.c1*r1*(array(particle.position_best) -
                         array(particle.position))
        t3 = self.c2*r2*(array(x_best) - array(particle.position))
        return list(t1 + t2 + t3)

    def position(self, particle: pso_particle):
        return list(self.parameter_space.constrain_position(array(particle.position) + array(particle.velocity)))

    def __init__(
        self,
        param_data: List[Tuple[str, float, float]],
        cost_fn: Callable[[float, float], float],
        n_particles: int = 20,
        n_iter: int = 30,
        c0: float = 0.95,
        c1: float = 2,
        c2: float = 2,
        mode=PSO_MODE.MIN,
        constraints=[]
    ):
        self.constraints = constraints
        self.mode = mode
        self.particles = []
        self.parameter_space = parameter_space(constraints)
        self.snapshots = []
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.cost_best = inf if self.mode == PSO_MODE.MIN else -inf

        for p in param_data:
            param = parameter(p[0], p[1], p[2])
            self.parameter_space.parameters.append(param)
        self.parameter_space.set_lower_upper()

        self.position_best = self.parameter_space.rand_param()
        self.cost_fn = cost_fn
        self.n_particles = n_particles
        self.n_iter = n_iter

        for p in range(0, self.n_particles):
            particle = pso_particle()
            particle.position = self.parameter_space.constrain_position(
                self.parameter_space.rand_param())
            particle.velocity = self.parameter_space.rand_velocity()
            particle.cost = inf if self.mode == PSO_MODE.MIN else -inf
            particle.cost_best = inf if self.mode == PSO_MODE.MIN else -inf
            particle.position_best = particle.position
            self.particles.append(particle)

    def r1(self):
        return random.random()

    def r2(self):
        return random.random()

    def run(self):
        for i in range(0, self.n_iter):
            self.next()

        return (self.position_best, self.cost_best)

    def next(self):
        updates = []

        for p in self.particles:
            r1 = self.r1()
            r2 = self.r2()
            update = pso_particle()
            update.position = p.position

            update.velocity = self.velocity(
                p, self.position_best, r1, r2)
            update.position = self.position(update)
            update.position_best = p.position_best
            update.cost = self.cost_fn(update.position)
            update.cost_best = p.cost_best

            is_best_local = update.cost < p.cost_best if self.mode == PSO_MODE.MIN else update.cost > p.cost_best
            is_best_global = update.cost < self.cost_best if self.mode == PSO_MODE.MIN else update.cost > self.cost_best

            if is_best_local:
                update.cost_best = update.cost
                update.position_best = deepcopy(update.position)

            if is_best_global:
                # print('Found new best:')
                # print('Cost: ', update.cost)
                # print('Params', update.position)
                self.cost_best = update.cost
                self.position_best = deepcopy(update.position)

            updates.append(update)

        self.snapshots.append(
            (self.particles, self.cost_best, self.position_best))
        if len(self.snapshots) > 10:
            self.snapshots.pop(0)
        self.particles = updates

        self.iter += 1

    def get_particle_positions(self):
        return [p.position for p in self.particles]
