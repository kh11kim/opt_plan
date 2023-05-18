import jax
import jax.numpy as jnp
from jax import Array
import jax_dataclasses as jdc
from jaxlie import *
from typing import Tuple

def safe_2norm(d):
    """differentiable 2-norm"""
    is_zero = jnp.allclose(d, 0.)
    d = jnp.where(is_zero, jnp.ones_like(d), d)
    l = jnp.linalg.norm(d)
    l = jnp.where(is_zero, 0., l)
    l = jnp.max(jnp.array([l, 1e-6]))
    return l

@jdc.pytree_dataclass
class SDF:
    def distance(self, points):
        raise NotImplementedError
    
@jdc.pytree_dataclass
class Circle(SDF):
    center: Array
    r: int
    def distance(self, point):
        return safe_2norm(point - self.center) - self.r

@jdc.pytree_dataclass
class Box(SDF):
    box_pose: SE3
    half_extents: Array
    def distance(self, point):
        point = self.box_pose.inverse().apply(point)
        q = jnp.abs(point) - self.half_extents
        return safe_2norm(jnp.maximum(q, 0)) + \
            jnp.minimum(jnp.maximum(q[0], jnp.maximum(q[1], q[2])), 0)
    # def penetration(self, points):
    #     distances = jax.vmap(self.distance_single)(points)
    #     return -distances

@jdc.pytree_dataclass
class EnvSDF(SDF):
    sdfs: Tuple[SDF]
    safe_dist: float
    def task_space_potential(self, d):
        def free(d):
            return 0.
        # def in_safe_dist(d):
        #     return 1/(2*self.safe_dist)*(d-self.safe_dist)**2
        # def in_col(d):
        #     return -d + 1/2*self.safe_dist
        def in_col(d):
            return (d - self.safe_dist)**2
        is_in_col = d - self.safe_dist < 0.
        #is_in_safe_dist = (0. <= d) & (d <= self.safe_dist)
        # is_free = safe_dist < d
        #switch_var = is_in_col + is_in_safe_dist*2
        #return jax.lax.switch(switch_var, [free, in_col, in_safe_dist], d)
        return jax.lax.cond(is_in_col, in_col, free, d)
    
    def distance(self, point):
        min_dist = jnp.inf
        for sdf in self.sdfs:
            min_dist = jnp.minimum(min_dist, sdf.distance(point))
        return min_dist
    
    def penetration(self, point):
        d = self.distance(point)
        p = self.task_space_potential(d)
        return p
    
    def penetrations(self, points):
        return jax.vmap(self.penetration)(points).sum()

