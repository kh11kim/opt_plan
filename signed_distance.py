import jax
import jax.numpy as jnp
from jax import Array
import jax_dataclasses as jdc
from jaxlie import *

d = jnp.zeros(3)
def distance(d):
    is_zero = jnp.allclose(d, 0.)
    d = jnp.where(is_zero, jnp.ones_like(d), d)
    l = jnp.linalg.norm(d)
    l = jnp.where(is_zero, 0., l)
    l = jnp.max(jnp.array([l, 1e-6]))
    return l

@jdc.pytree_dataclass
class circle:
    center: Array
    r: int
    def penetration(self, points):
        d = jax.vmap(distance)(points - self.center) - self.r
        return - (d)

@jdc.pytree_dataclass
class box:
    box_pose: SE3
    half_extents: Array
    def distance_single(self, point):
        point = self.box_pose.inverse().apply(point)
        q = jnp.abs(point) - self.half_extents
        return distance(jnp.maximum(q, 0)) + \
            jnp.minimum(jnp.maximum(q[0], jnp.maximum(q[1], q[2])), 0)
    def penetration(self, points):
        distances = jax.vmap(self.distance_single)(points)
        return -distances

@jdc.pytree_dataclass
class EnvSDF:
    sdfs: tuple
    def penetration(self, points, num_points:int, safe_dist:float=0.):
        result = jnp.zeros(num_points)
        for sdf in self.sdfs:
            result = jnp.maximum(sdf.penetration(points), result)
        return result + safe_dist
    
    def penetration_sum(self, points, num_points:int, safe_dist:float=0.):
        return self.penetration(points, num_points, safe_dist).sum()

