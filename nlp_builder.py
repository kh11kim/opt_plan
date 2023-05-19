import jax
import jax.numpy as jnp
from functools import partial

class NLPBuilder:
    def __init__(self, dim):
        self.dim = dim
        self.f = None
        self._g = [] # eq = 0.
        self._h = [] # ineq <= 0.
        self.lb = None
        self.ub = None
        self.g_dim = 0
        self.h_dim = 0
    
    @property
    def const_dim(self):
        return self.g_dim + self.h_dim
    
    def get_output_len(self, fn):
        out = fn(jnp.zeros(self.dim))
        if out.shape == tuple():
            return 1
        else:
            return out.shape[0]
            
    def set_f(self, fn):
        self.f = fn
    
    def add_eq_const(self, fn, val):
        fn_new = lambda x: fn(x) - val
        self._g.append(fn_new)
        self.g_dim += self.get_output_len(fn_new)

    def add_ineq_const_lb(self, fn, lb):
        fn_new = lambda x: -fn(x) + lb
        self._h.append(fn_new)
        self.h_dim += self.get_output_len(fn_new)
    
    def add_ineq_const_ub(self, fn, ub):
        fn_new = lambda x: fn(x) - ub
        self._h.append(fn_new)
        self.h_dim += self.get_output_len(fn_new)

    def add_ineq_const_b(self, fn, lb, ub):
        lb_fn = lambda x: -fn(x) + lb
        ub_fn = lambda x: fn(x) - ub
        self._h.append(lb_fn)
        self._h.append(ub_fn)
        self.h_dim += self.get_output_len(lb_fn) * 2

    def set_state_bound(self, xlb, xub):
        def state_bound_fn(x):
            return jnp.hstack([-x + xlb, x - xub])
        self._h.append(state_bound_fn)
        self.h_dim += self.dim*2

    def get_g_fn(self):
        return lambda x: jnp.hstack([fn(x) for fn in self._g])
    
    def get_h_fn(self):
        return lambda x: jnp.hstack([fn(x) for fn in self._h])
    
    def get_gh_fn(self):
        return lambda x: jnp.hstack([fn(x) for fn in self._g+self._h])

    def get_lagrangian_fn(self):
        gh = self.get_gh_fn()
        return lambda x, lmbda: self.f(x) + gh(x) @ lmbda
    
    def get_merit_fn(self):
        f = self.f
        g = self.get_g_fn()
        h = self.get_h_fn()
        def merit_fn(x, sigma=0.):
            eq_norm = jnp.linalg.norm(g(x), 1)
            ineq_norm = jnp.linalg.norm(jnp.clip(h(x), a_min=0.), 1)
            return f(x) + sigma * (eq_norm + ineq_norm)
        return merit_fn
    
    def get_merit_direc_deriv_fn(self):
        g = self.get_g_fn()
        h = self.get_h_fn()
        f_grad_fn = jax.grad(self.f)
        eq_dim = self.g_dim
        ineq_dim = self.h_dim
        def merit_direc_deriv(x, direction, sigma):
            direct_deriv = f_grad_fn(x) @ direction
            if eq_dim != 0:
                eq_1norm = jnp.linalg.norm(g(x), 1)
            else:
                eq_1norm = 0.
            ineq_1norm = 0. if ineq_dim == 0 \
                else jnp.linalg.norm(jnp.clip(h(x),a_min=0.), 1)
            return direct_deriv - sigma * (eq_1norm + ineq_1norm)
        return merit_direc_deriv
