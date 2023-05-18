import jax
import jax.numpy as jnp
from functools import partial

def value_and_jacrev(x, f):
    y, pullback = jax.vjp(f, x)
    basis = jnp.eye(y.size, dtype=y.dtype)
    jac = jax.vmap(pullback)(basis)[0]
    return y, jac

def regularized_hess(P):
    def regularization(P):
        eigs, vecs = jnp.linalg.eigh(P)
        delta = 1e-6
        eigs_modified = jnp.where(eigs < delta, delta, eigs) #jnp.maximum(-eigs, delta)
        return vecs @ jnp.diag(eigs_modified) @ vecs.T
    def identity(P):
        return P
    #is_not_pos_def = jnp.isnan(jnp.linalg.cholesky(P)).any()
    #P = jax.lax.cond(is_not_pos_def, regularization, identity, P)    
    return regularization(P)

class NLPBuilder:
    def __init__(self, dim):
        self.dim = dim
        self.f = None
        self._g = [lambda x: []] # eq = 0.
        self._h = [lambda x: []] # ineq <= 0.
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
        # self.lb = xlb
        # self.ub = xub

    def get_g(self):
        return lambda x: jnp.hstack([fn(x) for fn in self._g])
    
    def get_h(self):
        return lambda x: jnp.hstack([fn(x) for fn in self._h])
    
    def get_gh(self):
        return lambda x: jnp.hstack([fn(x) for fn in self._g+self._h])

    def get_lagrangian_fn(self):
        gh = self.get_gh()
        return lambda x, lmbda: self.f(x) + gh(x) @ lmbda
    
    def get_merit_fn(self):
        f = self.f
        g = self.get_g()
        h = self.get_h()
        def merit_fn(x, sigma=0.):
            eq_norm = jnp.linalg.norm(g(x), 1)
            ineq_norm = jnp.linalg.norm(jnp.clip(h(x), a_min=0.), 1)
            return f(x) + sigma * (eq_norm + ineq_norm)
        return merit_fn
    
    def get_merit_direc_deriv_fn(self):
        g = self.get_g()
        h = self.get_h()
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
    
    def get_PqGhAb_fn(self):
        lag_hess_fn = jax.hessian(self.get_lagrangian_fn())
        f_grad_fn = jax.grad(self.f)
        value_and_jac_g = partial(value_and_jacrev, f=self.get_g())
        value_and_jac_h = partial(value_and_jacrev, f=self.get_h())
        # eq_dim = self.g_dim
        # ineq_dim = self.h_dim
        def regularized_hess(P):
            eigs, vecs = jnp.linalg.eigh(P)
            delta = 1e-6
            eigs_modified = jnp.where(eigs < delta, jnp.maximum(-eigs, delta), eigs) #jnp.maximum(-eigs, delta)
            return vecs @ jnp.diag(eigs_modified) @ vecs.T
        
        def PqGhAb(x, lmbda):
            #exact newton
            P = lag_hess_fn(x, lmbda)
            is_not_pos_def = jnp.isnan(jnp.linalg.cholesky(P)).any()
            P = jax.lax.cond(is_not_pos_def, regularized_hess, lambda P: P, P)
            q = f_grad_fn(x)
            b, A = value_and_jac_g(x) #eq const
            h, G = value_and_jac_h(x) #ineq const
            return P, q, G, h, A, b
        return PqGhAb
    
    def get_convexify_fn(self):
        lag_hess_fn = jax.hessian(self.get_lagrangian_fn())
        f_grad_fn = jax.grad(self.f)
        value_and_jac_g = partial(value_and_jacrev, f=self.get_g())
        value_and_jac_h = partial(value_and_jacrev, f=self.get_h())
        eq_dim = self.g_dim
        ineq_dim = self.h_dim
        
        def PqAlu(x, lmbda):
            #exact newton
            P = lag_hess_fn(x, lmbda)
            is_not_pos_def = jnp.isnan(jnp.linalg.cholesky(P)).any()
            P = jax.lax.cond(is_not_pos_def, regularized_hess, lambda P: P, P)
            q = f_grad_fn(x)
            l, u = jnp.array([]), jnp.array([])
            if eq_dim != 0 and ineq_dim == 0: #eq only
                b, A = value_and_jac_g(x) #eq const
                l = -b
                u = -b
            elif eq_dim == 0 and ineq_dim != 0: #ineq only
                h, G = value_and_jac_h(x) #ineq const
                l = jnp.full(len(h), -jnp.inf)
                u = -h
                A = G
            elif eq_dim != 0 and ineq_dim != 0: #all
                h, G = value_and_jac_h(x)
                b, A = value_and_jac_g(x)
                l = jnp.hstack([-b, jnp.full(len(h), -jnp.inf)])  
                u = jnp.hstack([-b,-h])
                A = jnp.vstack([A, G])
            # l = jnp.hstack([-b, jnp.full(len(h), -jnp.inf)])
            # u = jnp.hstack([-b,-h])
            return P, q, A, l, u
        return PqAlu
    
    def get_const_viol_fn(self):
        g = self.get_g()
        h = self.get_h()
        eq_dim = self.g_dim
        ineq_dim = self.h_dim
        def const_viol(x, m):
            if eq_dim != 0:
                max_c_eq = jnp.linalg.norm(g(x), jnp.inf)
            else:
                max_c_eq = 0.
            if ineq_dim != 0:
                max_c_ineq = jnp.linalg.norm(jnp.clip(h(x), a_min=0), jnp.inf)
            else:
                max_c_ineq = 0.
            # max_c_eq = jnp.linalg.norm(g(x), jnp.inf)
            # max_c_ineq = jnp.linalg.norm(jnp.clip(h(x), a_min=0), jnp.inf)
            return jnp.maximum(max_c_eq, max_c_ineq)
        return const_viol

    def get_grad_and_const_viol(self):
        g = self.get_g()
        h = self.get_h()
        lag_grad_fn = jax.grad(self.get_lagrangian_fn())
        def grad_and_const_viol(x, m):
            max_c_eq = jnp.linalg.norm(g(x), jnp.inf)
            max_c_ineq = jnp.linalg.norm(jnp.clip(h(x), a_min=0), jnp.inf)
            max_lagr_grad = jnp.linalg.norm(lag_grad_fn(x, m))
            #max_viol = jnp.hstack([max_c_eq, max_c_ineq, max_lagr_grad]).max()
            return jnp.maximum(max_c_eq, max_c_ineq), max_lagr_grad
        return grad_and_const_viol
    
    def get_armijo_merit(self):
        g = self.get_g()
        h = self.get_h()
        f_grad_fn = jax.grad(self.f)
        eq_dim = self.g_dim
        ineq_dim = self.h_dim
        def armijo_merit(x, direction, sigma):
            direct_deriv = f_grad_fn(x) @ direction
            if eq_dim != 0:
                eq_1norm = jnp.linalg.norm(g(x), 1)
            else:
                eq_1norm = 0.
            ineq_1norm = 0. if ineq_dim == 0 \
                else jnp.linalg.norm(jnp.clip(h(x),a_min=0.), 1)
            return direct_deriv - sigma * (eq_1norm + ineq_1norm)
        return armijo_merit

    # def get_PqGhAb_fn(self):
    #     lag_hess_fn = jax.hessian(self.get_lagrangian_fn())
    #     f_grad_fn = jax.grad(self.f)
    #     const_and_jac_fn = partial(value_and_jacrev, f=self.get_gh())
    #     eq_dim = self.g_dim
    #     ineq_dim = self.h_dim
    #     def regularized_hess(P):
    #         eigs, vecs = jnp.linalg.eigh(P)
    #         delta = 1e-6
    #         eigs_modified = jnp.where(eigs < delta, delta, jnp.maximum(-eigs, delta))
    #         return vecs @ jnp.diag(eigs_modified) @ vecs.T
    #     def PqGhAb(x, lmbda):
    #         #exact newton
    #         P = lag_hess_fn(x, lmbda)
    #         is_not_pos_def = jnp.isnan(jnp.linalg.cholesky(P)).any()
    #         P = jax.lax.cond(is_not_pos_def, regularized_hess, lambda P: P, P)
    #         q = f_grad_fn(x)
    #         const, const_jac = const_and_jac_fn(x)
    #         A, b = const_jac[:eq_dim,:], -const[:eq_dim]
    #         G, h = const_jac[eq_dim:,:], -const[eq_dim:]
    #         l, u = jnp.asarray(self.lb), jnp.asarray(self.ub)
    #         return P, q, G, h, A, b, l, u
    #     return PqGhAb
    
    
    # def get_PqAlu_fn(self):
    #     lag_grad_fn = jax.grad(self.get_lagrangian_fn())
    #     f_grad_fn = jax.grad(self.f)
    #     const_and_jac_fn = partial(value_and_jacrev, f=self.get_gh())
    #     eq_dim = self.g_dim
    #     ineq_dim = self.h_dim
    #     def BFGS(B_prev, lag_grad_curr, lag_grad_prev, x_curr, x_prev):
    #         def constant_zero(B_prev, incr1, incr2):
    #             return B_prev
    #         def normal(B_prev, incr1, incr2):
    #             return B_prev + incr1 - incr2
    #         y = x_curr - x_prev
    #         s = lag_grad_curr - lag_grad_prev
    #         c1 = jnp.inner(y, s)
    #         c2 = s @ B_prev @ s
    #         incr1 = jnp.outer(y, y) / c1
    #         incr2 = B_prev @ jnp.outer(s, s) @ B_prev / c2
    #         cond = jnp.isclose(c1, 0.) | jnp.isclose(c2, 0.)
    #         return jax.lax.cond(cond, constant_zero, normal, B_prev, incr1, incr2)
    #     def PqAlu(x, lmbda, P_prev, lag_grad_prev, x_prev):
    #         # hessian by BFGS
    #         lag_grad_curr = lag_grad_fn(x, lmbda)
    #         P = BFGS(P_prev, lag_grad_curr, lag_grad_prev, x, x_prev)
    #         q = f_grad_fn(x)
    #         const, const_jac = const_and_jac_fn(x)
    #         A, u = const_jac, -const
    #         l = jnp.hstack([u[:eq_dim], jnp.full(ineq_dim, -jnp.inf)])
    #         return P, q, A, l, u
    #     return PqAlu
