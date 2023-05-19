import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
from typing import List, Dict, Callable
import jax_dataclasses as jdc
from nlp_builder import NLPBuilder
from cvxopt import matrix, sparse, solvers
from functools import partial

@jdc.pytree_dataclass
class OptState:
    x: Array
    m: Array
    sigma: float
    lag_grad: Array
    lag_hess: Array
    f_val: Array
    f_grad: Array
    g_val: Array
    g_jac: Array
    h_val: Array
    h_jac: Array

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


class SQPSolver:
    def __init__(
        self, 
        dim: int, 
        const_dim: int,
        eval_fn:Callable,
        armijo_cond_fn: Callable,
    ):
        self.dim = dim
        self.const_dim = const_dim
        self.eval_fn = eval_fn
        self.armijo_cond_fn = armijo_cond_fn
        self.history = []

    def prebuild(self):
        #???
        x = jnp.zeros(self.dim)
        m = jnp.zeros(self.const_dim)
        sigma = 0.
        state = self.eval_fn(x, m, sigma)
        self.armijo_cond_fn(x, x, 0., 0., 0.)
        self.max_viol(state)
        self.max_grad(state)
        self.backtrack(state, x)
        self.update(state, 0., x, m)

    
    def backtrack(
        self, 
        state: OptState,
        direction: Array,
        alpha=1.,
        beta=0.8,
        gamma=0.1,
        max_iter=20,
    ):
        x, sigma = state.x, state.sigma
        for i in range(max_iter):
            if self.armijo_cond_fn(x, direction, alpha, gamma, sigma):
                break
            alpha *= beta
        return alpha
    
    def convexify(self, state):
        P = matrix(np.asarray(state.lag_hess).astype(np.double))
        q = matrix(np.asarray(state.f_grad).astype(np.double))
        G = sparse(matrix(np.asarray(state.h_jac).astype(np.double)))
        h = matrix(-np.asarray(state.h_val).astype(np.double))
        A = sparse(matrix(np.asarray(state.g_jac).astype(np.double)))
        b = matrix(-np.asarray(state.g_val).astype(np.double))
        return P, q, G, h, A, b
    
    def max_viol(self, state):
        max_c_eq = jnp.linalg.norm(state.g_val, jnp.inf)
        max_c_ineq = jnp.linalg.norm(jnp.clip(state.h_val, a_min=0.), jnp.inf)
        return jnp.maximum(max_c_eq, max_c_ineq)
    
    def max_grad(self, state):
        return jnp.linalg.norm(state.lag_grad, jnp.inf)
    
    def solve(
        self,
        x0,
        max_iter=50,
        tol=0.05,
        const_viol_tol=0.001,
        qp_reltol=0.01,
        save_history=False,
        verbose=True,
    ):
        qp_options = {
            "reltol":qp_reltol,
            "show_progress":False,
        }
        self.history = []

        x = x0.copy()
        m = jnp.zeros(self.const_dim)
        sigma = 0.1
        state = self.eval_fn(x, m, sigma)

        for i in range(max_iter):
            # Check terminal condition
            if self.max_grad(state) < tol and \
               self.max_viol(state) < const_viol_tol:
                break
            elif save_history:
                self.history.append(state)

            # Solve QP
            P, q, G, h, A, b = self.convexify(state)
            sol = solvers.qp(
                P, q, G, h, A, b,
                options=qp_options
            )
            if sol['status'] != 'optimal':
                raise ValueError(f"qp infeaslbie! QP:{sol['status']}")
            direction = jnp.asarray(sol["x"]).flatten()
            dual_eq, dual_ineq = jnp.asarray(sol["y"]), jnp.asarray(sol["z"])
            dual_sol = jnp.vstack([dual_eq, dual_ineq]).flatten()
            
            # linesearch and update
            alpha = self.backtrack(state, direction)
            x, m, sigma = self.update(state, alpha, direction, dual_sol)
            state = self.eval_fn(x, m, sigma)
            if verbose:
                print(f"{i}: grad:{self.max_grad(state):.4f} | viol:{self.max_viol(state):.4f} | alpha:{alpha:.4f}")
        print("SQP solved !")
        return state.x
    
    def update(self, state:OptState, alpha, direction, dual_sol):
        x = state.x + alpha * direction
        m = state.m * (1-alpha) + alpha * dual_sol
        sigma = jnp.linalg.norm(jnp.hstack([1.01*dual_sol, state.sigma]), jnp.inf)
        return x, m, sigma
    
    @classmethod
    def from_builder(cls, nlp:NLPBuilder):
        lag_fn = nlp.get_lagrangian_fn()
        f_fn = nlp.f
        g_fn = nlp.get_g_fn()
        h_fn = nlp.get_h_fn()
        merit_fn = nlp.get_merit_fn()
        merit_direc_deriv_fn = nlp.get_merit_direc_deriv_fn()
        @jax.jit
        def eval_fn(x, m, sigma):
            lag_grad = jax.grad(lag_fn)(x, m)
            lag_hess = jax.hessian(lag_fn)(x, m)
            lag_hess = regularized_hess(lag_hess)
            f_val, f_grad = jax.value_and_grad(f_fn)(x)
            g_val, g_jac = value_and_jacrev(x, f=g_fn)
            h_val, h_jac = value_and_jacrev(x, f=h_fn)
            return OptState(x, m, sigma, lag_grad, lag_hess, f_val, f_grad,
                            g_val, g_jac, h_val, h_jac)
        @jax.jit
        def armijo_cond_fn(x, direction, alpha, gamma, sigma):
            curr_merit = merit_fn(x, sigma)
            next_merit = merit_fn(x+alpha*direction, sigma)
            merit_direc_deriv = merit_direc_deriv_fn(x, direction, sigma)
            armijo = gamma * alpha * merit_direc_deriv
            return next_merit < curr_merit + armijo
        return cls(
            nlp.dim, nlp.const_dim,
            eval_fn, armijo_cond_fn
        )
        

    # def solve(self, x0, lb, ub, cl, cu, opt:Dict):
        
    #     solver = cyipopt.Problem(
    #         n=self.prob.n, m=self.prob.m,
    #         problem_obj=self.prob,
    #         lb=lb, ub=ub, cl=cl, cu=cu
    #     )
    #     for name, val in opt.items():
    #         solver.add_option(name, val)
    #     return solver.solve(x0)

    # def build(self, f:Callable, g:List[Callable]):
    #     x_rand = jnp.array(np.random.random(self.dim))
    #     fns = self.parse_functions(f, g)
    #     const_dim = len(fns['constraints'](x_rand))
    #     lambda_rand = jnp.array(np.random.random(const_dim))
    #     jitted_fns = self.jit(fns, x_rand, lambda_rand)
    #     self.prob = self.get_prob_obj(jitted_fns)
    
    # def parse_functions(self, f:Callable, g:List[Callable]):
    #     g_concat = lambda x: jnp.hstack([fn(x) for fn in g])
    #     x_rand = jnp.array(np.random.random(self.dim))
    #     const_dim = len(g_concat(x_rand))
    #     lambda_rand = jnp.array(np.random.random(const_dim))

    #     fns = {}
    #     fns['objective'] = f
    #     fns['constraints'] = g_concat
    #     fns['gradient'] = jax.grad(f)
    #     fns['obj_hessian'] = jax.hessian(f)
    #     fns['const_jacobian'] = jax.jacrev(g_concat)
    #     fns['const_hessian'] = jax.hessian(g_concat)
    #     def _hessian(x, lagrange, obj_factor):
    #         H = obj_factor * fns['obj_hessian'](x)
    #         H += jnp.einsum('i,ijk->jk', lagrange, fns['const_hessian'](x))
    #         return H
    #     jac_row, jac_col = jnp.nonzero(fns['const_jacobian'](x_rand))
    #     fns['jacobianstructure'] = lambda : (jac_row, jac_col)
    #     hess_row, hess_col = jnp.nonzero(jnp.tril(_hessian(x_rand, lambda_rand, 1.)))
    #     fns['hessianstructure'] = lambda : (hess_row, hess_col)
    #     def jacobian(x):
    #         J = fns['const_jacobian'](x)
    #         row, col = fns['jacobianstructure']()
    #         return J[row, col]
    #     def hessian(x, lagrange, obj_factor):
    #         H = _hessian(x, lagrange, obj_factor)
    #         row, col = fns['hessianstructure']()
    #         return H[row, col]
    #     fns['jacobian'] = jacobian
    #     fns['hessian'] = hessian
    #     return fns

    # def jit(self, fns, x_sample, lambda_sample):
    #     jitted_fns = {}
    #     for name, fn in fns.items():
    #         if name in ['objective', 'constraints', 'gradient', 'jacobian']:
    #             jitted_fns[name] = jax.jit(fn).lower(x_sample).compile()
    #         elif 'structure' in name:
    #             jitted_fns[name] = jax.jit(fn).lower().compile()
    #         elif name in ['hessian']:
    #             jitted_fns[name] = jax.jit(fn).lower(x_sample, lambda_sample, 1.).compile()
    #     return jitted_fns

    # def get_prob_obj(self, jitted_fns:Dict[str,Callable]):
    #     class Prob:
    #         pass
    #     prob = Prob()
    #     for name, fn in jitted_fns.items():
    #         setattr(prob, name, fn)
    #     x_rand = jnp.zeros(self.dim)
    #     prob.n = self.dim
    #     prob.m = len(jitted_fns['constraints'](x_rand))
    #     return prob
    