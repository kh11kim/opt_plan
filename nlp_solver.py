import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
from typing import List, Dict, Tuple, Callable
import jax_dataclasses as jdc
from nlp_builder import NLPBuilder
from cvxopt import matrix, sparse, solvers, spmatrix
from functools import partial
from jax.experimental.sparse import BCOO as jsparse

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

@jdc.pytree_dataclass
class History:
    alpha: Array
    direction: Array

def value_and_jacrev(x, f):
    y, pullback = jax.vjp(f, x)
    basis = jnp.eye(y.size, dtype=y.dtype)
    jac = jax.vmap(pullback)(basis)[0]
    return y, jac

def value_and_jacrev_sparse(x, f, row, col):
    y, pullback = jax.vjp(f, x)
    basis = jnp.eye(y.size, dtype=y.dtype)
    jac = jax.vmap(pullback)(basis)[0]
    return y, jac[row, col]

def regularized_hess(P):
    def regularization(P):
        eigs, vecs = jnp.linalg.eigh(P)
        delta = 1e-5
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
        eq_dim: int,
        ineq_dim: int,
        f_fn: Callable,
        g_fn: Callable,
        h_fn: Callable,
        g_jac_sparsity: Array=None,
        h_jac_sparsity: Array=None
    ):
        self.dim = dim
        self.eq_dim = eq_dim
        self.ineq_dim = ineq_dim
        self.const_dim = eq_dim + ineq_dim
        self.states = []
        self.history = []

        self.f_fn = f_fn
        self.g_fn = g_fn
        self.h_fn = h_fn
        self.lag_fn = lambda x, m: \
            self.f_fn(x) + jnp.hstack([self.g_fn(x), self.h_fn(x)]) @ m
        self.f_val_grad_fn = jax.value_and_grad(self.f_fn)
        if g_jac_sparsity is None or h_jac_sparsity is None:
            self.sparse = False
            self.g_val_jac_fn = partial(value_and_jacrev, f=self.g_fn)
            self.h_val_jac_fn = partial(value_and_jacrev, f=self.h_fn)
        else:
            self.sparse = True
            self.g_jac_sparsity = g_jac_sparsity
            self.h_jac_sparsity = h_jac_sparsity
            self.g_jac_sparsity_rowcol = jnp.nonzero(g_jac_sparsity)
            self.h_jac_sparsity_rowcol = jnp.nonzero(h_jac_sparsity)
            self.g_val_jac_fn = partial(value_and_jacrev_sparse, 
                       f=self.g_fn, 
                       row=self.g_jac_sparsity_rowcol[0],
                       col=self.g_jac_sparsity_rowcol[1])
            self.h_val_jac_fn = partial(value_and_jacrev_sparse, 
                       f=self.h_fn, 
                       row=self.h_jac_sparsity_rowcol[0],
                       col=self.h_jac_sparsity_rowcol[1])
        
    def prebuild(self, mode="exact"):
        def eval_fn(x, m):
            f_val, f_grad = self.f_val_grad_fn(x)
            g_val, g_jac = self.g_val_jac_fn(x)
            h_val, h_jac = self.h_val_jac_fn(x)
            lag_grad = jax.grad(self.lag_fn)(x, m)
            return f_val, f_grad, g_val, g_jac, h_val, h_jac, lag_grad

        def eval_fn_exact(x, m, sigma, state_prev:OptState):
            f_val, f_grad, g_val, g_jac, h_val, h_jac, lag_grad = eval_fn(x, m)
            lag_hess = jax.hessian(self.lag_fn)(x, m)
            lag_hess = regularized_hess(lag_hess)
            return OptState(x, m, sigma, 
                            lag_grad, lag_hess, f_val, f_grad,
                            g_val, g_jac, h_val, h_jac)
        
        def eval_fn_BFGS(x, m, sigma, state_prev:OptState):
            f_val, f_grad, g_val, g_jac, h_val, h_jac, lag_grad = eval_fn(x, m)
            B_prev = state_prev.lag_hess
            s = x - state_prev.x
            y = lag_grad - state_prev.lag_grad
            den1 = s @ B_prev @ s
            den2 = jnp.inner(y, s)
            # term1 = B_prev @ jnp.outer(s, s) @ B_prev / den1
            # term2 = jnp.outer(y, y) / den2

            # powell's trick (preserving positive-definiteness)
            powell_cond = den2 >= 0.2 * den1
            true_fn = lambda den1, den2: 1.
            false_fn = lambda den1, den2: 0.8 * den1 / (den1 - den2)
            theta = jax.lax.cond(powell_cond, true_fn, false_fn, den1, den2)
            r = theta*y + (1 - theta) * B_prev@s
            term1 = B_prev @ jnp.outer(s, s) @ B_prev / den1
            term2 = jnp.outer(r,r)/jnp.inner(s, r)
            lag_hess = B_prev - term1 + term2
            return OptState(x, m, sigma, 
                            lag_grad, lag_hess, f_val, f_grad,
                            g_val, g_jac, h_val, h_jac)
        
        state_init_fn_exact = partial(eval_fn_exact, state_prev=None)
        def state_init_fn_BFGS(x, m, sigma):
            f_val, f_grad, g_val, g_jac, h_val, h_jac, lag_grad = eval_fn(x, m)
            lag_hess = jnp.eye(x.shape[0])
            return OptState(x, m, sigma, 
                            lag_grad, lag_hess, f_val, f_grad,
                            g_val, g_jac, h_val, h_jac)
        
        def merit_fn(x, sigma=0.):
            eq_norm = jnp.linalg.norm(self.g_fn(x), 1)
            ineq_norm = jnp.linalg.norm(jnp.clip(self.h_fn(x), a_min=0.), 1)
            return self.f_fn(x) + sigma * (eq_norm + ineq_norm)
        def merit_direc_deriv_fn(x, direction, sigma):
            direct_deriv = jax.grad(self.f_fn)(x) @ direction
            eq_1norm = jnp.linalg.norm(self.g_fn(x), 1)
            ineq_1norm = jnp.linalg.norm(jnp.clip(self.h_fn(x),a_min=0.), 1)
            return direct_deriv - sigma * (eq_1norm + ineq_1norm)
        def armijo_cond_fn(x, direction, alpha, gamma, sigma):
            curr_merit = merit_fn(x, sigma)
            next_merit = merit_fn(x+alpha*direction, sigma)
            merit_direc_deriv = merit_direc_deriv_fn(x, direction, sigma)
            armijo = gamma * alpha * merit_direc_deriv
            return next_merit < curr_merit + armijo
        self.armijo_cond_fn = armijo_cond_fn
        
        if mode == "exact":
            self.eval_fn = eval_fn_exact
            self.state_init_fn = state_init_fn_exact
        elif mode == "BFGS":
            self.eval_fn = eval_fn_BFGS
            self.state_init_fn = state_init_fn_BFGS

        # precompile    
        for att in dir(self):
            if "fn" in att:
                fn = jax.jit(getattr(self, att))
                setattr(self, att, fn)
    
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
    
    def convexify(self, state:OptState):
        P = matrix(np.asarray(state.lag_hess).astype(np.double))
        q = matrix(np.asarray(state.f_grad).astype(np.double))
        h = matrix(-np.asarray(state.h_val).astype(np.double))
        b = matrix(-np.asarray(state.g_val).astype(np.double))
        if self.sparse:
            A = spmatrix(
                np.asarray(state.g_jac, dtype=np.double), 
                np.asarray(self.g_jac_sparsity_rowcol[0], dtype=int),
                np.asarray(self.g_jac_sparsity_rowcol[1], dtype=int),
            )
            G = spmatrix(
                np.asarray(state.h_jac, dtype=np.double), 
                np.asarray(self.h_jac_sparsity_rowcol[0], dtype=int),
                np.asarray(self.h_jac_sparsity_rowcol[1], dtype=int),
            )
        else:
            G = sparse(matrix(np.asarray(state.h_jac).astype(np.double)))
            A = sparse(matrix(np.asarray(state.g_jac).astype(np.double)))
        
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
        self.states = []

        x = x0.copy()
        m = jnp.zeros(self.const_dim)
        sigma = 0.1
        xdiff = jnp.inf #initial value
        state = self.state_init_fn(x, m, sigma)

        for i in range(max_iter):
            # Check terminal condition
            if xdiff < tol and \
               self.max_viol(state) < const_viol_tol:
                break
            elif save_history:
                self.states.append(state)

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
            state_prev = state
            x, m, sigma = self.update(state, alpha, direction, dual_sol)
            xdiff = jnp.linalg.norm(direction)
            state = self.eval_fn(x, m, sigma, state_prev)
            if verbose:
                print(f"{i}: grad:{self.max_grad(state):.4f} | viol:{self.max_viol(state):.4f} | alpha:{alpha:.4f}")
            if save_history:
                self.history.append(History(alpha, direction))
        print("SQP solved !")
        return state.x
    
    def update(self, state:OptState, alpha, direction, dual_sol):
        x = state.x + alpha * direction
        m = (1-alpha) + alpha * dual_sol
        sigma = jnp.linalg.norm(jnp.hstack([1.01*dual_sol, state.sigma]), jnp.inf)
        return x, m, sigma
    
    @classmethod
    def from_builder(cls, nlp:NLPBuilder, sparsity=True):
        if sparsity:
            return cls(
                nlp.dim, nlp.g_dim, nlp.h_dim,
                nlp.f, nlp.get_g_fn(), nlp.get_h_fn(),
                nlp.get_g_jac_sparsity(),
                nlp.get_h_jac_sparsity()
            )
        else:
            return cls(
                nlp.dim, nlp.g_dim, nlp.h_dim,
                nlp.f, nlp.get_g_fn(), nlp.get_h_fn()
            )
        # lag_fn = nlp.get_lagrangian_fn()
        # lag_grad_fn = jax.grad(lag_fn)
        # lag_hess_fn = lambda x, m: regularized_hess(jax.hessian(lag_fn)(x, m))
        # f_fn = nlp.f
        # g_fn = nlp.get_g_fn()
        # h_fn = nlp.get_h_fn()
        # f_value_and_grad_fn = jax.value_and_grad(f_fn)
        # g_value_and_jacrev_fn = partial(value_and_jacrev, f=g_fn)
        # h_value_and_jacrev_fn = partial(value_and_jacrev, f=h_fn)
        # merit_fn = nlp.get_merit_fn()
        # merit_direc_deriv_fn = nlp.get_merit_direc_deriv_fn()
        
        # def eval_fn_exact_hess(x, m, sigma, state_prev):
        #     lag_grad = lag_grad_fn(x, m)
        #     lag_hess = lag_hess_fn(x, m)
        #     f_val, f_grad = f_value_and_grad_fn(x)
        #     g_val, g_jac = g_value_and_jacrev_fn(x)
        #     h_val, h_jac = h_value_and_jacrev_fn(x)
        #     return OptState(x, m, sigma, lag_grad, lag_hess, f_val, f_grad,
        #                     g_val, g_jac, h_val, h_jac)
        
        # def eval_fn_BFGS(x, m, sigma, state_prev):
        #     lag_grad = lag_grad_fn(x, m)
        #     lag_hess = lag_hess_fn(x, m)
        #     f_val, f_grad = f_value_and_grad_fn(x)
        #     g_val, g_jac = g_value_and_jacrev_fn(x)
        #     h_val, h_jac = h_value_and_jacrev_fn(x)
        #     return OptState(x, m, sigma, lag_grad, lag_hess, f_val, f_grad,
        #                     g_val, g_jac, h_val, h_jac)
        
        # def armijo_cond_fn(x, direction, alpha, gamma, sigma):
        #     curr_merit = merit_fn(x, sigma)
        #     next_merit = merit_fn(x+alpha*direction, sigma)
        #     merit_direc_deriv = merit_direc_deriv_fn(x, direction, sigma)
        #     armijo = gamma * alpha * merit_direc_deriv
        #     return next_merit < curr_merit + armijo
        
        

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
    