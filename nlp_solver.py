import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Dict, Callable
import cyipopt

class NLPSolver:
    def __init__(self, dim):
        self.dim = dim
        self.prob = None
        self.solver = None
        
    def solve(self, x0, lb, ub, cl, cu, opt:Dict):
        solver = cyipopt.Problem(
            n=self.prob.n, m=self.prob.m,
            problem_obj=self.prob,
            lb=lb, ub=ub, cl=cl, cu=cu
        )
        for name, val in opt.items():
            solver.add_option(name, val)
        return solver.solve(x0)

    def build(self, f:Callable, g:List[Callable]):
        x_rand = jnp.array(np.random.random(self.dim))
        fns = self.parse_functions(f, g)
        const_dim = len(fns['constraints'](x_rand))
        lambda_rand = jnp.array(np.random.random(const_dim))
        jitted_fns = self.jit(fns, x_rand, lambda_rand)
        self.prob = self.get_prob_obj(jitted_fns)
    
    def parse_functions(self, f:Callable, g:List[Callable]):
        g_concat = lambda x: jnp.hstack([fn(x) for fn in g])
        x_rand = jnp.array(np.random.random(self.dim))
        const_dim = len(g_concat(x_rand))
        lambda_rand = jnp.array(np.random.random(const_dim))

        fns = {}
        fns['objective'] = f
        fns['constraints'] = g_concat
        fns['gradient'] = jax.grad(f)
        fns['obj_hessian'] = jax.hessian(f)
        fns['const_jacobian'] = jax.jacrev(g_concat)
        fns['const_hessian'] = jax.hessian(g_concat)
        def _hessian(x, lagrange, obj_factor):
            H = obj_factor * fns['obj_hessian'](x)
            H += jnp.einsum('i,ijk->jk', lagrange, fns['const_hessian'](x))
            return H
        jac_row, jac_col = jnp.nonzero(fns['const_jacobian'](x_rand))
        fns['jacobianstructure'] = lambda : (jac_row, jac_col)
        hess_row, hess_col = jnp.nonzero(jnp.tril(_hessian(x_rand, lambda_rand, 1.)))
        fns['hessianstructure'] = lambda : (hess_row, hess_col)
        def jacobian(x):
            J = fns['const_jacobian'](x)
            row, col = fns['jacobianstructure']()
            return J[row, col]
        def hessian(x, lagrange, obj_factor):
            H = _hessian(x, lagrange, obj_factor)
            row, col = fns['hessianstructure']()
            return H[row, col]
        fns['jacobian'] = jacobian
        fns['hessian'] = hessian
        return fns

    def jit(self, fns, x_sample, lambda_sample):
        jitted_fns = {}
        for name, fn in fns.items():
            if name in ['objective', 'constraints', 'gradient', 'jacobian']:
                jitted_fns[name] = jax.jit(fn).lower(x_sample).compile()
            elif 'structure' in name:
                jitted_fns[name] = jax.jit(fn).lower().compile()
            elif name in ['hessian']:
                jitted_fns[name] = jax.jit(fn).lower(x_sample, lambda_sample, 1.).compile()
        return jitted_fns

    def get_prob_obj(self, jitted_fns:Dict[str,Callable]):
        class Prob:
            pass
        prob = Prob()
        for name, fn in jitted_fns.items():
            setattr(prob, name, fn)
        x_rand = jnp.zeros(self.dim)
        prob.n = self.dim
        prob.m = len(jitted_fns['constraints'](x_rand))
        return prob
    