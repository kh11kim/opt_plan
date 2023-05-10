import jax.numpy as jnp
import jax
from jax import jit, grad, jacfwd, jacrev
import cyipopt
from cyipopt import minimize_ipopt

def objective(x):
    return x[0]*x[3]*jnp.sum(x[:3]) + x[2]

def eq_constraints(x):
    return jnp.sum(x**2)

def ineq_constrains(x):
    return jnp.prod(x)

# jit the functions
obj_jit = jit(objective)
con_eq_jit = jit(eq_constraints)
con_ineq_jit = jit(ineq_constrains)

# build the derivatives and jit them
obj_grad = jit(grad(obj_jit))  # objective gradient
obj_hess = jit(jacrev(jacfwd(obj_jit))) # objective hessian
con_eq_jac = jit(jacfwd(con_eq_jit))  # jacobian
con_ineq_jac = jit(jacfwd(con_ineq_jit))  # jacobian
con_eq_hess = jacrev(jacfwd(con_eq_jit)) # hessian
con_eq_hessvp = jit(lambda x, v: con_eq_hess(x) * v[0]) # hessian vector-product
con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
con_ineq_hessvp = jit(lambda x, v: con_ineq_hess(x) * v[0]) # hessian vector-product

# constraints
cons = [
    {'type': 'eq', 'fun': con_eq_jit, 'jac': con_eq_jac, 'hess': con_eq_hessvp},
    {'type': 'ineq', 'fun': con_ineq_jit, 'jac': con_ineq_jac, 'hess': con_ineq_hessvp}
 ]

# starting point
x0 = jnp.array([1.0, 5.0, 5.0, 1.0])

# variable bounds: 1 <= x[i] <= 5
bnds = [(1, 5) for _ in range(x0.size)]

# # executing the solver
# res = minimize_ipopt(obj_jit, jac=obj_grad, hess=obj_hess, x0=x0, bounds=bnds,
#                   constraints=cons, options={'disp': 5})

@jax.jit
def constraints(x):
    return jnp.hstack([con_ineq_jit(x), con_eq_jit(x)])

const_jac = jax.jacfwd(constraints)
const_hess = jax.jacrev(const_jac)
sh = const_hess(x0).shape
print(sh)

@jax.jit
def hessian(x, lagrange, obj_factor):
    H_obj = obj_factor * obj_hess(x)
    H_const = jnp.einsum('i,ijk->jk', lagrange, const_hess(x))
    H = (H_obj + H_const)[jnp.tril_indices(x.size)]
    return H



prob_dict = dict(
    objective=obj_jit,
    gradient=obj_grad,
    constraints=constraints,
    jacobian=const_jac,
    hessian=hessian
)
class Problem:
    pass

p = Problem()
for fn_name, fn in prob_dict.items():
    setattr(p, fn_name, fn)

lb = [1.0, 1.0, 1.0, 1.0]
ub = [5.0, 5.0, 5.0, 5.0]
cl = [25.0, 40.0]
cu = [2.0e19, 40.0]

x0 = [1.0, 5.0, 5.0, 1.0]

nlp = cyipopt.Problem(
    n=len(x0),
    m=len(cl),
    problem_obj=p,
    lb=lb,ub=ub,cl=cl,cu=cu
)
x, info = nlp.solve(x0)
print(info)

# def hessian(self, x, lagrange, obj_factor):
#     H = obj_factor * self.obj_hess(x)  # type: ignore
#     # split the lagrangian multipliers for each constraint hessian
#     lagrs = np.split(lagrange, np.cumsum(self._constraint_dims[:-1]))
#     for hessian, args, lagr in zip(self._constraint_hessians,
#                                     self._constraint_args, lagrs):
#         H += hessian(x, lagr, *args)
#     return H[np.tril_indices(x.size)]
