import casadi as cs
import jax.numpy as jnp
import jax

class JaxFn(cs.Callback):
    def __init__(self, x_dim, y_dim, fn, inames=None, onames=None, opts={}):
        """
        t_in: list of inputs (tensorflow placeholders)
        t_out: list of outputs (tensors dependeant on those placeholders)
        session: a tensorflow session
        """
        cs.Callback.__init__(self)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.inames = inames
        self.onames = onames
        self.construct("JaxFn", opts)
        self.fn = jax.jit(fn)
        self.jit_fn = jax.jit(fn).lower(jnp.zeros(x_dim)).compile()
        self.refs = []

    def get_n_in(self): return 1
    def get_n_out(self): return 1
    def get_name_in(self, i):
        if self.inames is None:
            return f"i{i}"
        else:
            return self.inames[i]
    def get_name_out(self, i):
        if self.onames is None:
            return f"o{i}"
        else:
            return self.onames[i]

    def get_sparsity_in(self,i):
        return cs.Sparsity.dense(self.x_dim, 1)

    def get_sparsity_out(self,i):
        return cs.Sparsity.dense(self.y_dim, 1)

    def eval(self, arg):
        x = jnp.asarray(arg[0]).flatten()
        ret = self.jit_fn(x).flatten()
        return [cs.reshape(cs.vertcat(ret), self.y_dim, 1)]

    def has_jacobian(self): return True
    def get_jacobian(self, name, inames, onames, opts):
        jac_fn = jax.jit(jax.jacrev(self.fn)).lower(jnp.zeros(self.x_dim)).compile()
        class JacFun(cs.Callback):
            def __init__(self, dim_in, dim_out, fn, opts={}):
                self.dim_in = dim_in
                self.dim_out = dim_out
                self.dim_jac = (self.dim_out, self.dim_in)
                self.fn = fn
                cs.Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 2
            def get_n_out(self): return 1

            def get_sparsity_in(self,i):
                if i==0: # nominal input
                    return cs.Sparsity.dense(self.dim_in, 1)
                elif i==1: # nominal output
                    return cs.Sparsity.dense(self.dim_out, 1)

            def get_sparsity_out(self,i):
                return cs.Sparsity.dense(self.dim_jac)
            
            def eval(self, arg):
                x = jnp.asarray(arg[0]).flatten()
                ret = self.fn(x).flatten()
                return [cs.reshape(cs.vertcat(ret), *self.dim_jac)]
            
        self.jac_callback = JacFun(self.x_dim, self.y_dim, jac_fn)
        return self.jac_callback



# fn = lambda x: x[0]*x[3]*jnp.sum(x[:3]) + x[2]
# #fns = lambda x: jnp.vstack([[fn(x), fn(x)],[fn(x), fn(x)]])
# x0 = cs.DM([1., 5., 5., 1.])
# x = cs.MX.sym("x",4)
# f = JaxFn((4,1), (1,1), fn)
# J = cs.Function('J',[x],[cs.jacobian(f(x),x)])
# print(J(x0))