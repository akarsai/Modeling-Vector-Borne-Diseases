import numpy as np
from typing import List
from scipy.integrate import solve_ivp

class Simulator:

    def __init__(self, model, y0, parameters, args: tuple = (), integrator: str = "RK45"):
        self.model = model
        self.y0 = y0
        self.parameters = parameters
        self.args = args
        self.integrator = integrator

    def integrate(self, t: int, t_eval=None, parameters=None, y0=None, args: tuple=None):

        if t_eval is None:
            t_eval = np.linspace(0, t, t*2)
        if parameters is None:
            parameters = self.parameters
        if y0 is None:
            y0 = self.y0
        if args is None:
            args = self.args
        args = (parameters,) + args
        solution = solve_ivp(
            fun=self.model, y0=y0, args=args,
            t_span=(0, t), t_eval=t_eval, method=self.integrator
        )

        return (solution.t, solution.y)

