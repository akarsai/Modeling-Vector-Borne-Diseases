import numpy as np
import pandas as pd
from typing import List
from scipy.optimize import least_squares

from scripts.simulator import Simulator
from scripts.helpers import timeit

class FitParameter:
    """ Class to store parameters, their bounds and initial values for fitting
    """
    def __init__(self, pid, initial_value, lower_bound, upper_bound):
        self.pid = pid
        self.initial_value = initial_value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class OptimizationProblem:

    def __init__(self, opid: str, sim: Simulator, parameters: List[FitParameter], data: pd.DataFrame, fun=None):
        self.opid = opid
        self.sim = sim
        self.parameters = parameters
        self.data = data
        if fun == None:
            self.fun = lambda p: (np.sqrt(((self.sim.integrate(parameters=p, t=20, t_eval=self.data["time"])[1][0] - self.data["n_cases"])**2).sum(axis=0)))
            # self.fun = lambda p: (np.sqrt(((self.sim.integrate(parameters=p[:-2], t=20, t_eval=self.data["time"], y0=p[-2:])[1][0] - self.data["n_cases"])**2).sum(axis=0)))
        else:
            self.fun = fun

    def _create_summaryTable(self):
        """ Create an empty pandas dataframe with columnnames
        """
        pids = [fp.pid for fp in self.parameters]
        columns = ["success", "cost"] + pids + ["status", "message"]
        return pd.DataFrame(columns=columns)

    def _print_optimalParameters(self, df):
        """ Printing best parameter-set after fitting
        """
        print("-"*80)
        print("Optimal Parameter Values")
        for fp in self.parameters:
            print(f"{fp.pid} = {df[fp.pid].iloc[0]}")
        print("-"*80)

    def _parmas_to_dict(self, p: list):
        """ Conversion of 2D-array of parameter values to dictionary
        """
        p_dict = {}
        for i in range(len(self.parameters)):
            pid = self.parameters[i].pid
            p_dict[pid] = p[i]
        return p_dict

    def _sample_parameters(self):
        """ Sampling of parameters from a log-uniform distribution within given bounds
        """
        parameters = []
        for fp in self.parameters:
            lb_log = np.log10(fp.lower_bound)
            ub_log = np.log10(fp.upper_bound)
            p = np.power(10, lb_log + np.random.rand() * (ub_log - lb_log))
            parameters.append(p)
        return parameters

    @timeit
    def _lsq(self, p, bounds):
        """ Timed non-linear least-squares fit
        """
        return least_squares(fun=self.fun, x0=p, bounds=bounds, verbose=1)

    @timeit
    def fitting(self, size: int, results_path: str):
        """ Fitting of optimization problem by non-linear least-squares method

        Initial parameter values are sampled n times from a log-uniform distribution.
        Fitting is then performed by a bounded non-linear least-squares.
        The method returns a tsv-file, containing the best parameters found in each iteration,
        their cost and the respective stop condition.
        """
        summary_path = results_path / "fit_summary.tsv"

        # create empty data frame to store outputs
        fit_summary = self._create_summaryTable()

        # set initial parameter values and bounds
        p0 = [fp.initial_value for fp in self.parameters]
        lb = [fp.lower_bound for fp in self.parameters]
        ub = [fp.upper_bound for fp in self.parameters]
        bounds = (lb, ub)

        # create dictionary to store initial parameters and the discovered local minimum
        p_initial = {}
        p_final = {}

        cost_optimal = np.inf
        p_optimal = []
        for i in range(size):
            # non-linear least-squares fit
            lsq_fit = self._lsq(p=p0, bounds=bounds)

            # store results
            summary = [lsq_fit.success, lsq_fit.cost] + list(lsq_fit.x) + [lsq_fit.status, lsq_fit.message]
            fit_summary.loc[i] = summary

            # store initial and final parameters
            p_initial[i] = self._parmas_to_dict(p0)
            p_final[i] = self._parmas_to_dict(lsq_fit.x)

            if cost_optimal > lsq_fit.cost:
                cost_optimal = lsq_fit.cost
                p_optimal = lsq_fit.x

            # sample new parameter set
            p0 = self._sample_parameters()

        # save summary
        fit_summary.sort_values(by="cost", axis=0, ascending=True, inplace=True)
        fit_summary.to_csv(summary_path, sep='\t')

        # print optimal parameters
        self._print_optimalParameters(fit_summary)

        # TODO: create plots


        return p_optimal
