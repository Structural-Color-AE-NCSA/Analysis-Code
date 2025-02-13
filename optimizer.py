# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:13:45 2023

@author: Diao Group

Find global minima of f(x)=x**2
"""

import matplotlib.pyplot as plt
from skopt import Optimizer
from skopt import plots
import numpy as np
import pandas as pd
# h_mu = 70

# --------------------------Whatever function we want to find min
def Objective(h_mu):
    result = abs(180-h_mu)
    return result


# -----------------------What range of params can we search over
space = [(30.0, 1000.0),  # -----------------------PrintSpeed range
         (25.0, 45.0),  # -------------------------BedTemp range
         (20.0, 35.0),  # -------------------------Pressure range
         (0.01, 5.0)  # --------------------------ZHeight range
         ]



# Pre-allocate a Pandas dataframe for function evaluations
ExpData = pd.DataFrame({
    'PrintSpeed': [],
    'BedTemp': [],
    'Pressure': [],
    'ZHeight': [],
    'Error': [],
    'ExperimentNumber': []
})

def optimizer_init():
    # ----------------------BO framework
    opt = Optimizer(dimensions=space,
                    base_estimator='gp',  # indirect kernel selection
                    n_initial_points=5,
                    initial_point_generator='random',
                    n_jobs=1,
                    acq_func='EI',  # acquisition function
                    acq_optimizer='auto',
                    random_state=None,
                    model_queue_size=None,
                    acq_func_kwargs=None,
                    acq_optimizer_kwargs=None)
    return opt

def optimizer_get(opt):
    # -----------------------------------------give guess of new params
    PrintSpeed, BedTemp, Pressure, ZHeight = opt.ask()
    return PrintSpeed, BedTemp, Pressure, ZHeight


def optimizer_tell(opt, h_mu, PrintSpeed, BedTemp, Pressure, ZHeight):
    # --------------------------------------------find value at given params
    FunctionValue = Objective(h_mu)
    # -------------------------------------------------put value into ans
    answer = opt.tell([PrintSpeed, BedTemp, Pressure, ZHeight], FunctionValue)

    return answer

    # plots.plot_evaluations(answer, bins=20,
    #                        dimensions=None,
    #                        plot_dims=None)
    # plt.show()
    #
    # plt.plot(ExpData['ExperimentNumber'],
    #          ExpData['Error'],
    #          marker='o',
    #          linestyle='-',
    #          color='k')
    # plt.title('Convergence Chart')
    # plt.xlabel('Experiment Number')
    # plt.ylabel('Color Error')
    # plt.show()
