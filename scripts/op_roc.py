import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scripts.fitting import OptimizationProblem, FitParameter
from scripts.models import vectorborne
from scripts.simulator import Simulator

DATA_PATH = Path(__file__).parent.parent / "data"
RESULTS_PATH = Path(__file__).parent.parent / "results"


## Create Dataset ##
# Read data
filename = "00_PfPR_table_Global_admin0_2000-2017.csv"
filepath = DATA_PATH / filename
df = pd.read_csv(filepath, sep=',')

# filter data for 'Congo'
df = df[df["Name_0"] == "Congo"]

# Create data columns
df["n_cases"] = df["PAR"] * df["PfPR_rmean"]
df["time"] = (df["Year"] - df["Year"].min())


## Create simulator ##
# set model and initial values and parameters
model = vectorborne
Nh = 5.2E6
i_h = 1E6
i_v = 100000


# not important, fitting will be done later!
biting_rate = 0.3
infection_rate_h = 0.4
recovery_rate_h = 0.2
infection_rate_v = 0.2
birth_rate_v = 0.3
mortality_rate_v = 0.4

y0 = [i_h, i_v]
p = [biting_rate,
     infection_rate_h,
     recovery_rate_h,
     infection_rate_v,
     birth_rate_v,
     mortality_rate_v
     ]

# Create simulator
sim = Simulator(model=model, y0=y0, args=(Nh,), parameters=p, integrator="LSODA")

fit_parameters = [
    FitParameter(
        pid="biting_rate", initial_value=0.1,
        lower_bound=1E-6, upper_bound=1
    ),
    FitParameter(
        pid="infection_rate_h", initial_value=0.003,
        lower_bound=1E-8, upper_bound=1
    ),
    FitParameter(
        pid="recovery_rate_h", initial_value=0.1,
        lower_bound=1E-8, upper_bound=1
    ),
    FitParameter(
        pid="infection_rate_v", initial_value=0.001,
        lower_bound=1E-8, upper_bound=1
    ),
    FitParameter(
        pid="birth_rate_v", initial_value=0.01,
        lower_bound=1E-6, upper_bound=1E4
    ),
    FitParameter(
        pid="mortality_rate_v", initial_value=0.001,
        lower_bound=1E-12, upper_bound=1
    ),
]

op = OptimizationProblem(opid="congo_fit", sim=sim, parameters=fit_parameters, data=df)
p = op.fitting(size=1, results_path=RESULTS_PATH)

# plot fit
t, y = sim.integrate(t=20, parameters=p)
plt.plot(t, y[0], label=r"Prediction of $I_h$", color="black")
plt.plot(t, y[1], label=r"Prediction of $I_v$", color="blue")
# plt.xticks(t,[str(2000+x) for x in range(1,18)])
plt.plot(df.time, df.n_cases, label="Reference", color="red")
plt.ticklabel_format(axis='y',style='sci',scilimits=(6,6))
# plt.title('Fit for the Republic of Congo')
plt.xlim(0,17)
plt.xticks([x for x in range(18)],['\''+str(2000+x)[-2:] for x in range(18)])
plt.xlabel('Year')
plt.ylabel('Number of individuals')
plt.legend()
plt.show()
