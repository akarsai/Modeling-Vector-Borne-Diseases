import pandas as pd
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'figure.dpi': 200,
})
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

# filter data for 'Democratic Republic of the Congo'
df = df[df["Name_0"] == "Congo"]

# Create data columns
df["n_cases"] = df["PAR"] * df["PfPR_rmean"]
df["time"] = (df["Year"] - df["Year"].min())


## Create simulator ##
# set model and initial values and parameters
model = vectorborne
Nh  = df.PAR.mean()
i_h = df.n_cases.iloc[0]
i_v = 100000

# Optimal Parameter Values
biting_rate = 0.00036389530506576007
infection_rate_h = 0.007393896420445867
recovery_rate_h = 0.3161417377698296
infection_rate_v = 0.0011379397461297584
birth_rate_v = 21.49892889976263
mortality_rate_v = 0.0007105823099979525

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
t, y = sim.integrate(t=20, parameters=p)

# plot fit
fig, ax1 = plt.subplots()

# the humans
ax1.set_xlim(0,17)
ax1.set_xlabel('Year')
ax1.set_xticks([x for x in range(18)])
ax1.set_xticklabels(['\''+str(2000+x)[-2:] for x in range(18)])
ax1.set_ylabel('Number of humans')
ax1.ticklabel_format(axis='y',style='sci',scilimits=(6,6))
ax1.plot(t, y[0], label=r'Prediction of $I_h$', color='black')
ax1.plot(df.time, df.n_cases, label=r'Reference of $I_h$', color='black', linestyle='--')
ax1.tick_params(axis='y',labelcolor='black')
# ax1.legend()

# the vectors
ax2 = ax1.twinx()
ax2.set_ylabel('Number of vectors',color='blue')
ax2.ticklabel_format(axis='y',style='sci',scilimits=(3,3))
ax2.plot(t, y[1], label=r'Prediction of $I_v$', color='blue')
ax2.tick_params(axis='y',labelcolor='blue')
# ax2.legend()

# show
fig.legend(fancybox=True, framealpha=1, bbox_to_anchor=(0.9,0.938))
fig.tight_layout()
plt.show()
