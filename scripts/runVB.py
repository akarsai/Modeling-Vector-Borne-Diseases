import numpy as np
from scipy.integrate import solve_ivp
from models import vectorborneOld
import matplotlib.pyplot as plt

alphaval = 0.7

# set initial guess for y
ih = 100
iv = 1000
y0 = np.array([ih, iv])

# set initial parameters
alpha_h  = 0.02
beta_h   = 0.1
alpha_v  = 0.01
lambda_v = 0.1
mu_v     = 0.1
nh       = 10000 #see data
params = [alpha_h, beta_h, alpha_v, lambda_v, mu_v, nh]

# other parameters
tmax = 50
t = np.linspace(0, tmax, 300)
t_range = (0, tmax)
model = vectorborneOld
method = 'LSODA'

# solve
result1 = solve_ivp(model, t_range, y0, method=method, t_eval=t, args=(params,)).y


# update parameters
alpha_h  = 0.02
beta_h   = 0.8
params = [alpha_h, beta_h, alpha_v, lambda_v, mu_v, nh]

# solve
result2 = solve_ivp(model, t_range, y0, method=method, t_eval=t, args=(params,)).y



# update parameters
alpha_h  = 0.05
beta_h   = 0.1
params = [alpha_h, beta_h, alpha_v, lambda_v, mu_v, nh]

# solve
result3 = solve_ivp(model, t_range, y0, method=method, t_eval=t, args=(params,)).y



# update parameters
alpha_h  = 0.05
beta_h   = 0.8
params = [alpha_h, beta_h, alpha_v, lambda_v, mu_v, nh]

# solve
result4 = solve_ivp(model, t_range, y0, method=method, t_eval=t, args=(params,)).y



## plot simulations
fig, ax1 = plt.subplots()

# the humans
ax1.set_xlabel(r'Time $t$')
ax1.set_ylabel('Number of humans')
ax1.plot(t,result3[0],label=r'Prediction of $I_h$ with $\alpha_h=0.05,~ \beta_h = 0.1$',alpha=alphaval, color='black',linestyle='solid')
ax1.plot(t,result1[0],label=r'Prediction of $I_h$ with $\alpha_h=0.02,~ \beta_h = 0.1$',alpha=alphaval, color='black',linestyle='dashed')
ax1.plot(t,result4[0],label=r'Prediction of $I_h$ with $\alpha_h=0.05,~ \beta_h = 0.8$',alpha=alphaval, color='black',linestyle='dotted')
ax1.plot(t,result2[0],label=r'Prediction of $I_h$ with $\alpha_h=0.02,~ \beta_h = 0.8$',alpha=alphaval, color='black',linestyle='dashdot')
ax1.tick_params(axis='y',labelcolor='black')

# the vectors
ax2 = ax1.twinx()
ax2.set_ylabel('Number of vectors',color='blue')
ax2.plot(t, result4[1], label=r'Prediction of $I_v$', color='blue')
ax2.tick_params(axis='y',labelcolor='blue')

# show
fig.legend(bbox_to_anchor=(0.89,0.97))
fig.tight_layout()
plt.show()
