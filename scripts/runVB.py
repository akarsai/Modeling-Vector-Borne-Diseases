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



# plot simulations
plt.plot(t,result3[0],label=r'$I_h$ with $\alpha_h=0.05,~ \beta_h = 0.1$',alpha=alphaval, color='black',linestyle='solid')
plt.plot(t,result1[0],label=r'$I_h$ with $\alpha_h=0.02,~ \beta_h = 0.1$',alpha=alphaval, color='black',linestyle='dashed')
plt.plot(t,result4[0],label=r'$I_h$ with $\alpha_h=0.05,~ \beta_h = 0.8$',alpha=alphaval, color='black',linestyle='dotted')
plt.plot(t,result2[0],label=r'$I_h$ with $\alpha_h=0.02,~ \beta_h = 0.8$',alpha=alphaval, color='black',linestyle='dashdot')
plt.plot(t,result4[1],label=r'$I_v$',alpha=alphaval,color='blue')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Number of individuals')
plt.legend()
plt.show()
