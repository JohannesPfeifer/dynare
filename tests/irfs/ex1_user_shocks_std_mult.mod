/*
 * Example 1 from F. Collard (2001): "Stochastic simulations with DYNARE:
 * A practical guide" (see "guide.pdf" in the documentation directory).
 */

/*
 * Copyright (C) 2001-2010 Dynare Team
 *
 * This file is part of Dynare.
 *
 * Dynare is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Dynare is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Dynare.  If not, see <http://www.gnu.org/licenses/>.
 */


var y, c, k, a, h, b;
varexo e, u;

parameters beta, rho, alpha, delta, theta, psi, tau;

alpha = 0.36;
rho   = 0.95;
tau   = 0.025;
beta  = 0.99;
delta = 0.025;
psi   = 0;
theta = 2.95;

phi   = 0.1;

model;
c*theta*h^(1+psi)=(1-alpha)*y;
k = beta*(((exp(b)*c)/(exp(b(+1))*c(+1)))
    *(exp(b(+1))*alpha*y(+1)+(1-delta)*k));
y = exp(a)*(k(-1)^alpha)*(h^(1-alpha));
k = exp(b)*(y-c)+(1-delta)*k(-1);
a = rho*a(-1)+tau*b(-1) + e;
b = tau*a(-1)+rho*b(-1) + u;
end;

initval;
y = 1.08068253095672;
c = 0.80359242014163;
h = 0.29175631001732;
k = 11.08360443260358;
a = 0;
b = 0;
e = 0;
u = 0;
end;


//*************************************************************************
//************ Test user specified shocks with vector [1 1 0; 1 0 1], specified in terms of standard devations
//*************************************************************************


//*******************set variances to compute baseline IRFs

shocks;
var e; stderr 0.018;
var u; stderr 0.018;
var e, u = phi*0.018*0.018;
end;

options_.irf_opt.irf_shocks=[];
options_.irf_opt.percent=0;
stoch_simul(order=1,nomoments,noprint) y c k a b;
baseline_irf=[y_e c_e k_e a_e b_e y_u c_u k_u a_u b_u];
baseline_irf_old=[baseline_irf(:,1:5)+baseline_irf(:,6:10) baseline_irf];


//*******************set user specified shocks as multiple of standard deviations
generate_irfs;
e, e=2,u=0;
u, u=2,e=0;
e_u,e=2,u=2;
end;

//*******************reset variances to baseline 

shocks;
var e; stderr 0.009;
var u; stderr 0.009;
var e, u = phi*0.009*0.009;
end;

//*******************


stoch_simul(order=1,nomoments,noprint,stderr_multiples) y c k a b;
baseline_irf=[y_e_u c_e_u k_e_u a_e_u b_e_u y_e c_e k_e a_e b_e y_u c_u k_u a_u b_u];

if max(max(abs(baseline_irf-baseline_irf_old)))>1e-8
    error('Standard error multiple for standard IRFs wrong')
else 
    fprintf('\n Standard error multiple for standard IRFs at order 1 passed. Difference: %3.2e \n',max(max(abs(baseline_irf-baseline_irf_old))));
end

//****EM IRFs
stoch_simul(order=1,nomoments,noprint,EMAS_GIRF) y c k a b;
EM_irf=[y_e_u c_e_u k_e_u a_e_u b_e_u y_e c_e k_e a_e b_e y_u c_u k_u a_u b_u];

if max(max(abs(baseline_irf-EM_irf)))>1e-8
    error('Standard error multiple for EM IRFs wrong')
else 
    fprintf('\nStandard error multiple for EM IRFs at order 1 passed. Difference: %3.2e \n',max(max(abs(baseline_irf-EM_irf))));
    clear EM_irf
end

//********GIRFs
options_.irf_opt.ergodic_mean_irf=0;

stoch_simul(order=1,nomoments,noprint,analytical_GIRF) y c k a b;
G_irf=[y_e_u c_e_u k_e_u a_e_u b_e_u y_e c_e k_e a_e b_e y_u c_u k_u a_u b_u];

if max(max(abs(baseline_irf-G_irf)))>1e-8
    error('Standard error multiple for GIRFs wrong')
else 
    fprintf('\nStandard error multiple for GIRFs passed. Difference: %3.2e \n',max(max(abs(baseline_irf-G_irf))));
    clear G_irf
end