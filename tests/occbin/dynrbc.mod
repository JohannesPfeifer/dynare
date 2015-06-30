addpath /home/michel/dynare/git/master/matlab/occbin
// variables
var a, c, i, k, lambdak;
 
// innovations to shock processes
varexo erra;


// parameters
parameters ALPHA, DELTAK, BETA, GAMMAC, RHOA, PHII, PHIK, PSI, PSINEG, ISS, KSS, imin ;

BETA=0.96;
ALPHA=0.33;
DELTAK=0.10;
GAMMAC=2;
RHOA = 0.9;
PHII = 0.975;
PHIK = 0.0;
PSI = 0;        % adjustment cost for capital
PSINEG = 0;     % adjustment cost if investment is negative

model;

# zkss = ((1/BETA-1+DELTAK)/ALPHA)^(1/(ALPHA-1));
# zcss = -DELTAK*zkss + zkss^ALPHA;
# ziss = DELTAK*zkss;
# zuss = (zcss^(1-GAMMAC)-1)/(1-GAMMAC);
# zvss = zuss/(1-BETA);

/////////////////////////////////////////////////////////////////
// 1.
-exp(c)^(-GAMMAC)*(1+2*PSI*(exp(k)/exp(k(-1))-1)/exp(k(-1)))
+ BETA*exp(c(1))^(-GAMMAC)*((1-DELTAK)-2*PSI*(exp(k(1))/exp(k)-1)*
  (-exp(k(1))/exp(k)^2)+ALPHA*exp(a(1))*exp(k)^(ALPHA-1))= 
  -lambdak+BETA*(1-DELTAK+PHIK)*lambdak(1);

// 2a.
[OCCBIN = 'i >= imin']
lambdak = 0;

// 2b
[OCCBIN = 'lambdak > 0']
i = imin;

// 3.
exp(c)+exp(k)-(1-DELTAK)*exp(k(-1))+
PSI*(exp(k)/exp(k(-1))-1)^2=exp(a)*exp(k(-1))^(ALPHA);

// 4.
exp(i) = exp(k)-(1-DELTAK)*exp(k(-1));

// 5. 
a = RHOA*a(-1)+erra;

end;

steady_state_model;
a = 0;
k = log(((1/BETA-1+DELTAK)/ALPHA)^(1/(ALPHA-1)));
c = log(-DELTAK*exp(k) + exp(k)^ALPHA);
i = log(DELTAK*exp(k));
imin = log(PHII) + i;
lambdak = 0;
kprev = k;
end;

shocks;
  var erra; stderr 0.015;
end;


steady(nocheck);


[zdatalinear_ zdatapiecewise_ zdatass_ oobase_ ] = ...
    solve_one_constraint(M_,oo_,options_);

return;

perfect_foresight_setup(periods=100);
perfect_foresight_solver(occbin);



