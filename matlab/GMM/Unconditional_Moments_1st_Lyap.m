function unconditionalmoments= Unconditional_Moments_1st_Lyap(dr,options_,lags)
% Calculates the unconditional first and second moments based on the the pruning
% scheme for a DSGE model approximated up to second order. 
%    The system reads
%     xf(t+1) = hx*xf(t) + sig*eta*eps(t);
%     y(t+1)  = gx*(xf(t+1))
%
% IMPORTANT: Variances are computed by solving the Lyapunov system
%
% INPUTS:
% dr :                Dynare decision rules structure
% options_   :        Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
% lags:               lags for autocorrelation
%
% OUTPUTS: 
% unconditionalmoments:     structure storing the centered and uncentered
%                           unconditional moments, including:

% E_y          : mean of y
% Var_xf       : variance of xf
% Var_x        : variance of x
% Var_y        : variance of y
% Cov_xf       : the auto-covariances for xf, dimension nx*nx*autolag
% Cov_x        : the auto-covariances for x,  dimension nx*nx*autolag
% Cov_y        : the auto-covariances for y,  dimension ny*ny*autolag
% E_yy         : uncentered contemporaneous second moments
% autoE_yy     : uncentered lagged second moments
% By M. Andreasen, 2010 and Johannes Pfeifer, 2013


autolag=max(lags);
% The dimensions
ny = size(dr.g_v,1);
[nx,ne] = size(dr.eta);
sigeta  = dr.sig*dr.eta;

% The auxiliary system z=[xf]
% The loading matrix
A = [dr.h_v];
% The constant term
c = [zeros(nx,1)];

% the mean value of the states in the auxiliary system
E_z     = (eye(nx)-A)\c;
E_xf    = E_z(1:nx,1);

% The size of the variance of the innovations
Var_inov = zeros(nx,nx); 

% We compute Var_inov_11
Var_inov(1:nx,1:nx) = sigeta*sigeta';

% Computing the variances of the state variables, using the Lyaponov routine

[Var_z]     = dlyap_doubling(A,Var_inov);

% Var_z = lyapunov_symm(A,Var_inov,1e-10,1e-15,3); %use Dynare function
% Var_z       = dlyap_doubling(A,Var_inov);
Var_xf      = Var_z(1:nx,1:nx);
Var_x       = Var_xf;

% Computing the auto-corelations 
Cov_z       = zeros(size(Var_z,1),size(Var_z,2),autolag);
for i=1:autolag
    Cov_z(:,:,i)  = A^i*Var_z;
end
Cov_xf = Cov_z(1:nx,1:nx,:);
Cov_x  = Cov_xf;

% Moments of y: Recall y = C*z+d

C          = [dr.g_v];
d          = zeros(ny,1);

E_y        = dr.g_s0+C*E_z+d; %adds steady state
Var_y      = C*Var_z*C';
Cov_y      = zeros(ny,ny,autolag);
for i=1:autolag
    Cov_y(:,:,i) = C*Cov_z(:,:,i)*C';
end

unconditionalmoments.E_y=E_y;
unconditionalmoments.E_v=dr.h_s0+E_xf;
unconditionalmoments.E_vf=E_xf;
unconditionalmoments.Var_y=Var_y;
unconditionalmoments.autoCov_y=Cov_y;
unconditionalmoments.Var_vf=Var_xf;
unconditionalmoments.Var_v=Var_x;
unconditionalmoments.Cov_vf=Cov_xf;
unconditionalmoments.autoCov_v=Cov_x;

%% compute uncentered moments for GMM
unconditionalmoments.E_yy=Var_y+E_y*E_y';
unconditionalmoments.autoE_yy=Cov_y+repmat(E_y*E_y',[1 1 autolag]);

