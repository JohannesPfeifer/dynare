function unconditionalmoments= Unconditional_Moments_2nd_Lyap(dr,options_,HigherMoments,lags)
% Calculates the unconditional first and second moments based on the the pruning
% scheme for a DSGE model approximated up to second order. 
%    The system reads
%     xf(t+1) = hx*xf(t) + sig*eta*eps(t);
%     xs(t+1) = hx*xs(t) + 0.5*reshape(hxx,nx,nx^2)*kron(xf(t),xf(t))+ 1/2*sig^2*hss;
%     y(t+1)  = gx*(xf(t+1)+xs(t+1)) + 0.5*reshape(gxx,ny,nx^2)*kron(xf(t+1),xf(t+1))+ 1/2*sig^2*gss
%
% IMPORTANT: Variances are computed by solving the Lyapunov system
%
% INPUTS:
% dr :                Dynare decision rules structure
% options_   :        Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
% HigherMoments:      Structure describing higher moments
% lags:               lags for autocorrelation
%
% OUTPUTS: 
% unconditionalmoments:     structure storing the centered and uncentered
%                           unconditional moments, including:

% E_xs         : mean of xs
% E_y          : mean of y
% E_xf_xf      : mean of xf*xf'
% Var_xf       : variance of xf (same as E_xf_xf because E_xf = 0)
% Var_xs       : variance of xs
% Var_xf_xs    : covariance between xf and xs
% Var_xfxf     : variance of kron(xf,xf)
% Var_xs_xfxf  : variance of xs*kron(xf,xf)'
% Var_x        : variance of x
% Var_y        : variance of y
% Cov_xf       : the auto-covariances for xf, dimension nx*nx*autolag
% Cov_xs       : the auto-covariances for xs, dimension nx*nx*autolag
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

if options_.order ==1
     % The auxiliary system z=[xf xs kron(xf,xf)]
    % The loading matrix
    A = [dr.h_v                ,zeros(nx,nx)       ,zeros(nx,nx^2);
        zeros(nx,nx)           ,dr.h_v             ,zeros(nx,nx*nx);
        zeros(nx*nx,nx)        ,zeros(nx*nx,nx)    ,kron(dr.h_v,dr.h_v)          ];
    % The constant term
    c = [zeros(nx,1);
         zeros(nx,1);
        kron(sigeta,sigeta)*reshape(eye(ne),ne^2,1)];
else
% The auxiliary system z=[xf xs kron(xf,xf)]
    % The loading matrix
    A = [dr.h_v                ,zeros(nx,nx)       ,zeros(nx,nx^2);
        zeros(nx,nx)           ,dr.h_v             ,0.5*reshape(dr.h_vv,nx,nx*nx);
        zeros(nx*nx,nx)        ,zeros(nx*nx,nx)    ,kron(dr.h_v,dr.h_v)          ];
    % The constant term
    c = [zeros(nx,1);
        0.5*dr.h_ss*dr.sig^2;
        kron(sigeta,sigeta)*reshape(eye(ne),ne^2,1)];
end

% the mean value of the states in the auxiliary system
E_z     = (eye(2*nx+nx^2)-A)\c;
E_xf    = E_z(1:nx,1);
E_xs    = E_z(nx+1:2*nx,1);
E_xf_xf = reshape(E_z(2*nx+1:2*nx+nx^2,1),nx,nx);


% The size of the variance of the innovations
Var_inov = zeros(2*nx+nx^2,2*nx+nx^2); 

% We compute Var_inov_11
Var_inov(1:nx,1:nx) = sigeta*sigeta';

% We compute Var_inov_13 Var_inov_31
E_eps_eps2 = zeros(ne,ne^2);
% any: 1 if any element of a vector is a nonzero number
if any(HigherMoments.vectorMom3) == 1
    for phi1=1:ne
        index = 0;
        for phi2=1:ne
            for phi3=1:ne
                index = index + 1;
                if (phi1 == phi2 && phi1 == phi3)
                    E_eps_eps2(phi1,index) = HigherMoments.vectorMom3(1,phi1);
                end
            end
        end
    end
end
Var_inov(1:nx,2*nx+1:2*nx+nx^2) = sigeta*E_eps_eps2*(kron(sigeta',sigeta')); %p. 25 above 3.3.2
Var_inov(2*nx+1:2*nx+nx^2,1:nx) = Var_inov(1:nx,2*nx+1:2*nx+nx^2)';

E_xfeps_epsxf = zeros(nx*ne,nx*ne);
for shock=1:ne
    E_xfeps_epsxf(shock:ne:end-(ne-shock),(shock-1)*nx+1:shock*nx) = E_xf_xf;   
end

E_eps2_eps2 = zeros(ne^2,ne^2);
index1 = 0;
for phi4=1:ne
    for phi1=1:ne
        index1 = index1 + 1;
        index2 = 0;
        for phi3=1:ne
            for phi2=1:ne
                index2 = index2 + 1;
                % The second moments
                if (phi1 == phi2 && phi3 == phi4 && phi1 ~= phi4)
                    E_eps2_eps2(index1,index2) = 1;
                elseif (phi1 == phi3 && phi2 == phi4 && phi1 ~= phi2)
                    E_eps2_eps2(index1,index2) = 1;
                elseif (phi1 == phi4 && phi2 == phi3 && phi1 ~= phi2)                    
                    E_eps2_eps2(index1,index2) = 1;
                elseif (phi1 == phi2 && phi1 == phi3 && phi1 == phi4)
                    % The fourth moments
                    E_eps2_eps2(index1,index2) = HigherMoments.vectorMom4(1,phi1);
                end
            end
        end
    end
end
%define used terms
kron_h_v_sigeta=kron(dr.h_v,sigeta);
kron_sigeta_h_v=kron(sigeta,dr.h_v);
kron_sigeta_sigeta=kron(sigeta,sigeta);
%compute variance of innovations
Var_inov(2*nx+1:2*nx+nx^2,2*nx+1:2*nx+nx^2) = ....
     kron_h_v_sigeta*(kron(E_xf_xf,eye(ne)))*kron_h_v_sigeta'...
    +kron_h_v_sigeta*E_xfeps_epsxf*kron_sigeta_h_v'...
    +kron_sigeta_h_v*E_xfeps_epsxf'*kron_h_v_sigeta'...
    +kron_sigeta_h_v*(kron(eye(ne),E_xf_xf))*kron_sigeta_h_v'...
    +kron_sigeta_sigeta*(E_eps2_eps2-reshape(eye(ne),ne^2,1)*reshape(eye(ne),ne^2,1)')...
    *kron_sigeta_sigeta'; %p. 28

% Computing the variances of the state variables, using the Lyaponov routine

[Var_z]     = dlyap_doubling(A,Var_inov);

% Var_z = lyapunov_symm(A,Var_inov,1e-10,1e-15,3); %use Dynare function
% Var_z       = dlyap_doubling(A,Var_inov);
Var_xf      = Var_z(1:nx,1:nx);
Var_xs      = Var_z(nx+1:2*nx,nx+1:2*nx);
Var_xf_xs   = Var_z(1:nx,nx+1:2*nx);
Var_xfxf    = Var_z(2*nx+1:2*nx+nx^2,2*nx+1:2*nx+nx^2);
Var_xs_xfxf = Var_z(nx+1:2*nx,2*nx+1:2*nx+nx^2);
Var_x       = Var_xf + Var_xs + Var_xf_xs + Var_xf_xs';

% Computing the auto-corelations 
Cov_z       = zeros(size(Var_z,1),size(Var_z,2),autolag);
for i=1:autolag
    Cov_z(:,:,i)  = A^i*Var_z;
end
Cov_xf = Cov_z(1:nx,1:nx,:);
Cov_xs = Cov_z(nx+1:2*nx,nx+1:2*nx,:);
Cov_x  = Cov_xf + Cov_xs + Cov_z(1:nx,nx+1:2*nx,:) + Cov_z(nx+1:2*nx,1:nx,:);

% Moments of y: Recall y = C*z+d
if options_.order ==1
    C          = [dr.g_v dr.g_v 0.5*zeros(ny,nx*nx)];
    d          = zeros(ny,1);
else
    C          = [dr.g_v dr.g_v 0.5*reshape(dr.g_vv,ny,nx*nx)];
    d          = 0.5*dr.g_ss*dr.sig^2;
end
E_y        = dr.g_s0+C*E_z+d; %adds steady state
Var_y      = C*Var_z*C';
Cov_y      = zeros(ny,ny,autolag);
for i=1:autolag
    Cov_y(:,:,i) = C*Cov_z(:,:,i)*C';
end

unconditionalmoments.E_y=E_y;
if options_.order ==1
    unconditionalmoments.E_v=dr.h_s0+E_xf;
else
    unconditionalmoments.E_v=dr.h_s0+E_xf+E_xs;
end
unconditionalmoments.E_vf=E_xf;
unconditionalmoments.E_vs=E_xs;
unconditionalmoments.Var_y=Var_y;
unconditionalmoments.autoCov_y=Cov_y;
unconditionalmoments.E_vf_vf=E_xf_xf;
unconditionalmoments.Var_vf=Var_xf;
unconditionalmoments.Var_vf_vs=Var_xf_xs;
unconditionalmoments.Var_vs=Var_xs;
unconditionalmoments.Var_vfvf=Var_xfxf;
unconditionalmoments.Var_vs_vfvf=Var_xs_xfxf;
unconditionalmoments.Var_v=Var_x;
unconditionalmoments.Cov_vf=Cov_xf;
unconditionalmoments.Cov_vs=Cov_xs;
unconditionalmoments.autoCov_v=Cov_x;

%% compute uncentered moments for GMM
unconditionalmoments.E_yy=Var_y+E_y*E_y';
unconditionalmoments.autoE_yy=Cov_y+repmat(E_y*E_y',[1 1 autolag]);

