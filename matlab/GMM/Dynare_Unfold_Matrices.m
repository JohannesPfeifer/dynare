function [dr] = Dynare_Unfold_Matrices(M_,options_,dr_)
% function [oo_] = Dynare_Unfold_Matrices(M_,options_,oo_)
% This function unfolds the matrices reported by Dynare up to a third order
% approximation in order to be linear in innovations
% Our notation is as follows
% z_t = f([x_t-1;u_t])
% where 'x'    the state variables
%       'u'    the innovations 
% and v_t = [x_t-1;u_t]
% Based on code by Martin Andreasen

% INPUTS 
%   o M_                        Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).          
%   o options_                  Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o dr_                       Matlab's decision rule structure
%  
% OUTPUTS 
%   o dr                        Decision rule structure for the linear
%                               innovation state space
%
% SPECIAL REQUIREMENTS
%   None.

% Copyright (C) 2013 Dynare Team
%
% This file is part of Dynare.
%
% Dynare is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% Dynare is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with Dynare.  If not, see <http://www.gnu.org/licenses/>.


% Dimensions
orderApp=options_.order;

nv = M_.nspred + M_.exo_nbr;
nz = M_.endo_nbr; % Number of control variables + state variables

% The cholesky factorisation of the covariance matrix

SS=M_.Sigma_e+1e-14*eye(M_.exo_nbr);

[sigma,error_mes] = chol(SS,'lower'); 
if error_mes ~= 0
    sigma = diag(sqrt(diag(M_.Sigma_e))); 
    fprintf('\nThe covariance matrix of the structural shocks is not positive definite.\n');
    fprintf('I continue the simulations using only the diagonal of the covariance matrix.\n');
end
% set singular covariance entries to 0
singular_entries=find(diag(M_.Sigma_e)==0);
if ~isempty(singular_entries)
    sigma(singular_entries,:)=0;
    sigma(:,singular_entries)=0;
end

% Steady states
f_s0 =  dr_.ys(dr_.order_var); %bring in DR order
% Decision rule for the linear approximation
fv = [dr_.ghx dr_.ghu];


% Decision rule for the 2nd order. 
% Given that g_2 are multiplied by 1/2, we multiply fvv by 2. See below.
if orderApp > 1
    if ~isfield(dr_,'g_2')
        error('GMM requires the use of k_order_solver')
    end
    f_2 = dr_.g_2;
    fss = dr_.ghs2;
    
    % we unfold the matrix f_22 and multiply by 2
    fvv = zeros(nz,nv,nv);
    index = 0;
    for alfa1=1:nv
        for alfa2=alfa1:nv
            index = index + 1;
            fvv(:,alfa1,alfa2) =  f_2(:,index);
            if alfa1 ~= alfa2
                fvv(:,alfa2,alfa1) =  f_2(:,index);
            end
        end
    end
    fvv = 2*fvv;
else
    fss = zeros(nz,1);
    fvv = zeros(nz,nv,nv);
end

% The third order terms are given by f_33 and by the difference between 
% f_31 and f_11. We multiply by 6 because the terms in Dynare++ are already
% multiplied by 1/6 (see below)
if orderApp > 2
    f_3  = dr_.g_3;  
    fssv = [dr_.ghxss dr_.ghuss];                                       
    % We unfold the matrix f_33 and multiply by 6
    fvvv = zeros(nz,nv,nv,nv);
    index = 0;
    for alfa1=1:nv
        for alfa2=alfa1:nv
            for alfa3=alfa2:nv
                index = index + 1;
                fvvv(:,alfa1,alfa2,alfa3) = f_3(:,index);
                % Using symmetry for alfa1 and alfa2
                if alfa1 == alfa2 && alfa2 ~= alfa3 %alfa1==alfa2~=alfa3
                    fvvv(:,alfa1,alfa3,alfa1) = fvvv(:,alfa1,alfa2,alfa3);
                    fvvv(:,alfa3,alfa1,alfa1) = fvvv(:,alfa1,alfa2,alfa3);                
                end
                % Using symmetry for alfa2 and alfa3            
                if alfa1 ~= alfa2 && alfa2 == alfa3  %alfa1~=alfa2==alfa3              
                    fvvv(:,alfa2,alfa1,alfa2) = fvvv(:,alfa1,alfa2,alfa3);                
                    fvvv(:,alfa2,alfa2,alfa1) = fvvv(:,alfa1,alfa2,alfa3);                                
                end            
                % Using symmetry for alfa1,alfa2, and alfa3            
                if alfa1 ~= alfa2 && alfa1 ~= alfa3 &&  alfa2 ~= alfa3 %alfa1~=alfa2~=alfa3
                    fvvv(:,alfa1,alfa3,alfa2) = fvvv(:,alfa1,alfa2,alfa3);
                    fvvv(:,alfa3,alfa1,alfa2) = fvvv(:,alfa1,alfa2,alfa3);
                    fvvv(:,alfa3,alfa2,alfa1) = fvvv(:,alfa1,alfa2,alfa3);
                    fvvv(:,alfa2,alfa3,alfa1) = fvvv(:,alfa1,alfa2,alfa3); 
                    fvvv(:,alfa2,alfa1,alfa3) = fvvv(:,alfa1,alfa2,alfa3);                 
                end
            end
        end
    end
    fvvv = 6*fvvv;
    fsss = zeros(nz,1);    %Dynare++ solves only for Gaussian shocks
else
    fsss = zeros(nz,1);
    fssv = zeros(nz,nv);
    fvvv = zeros(nz,nv,nv,nv);
end

dr.sig=1;
dr.f_s0=f_s0;
dr.f_v=fv;
dr.f_vv=fvv;
dr.f_ss=fss;
dr.f_vvv=fvvv;
dr.f_ssv=fssv;
dr.f_sss=fsss;
dr.sigma=sigma;
dr.order_var=dr_.order_var;
dr.inv_order_var=dr_.inv_order_var;
end

