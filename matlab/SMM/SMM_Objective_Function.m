function [fval,moments_difference,modelMoments,exit_flag,junk1,junk2,info,Model,DynareOptions,SMMinfo,DynareResults]...
    = SMM_Objective_Function(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,SMMinfo,DynareResults)
% [fval,moments_difference,modelMoments,exit_flag,ys,trend_coeff,info,Model,DynareOptions,SMMinfo,DynareResults]...
%    = SMM_Objective_Function(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,SMMinfo,DynareResults)
% This function evaluates the objective function for SMM estimation
%
% INPUTS
%   o xparam1:                  initial value of estimated parameters as returned by set_prior()
%   o DynareDataset:            data after required transformation
%   o DynareOptions:            Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o Model                     Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).          
%   o EstimatedParameters:      Matlab's structure describing the estimated_parameters (initialized by dynare, see @ref{estim_params_}).
%   o SMMInfo                   Matlab's structure describing the SMM settings (initialized by dynare, see @ref{bayesopt_}).
%   o DynareResults             Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%
% OUTPUTS
%   o fval:                     value of the quadratic form of the moment difference
%   o moments_difference:       [numMom x 1] vector with difference of empirical and model moments
%   o modelMoments:             [numMom x 1] vector with model moments
%   o exit_flag:                0 if no error, 1 of error
%   o info:                     vector storing error code and penalty 
%   o Model:                    Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).
%   o DynareOptions:            Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o SMMinfo:                  Matlab's structure describing the SMM parameter options (initialized by dynare, see @ref{SMMinfo_}).
%   o DynareResults:            Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).

% SPECIAL REQUIREMENTS
%   none

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


global objective_function_penalty_base

% Initialization of the returned variables and others...
fval        = NaN;
exit_flag   = 1;
info        = 0;
junk2       = [];
junk1       = [];
moments_difference=NaN(SMMinfo.numMom,1);
modelMoments=NaN(SMMinfo.numMom,1);
%------------------------------------------------------------------------------
% 1. Get the structural parameters & define penalties
%------------------------------------------------------------------------------

% Return, with endogenous penalty, if some parameters are smaller than the lower bound of the parameters.
if ~isequal(DynareOptions.mode_compute,1) && any(xparam1<SMMinfo.lb)
    k = find(xparam1<SMMinfo.lb);
    fval = objective_function_penalty_base+sum((SMMinfo.lb(k)-xparam1(k)).^2);
    exit_flag = 0;
    info = 41;
    return
end

% Return, with endogenous penalty, if some parameters are greater than the upper bound of the parameters.
if ~isequal(DynareOptions.mode_compute,1) && any(xparam1>SMMinfo.ub)
    k = find(xparam1>SMMinfo.ub);
    fval = objective_function_penalty_base+sum((xparam1(k)-SMMinfo.ub(k)).^2);
    exit_flag = 0;
    info = 42;
    return
end

% Set all parameters
Model = set_all_parameters(xparam1,EstimatedParameters,Model);


% Test if Q is positive definite.
Q = Model.Sigma_e;
if EstimatedParameters.ncx
    % Try to compute the cholesky decomposition of Q (possible iff Q is positive definite)
    [CholQ,testQ] = chol(Q);
    if testQ
        % The variance-covariance matrix of the structural innovations is not definite positive. We have to compute the eigenvalues of this matrix in order to build the endogenous penalty.
        a = diag(eig(Q));
        k = find(a < 0);
        if k > 0
            fval = objective_function_penalty_base+sum(-a(k));
            exit_flag = 0;
            info = 43;
            return
        end
    end
end

%------------------------------------------------------------------------------
% 2. call resol to compute steady state and model solution
%------------------------------------------------------------------------------

[dr_dynare_state_space,info,Model,DynareOptions,DynareResults] = resol(0,Model,DynareOptions,DynareResults);

if info(1) == 1 || info(1) == 2 || info(1) == 5 || info(1) == 7 || info(1) ...
            == 8 || info(1) == 22 || info(1) == 24 || info(1) == 19 || info(1) == 9
    fval = objective_function_penalty_base+1;
    info = info(1);
    exit_flag = 0;
    return
elseif info(1) == 3 || info(1) == 4 || info(1)==6 || info(1) == 20 || info(1) == 21  || info(1) == 23
    fval = objective_function_penalty_base+info(2);
    info = info(1);
    exit_flag = 0;
    return
end

if info(1)
    fval = NaN;
    return;
end

%------------------------------------------------------------------------------
% 3. Compute Moments of the model solution for normal innovations
%------------------------------------------------------------------------------
% create shock series with correct covariance matrix from iid standard
% normal shocks
i_exo_var = setdiff([1:Model.exo_nbr],find(diag(Model.Sigma_e) == 0 )); %find singular entries in covariance
chol_S = chol(Model.Sigma_e(i_exo_var,i_exo_var));
scaled_shock_series=zeros(size(DynareResults.smm.shock_series)); %initialize
scaled_shock_series(:,i_exo_var) = DynareResults.smm.shock_series(:,i_exo_var)*chol_S; %set non-zero entries


%% simulate series
y_sim = simult_(dr_dynare_state_space.ys,dr_dynare_state_space,scaled_shock_series,DynareOptions.order);

if any(any(isnan(y_sim))) || any(any(isinf(y_sim)))
    fval= objective_function_penalty_base;
    return
end
y_sim_after_burnin = y_sim(SMMinfo.varsindex,end-DynareOptions.smm.long:end)';
autolag=max(DynareOptions.smm.autolag);
if DynareOptions.smm.centeredmoments
   y_sim_after_burnin=bsxfun(@minus,y_sim_after_burnin,mean(y_sim_after_burnin,1)); 
end
[modelMoments, E_y, E_yy, autoE_yy] = moments_SMM_Data(y_sim_after_burnin,DynareOptions);
% write centered and uncentered simulated moments to results
DynareResults.smm.unconditionalmoments.E_y=E_y;
DynareResults.smm.unconditionalmoments.E_yy=E_yy;
DynareResults.smm.unconditionalmoments.autoE_yy=autoE_yy;
DynareResults.smm.unconditionalmoments.Var_y=E_yy-E_y*E_y';
DynareResults.smm.unconditionalmoments.Cov_y=autoE_yy-repmat(E_y*E_y',[1 1 autolag]);

%------------------------------------------------------------------------------
% 4. Compute quadratic target function using weighting matrix W
%------------------------------------------------------------------------------
moments_difference = DynareResults.smm.datamoments.momentstomatch-modelMoments;
fval = moments_difference'*DynareResults.smm.W*moments_difference;

if DynareOptions.smm.use_prior
    fval=fval+(xparam1-SMMinfo.p1)'/diag(SMMinfo.p2)*(xparam1-SMMinfo.p1);
end
end

