function [fval,moments_difference,modelMoments,exit_flag,junk1,junk2,info,Model,DynareOptions,GMMinfo,DynareResults]...
    = GMM_Objective_Function(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,DynareResults)
% [fval,moments_difference,modelMoments,exit_flag,ys,trend_coeff,info,Model,DynareOptions,GMMinfo,DynareResults]...
%    = GMM_Objective_Function(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,DynareResults)
% This function evaluates the objective function for GMM estimation
%
% INPUTS
%   o xparam1:                  initial value of estimated parameters as returned by set_prior()
%   o DynareDataset:            data after required transformation
%   o DynareOptions:            Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o Model                     Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).          
%   o EstimatedParameters:      Matlab's structure describing the estimated_parameters (initialized by dynare, see @ref{estim_params_}).
%   o GMMInfo                   Matlab's structure describing the GMM settings (initialized by dynare, see @ref{bayesopt_}).
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
%   o GMMinfo:                  Matlab's structure describing the GMM parameter options (initialized by dynare, see @ref{GMMinfo_}).
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
moments_difference=NaN(GMMinfo.numMom,1);
modelMoments=NaN(GMMinfo.numMom,1);
%------------------------------------------------------------------------------
% 1. Get the structural parameters & define penalties
%------------------------------------------------------------------------------

% Return, with endogenous penalty, if some parameters are smaller than the lower bound of the parameters.
if ~isequal(DynareOptions.mode_compute,1) && any(xparam1<GMMinfo.lb)
    k = find(xparam1<GMMinfo.lb);
    fval = objective_function_penalty_base+sum((GMMinfo.lb(k)-xparam1(k)).^2);
    exit_flag = 0;
    info = 41;
    return
end

% Return, with endogenous penalty, if some parameters are greater than the upper bound of the parameters.
if ~isequal(DynareOptions.mode_compute,1) && any(xparam1>GMMinfo.ub)
    k = find(xparam1>GMMinfo.ub);
    fval = objective_function_penalty_base+sum((xparam1(k)-GMMinfo.ub(k)).^2);
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
% 3. Set up state-space with linear innovations
%------------------------------------------------------------------------------

% Transformation of the approximated solution
DynareResults.gmm.dr = Dynare_Unfold_Matrices(Model,DynareOptions,dr_dynare_state_space);

% We set up the alternative state space representation and use only selected endogenous variables 
DynareResults.gmm.dr = State_Space_LinearInov(Model,DynareResults.gmm.dr,GMMinfo.control_indices,GMMinfo.state_indices);

%------------------------------------------------------------------------------
% 4. Compute Moments of the model solution for normal innovations
%------------------------------------------------------------------------------

DynareResults.gmm.unconditionalmoments = Get_Pruned_Unconditional_Moments(DynareResults.gmm.dr,DynareOptions,GMMinfo,DynareOptions.gmm.autolag);

% Get the moments implied by the model solution that are matched
if DynareOptions.gmm.centeredmoments
    modelMoments = collect_Moments(DynareResults.gmm.unconditionalmoments.E_y,DynareResults.gmm.unconditionalmoments.Var_y,DynareResults.gmm.unconditionalmoments.autoCov_y,DynareOptions);
else
    modelMoments = collect_Moments(DynareResults.gmm.unconditionalmoments.E_y,DynareResults.gmm.unconditionalmoments.E_yy,DynareResults.gmm.unconditionalmoments.autoE_yy,DynareOptions);
end

%------------------------------------------------------------------------------
% 4. Compute quadratic target function using weighting matrix W
%------------------------------------------------------------------------------
moments_difference = DynareResults.gmm.datamoments.momentstomatch-modelMoments;
fval = moments_difference'*DynareResults.gmm.W*moments_difference;

if DynareOptions.gmm.use_prior
    fval=fval+(xparam1-GMMinfo.p1)'/diag(GMMinfo.p2)*(xparam1-GMMinfo.p1);
end
end

