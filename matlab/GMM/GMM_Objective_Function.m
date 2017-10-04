function [fval,info,exit_flag,moments_difference,modelMoments,junk1,junk2,Model,DynareOptions,GMMinfo,DynareResults]...
= GMM_Objective_Function(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,BoundsInfo,DynareResults)
% [fval,info,exit_flag,moments_difference,modelMoments,junk1,junk2,Model,DynareOptions,GMMinfo,DynareResults]...
%    = GMM_Objective_Function(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,BoundsInfo,DynareResults)
% This function evaluates the objective function for GMM estimation
%
% INPUTS
%   o xparam1:                  initial value of estimated parameters as returned by set_prior()
%   o DynareDataset:            data after required transformation
%   o DynareOptions:            Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o Model                     Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).          
%   o EstimatedParameters:      Matlab's structure describing the estimated_parameters (initialized by dynare, see @ref{estim_params_}).
%   o GMMInfo                   Matlab's structure describing the GMM settings (initialized by dynare, see @ref{bayesopt_}).
%   o BoundsInfo                Matlab's structure containing prior bounds
%   o DynareResults             Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%
% OUTPUTS
%   o fval:                     value of the quadratic form of the moment difference
%   o info:                     vector storing error code and penalty 
%   o exit_flag:                0 if no error, 1 of error
%   o moments_difference:       [numMom x 1] vector with difference of empirical and model moments
%   o modelMoments:             [numMom x 1] vector with model moments
%   o Model:                    Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).
%   o DynareOptions:            Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o GMMinfo:                  Matlab's structure describing the GMM parameter options (initialized by dynare, see @ref{GMMinfo_}).
%   o DynareResults:            Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).

% SPECIAL REQUIREMENTS
%   none

% Copyright (C) 2013-17 Dynare Team
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
if ~isequal(DynareOptions.mode_compute,1) && any(xparam1<BoundsInfo.lb)
    k = find(xparam1<BoundsInfo.lb);
    fval = Inf;
    exit_flag = 0;
    info(1) = 41;
    info(4)= sum((BoundsInfo.lb(k)-xparam1(k)).^2);
    return
end

% Return, with endogenous penalty, if some parameters are greater than the upper bound of the parameters.
if ~isequal(DynareOptions.mode_compute,1) && any(xparam1>BoundsInfo.ub)
    k = find(xparam1>BoundsInfo.ub);
    fval = Inf;
    exit_flag = 0;
    info(1) = 42;
    info(4)= sum((xparam1(k)-BoundsInfo.ub(k)).^2);
    return
end

% Set all parameters
Model = set_all_parameters(xparam1,EstimatedParameters,Model);

% Test if Q is positive definite.
if ~issquare(Model.Sigma_e) || EstimatedParameters.ncx || isfield(EstimatedParameters,'calibrated_covariances')
    [Q_is_positive_definite, penalty] = ispd(Model.Sigma_e(EstimatedParameters.Sigma_e_entries_to_check_for_positive_definiteness,EstimatedParameters.Sigma_e_entries_to_check_for_positive_definiteness));
    if ~Q_is_positive_definite
        fval = Inf;
        exit_flag = 0;
        info(1) = 43;
        info(4) = penalty;
        return
    end
    if isfield(EstimatedParameters,'calibrated_covariances')
        correct_flag=check_consistency_covariances(Model.Sigma_e);
        if ~correct_flag
            penalty = sum(Model.Sigma_e(EstimatedParameters.calibrated_covariances.position).^2);
            fval = Inf;
            exit_flag = 0;
            info(1) = 71;
            info(4) = penalty;
            return
        end
    end
end

%------------------------------------------------------------------------------
% 2. call resol to compute steady state and model solution
%------------------------------------------------------------------------------

% Linearize the model around the deterministic steady state and extract the matrices of the state equation (T and R).
[dr_dynare_state_space,info,Model,DynareOptions,DynareResults] = resol(0,Model,DynareOptions,DynareResults);

% Return, with endogenous penalty when possible, if dynare_resolve issues an error code (defined in resol).
if info(1)
    if info(1) == 3 || info(1) == 4 || info(1) == 5 || info(1)==6 ||info(1) == 19 ||...
                info(1) == 20 || info(1) == 21 || info(1) == 23 || info(1) == 26 || ...
                info(1) == 81 || info(1) == 84 ||  info(1) == 85 ||  info(1) == 86
        %meaningful second entry of output that can be used
        fval = Inf;
        info(4) = info(2);
        exit_flag = 0;
        return
    else
        fval = Inf;
        info(4) = 0.1;
        exit_flag = 0;
        return
    end
end

% % check endogenous prior restrictions
% info=endogenous_prior_restrictions(T,R,Model,DynareOptions,DynareResults);
% if info(1)
%     fval = Inf;
%     info(4)=info(2);
%     exit_flag = 0;
%     return
% end

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

if DynareOptions.gmm.penalized_estimator
    fval=fval+(xparam1-GMMinfo.p1)'/diag(GMMinfo.p2)*(xparam1-GMMinfo.p1);
end
end

