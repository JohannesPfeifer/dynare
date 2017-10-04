function [SE] = get_Standard_Errors_GMM_SMM(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,DynareResults,BoundsInfo,GMM_SMM_indicator)
% [SE] = get_Standard_Errors_GMM_SMM(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,DynareResults,BoundsInfo,GMM_SMM_indicator)
% This function computes standard errors to the GMM estimates
%
% INPUTS 
%   o xparam1:                  initial value of estimated parameters as returned by set_prior()
%   o DynareDataset:            data after required transformation
%   o DynareOptions             Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o Model                     Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).          
%   o EstimatedParameters:      Matlab's structure describing the estimated_parameters (initialized by dynare, see @ref{estim_params_}).
%   o GMMInfo                   Matlab's structure describing the GMM settings (initialized by dynare, see @ref{bayesopt_}).
%   o DynareResults             Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%   o BoundsInfo                Matlab's structure containing prior bounds
%   o GMM_SMM_indicator         string indicating SMM or GMM
%  
% OUTPUTS 
%   o SE                       [nparam x 1] vector of standard errors
%
% SPECIAL REQUIREMENTS
%   None.

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


% Get the Jacobian of the moment difference function
D = fjaco('get_GMM_SMM_objectFun_moments',xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,DynareResults,BoundsInfo,GMM_SMM_indicator);
if strcmp(GMM_SMM_indicator,'SMM')
    Variance_correction_factor=DynareResults.smm.variance_correction_factor;
elseif strcmp(GMM_SMM_indicator,'GMM')
    Variance_correction_factor=1;
end


W=DynareResults.(lower(GMM_SMM_indicator)).W;
T    = DynareDataset.nobs; %Number of observations

if ~strcmp('optimal',DynareOptions.(lower(GMM_SMM_indicator)).weighting_matrix)
    Wopt=DynareResults.(lower(GMM_SMM_indicator)).Wopt;
    AVar = 1/T*Variance_correction_factor*eye(length(xparam1),length(xparam1))*((D'*W*D)\D'*W/Wopt*W*D/(D'*W*D));
else
    AVar = 1/T*Variance_correction_factor*eye(length(xparam1),length(xparam1))/(D'*W*D);
end

SE   = sqrt(diag(AVar));