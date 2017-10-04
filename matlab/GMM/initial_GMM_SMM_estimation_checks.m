function DynareResults = initial_GMM_SMM_estimation_checks(objective_function,xparam1,DynareDataset,Model,EstimatedParameters,DynareOptions,SMM_GMMinfo,BoundsInfo,DynareResults,GMM_SMM_indicator)
% DynareResults = initial_GMM_SMM_estimation_checks(objective_function,xparam1,DynareDataset,Model,EstimatedParameters,DynareOptions,SMM_GMMinfo,BoundsInfo,DynareResults,GMM_SMM_indicator)
% Checks data (complex values, initial values, BK conditions,..)
%
% INPUTS
%    xparam1:                   vector of parameters to be estimated
%    DynareDataset:             data after required transformation
%    Model:                     Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).
%    EstimatedParameters:       Matlab's structure describing the estimated_parameters (initialized by dynare, see @ref{estim_params_}).
%    DynareOptions              Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%    SMM_GMMinfo                Matlab's structure describing the GMM settings (initialized by dynare, see @ref{bayesopt_}).
%    BoundsInfo                 Matlab's structure containing prior bounds
%    DynareResults              Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%    GMM_SMM_indicator          string indicating SMM or GMM
% OUTPUTS
%    DynareResults     structure of temporary results
%
% SPECIAL REQUIREMENTS
%    none

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

if DynareOptions.order>1 && any(any(isnan(DynareDataset.data)))
    error('initial_GMM_SMM_estimation_checks:: GMM/SMM does not support missing observations')
end

if EstimatedParameters.nvn || EstimatedParameters.ncn
    error('initial_GMM_SMM_estimation_checks:: GMM/SMM does not support measurement error(s). Please specifiy them as a structural shock')
end

if ~isempty(DynareOptions.endogenous_prior_restrictions.irf) && ~isempty(DynareOptions.endogenous_prior_restrictions.moment)
    error(['initial_GMM_SMM_estimation_checks:: Endogenous prior restrictions are not supported.'])
end

% Evaluate the moment-function.
tic_id=tic;

[fval,info,exit_flag,moments_difference,modelMoments] = feval(objective_function,xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,SMM_GMMinfo,BoundsInfo,DynareResults);
elapsed_time=toc(tic_id);
if isnan(fval)
    error('The initial value of the target function is NaN')
elseif imag(fval)
    error('The initial value of the target function is complex')
end

if info(1) > 0
    error('Error in computing moments for initial parameter values')
end

fprintf('Time required to compute moments once: %5.4f seconds \n', elapsed_time);

data_mean=abs(mean(DynareDataset.data'));
if DynareOptions.(lower(GMM_SMM_indicator)).centeredmoments
    if sum(data_mean)/size(DynareDataset.data,1) >1e-9
        fprintf('The mean of the data is:\n')
        disp(data_mean);
        error('You are trying to perform GMM/SMM estimation with centered moments using uncentered data.')
    end
elseif ~isempty(data_mean(DynareOptions.(lower(GMM_SMM_indicator)).firstmoment_selector==1)) %if first moments are used
    if sum(data_mean(DynareOptions.(lower(GMM_SMM_indicator)).firstmoment_selector==1))/sum(DynareOptions.(lower(GMM_SMM_indicator)).firstmoment_selector==1) <1e-2
        warning('You are trying to perform GMM estimation with uncentered moments, but the data are (almost) mean 0. Check if this is desired.')
    end    
end

fprintf('Initial value of the objective function with identity weighting matrix: %6.4f \n\n', fval);