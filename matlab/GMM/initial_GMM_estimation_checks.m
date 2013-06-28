function DynareResults = initial_GMM_estimation_checks(objective_function,xparam1,DynareDataset,Model,EstimatedParameters,DynareOptions,GMMinfo,DynareResults)
% function initial_estimation_checks(xparam1,gend,data,data_index,number_of_observations,no_more_missing_observations)
% Checks data (complex values, initial values, BK conditions,..)
%
% INPUTS
%    xparam1:                   vector of parameters to be estimated
%    DynareDataset:             data after required transformation
%    Model:                     Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).
%    EstimatedParameters:       Matlab's structure describing the estimated_parameters (initialized by dynare, see @ref{estim_params_}).
%    DynareOptions              Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%    GMMInfo                    Matlab's structure describing the GMM settings (initialized by dynare, see @ref{bayesopt_}).
%    DynareResults              Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
% OUTPUTS
%    DynareResults     structure of temporary results
%
% SPECIAL REQUIREMENTS
%    none

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

if DynareDataset.info.nvobs>Model.exo_nbr+EstimatedParameters.nvn
    error(['initial_estimation_checks:: Estimation can''t take place because there are less declared shocks than observed variables!'])
end

if EstimatedParameters.nvn || EstimatedParameters.ncn
    error('GMM does not support measurment error(s). Please specifiy them as a structural shock')
end

% Evaluate the moment-function.
tic_id=tic;
[fval,moments_difference,modelMoments,exit_flag,junk2,junk3,info] = GMM_Objective_Function(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,DynareResults);
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
if DynareOptions.gmm.centeredmoments
    if sum(data_mean)/size(DynareDataset.data,1) >1e-9
        fprintf('The mean of the data is:\n')
        disp(data_mean);
        error('You are trying to perform GMM estimation with centered moments using uncentered data.')
    end
elseif ~isempty(data_mean(DynareOptions.gmm.firstmoment_selector==1)) %if first moments are used
    if sum(data_mean(DynareOptions.gmm.firstmoment_selector==1))/sum(DynareOptions.gmm.firstmoment_selector==1) <1e-2
        warning('You are trying to perform GMM estimation with uncentered moments, but the data are (almost) mean 0. Check if this is desired.')
    end    
end

% if any(abs(DynareResults.steady_state(GMMinfo.mfys))>1e-9) && (DynareOptions.prefilter==1)
%     disp(['You are trying to estimate a model with a non zero steady state for the observed endogenous'])
%     disp(['variables using demeaned data!'])
%     error('You should change something in your mod file...')
% end

fprintf('Initial value of the objective function with identity weighting matrix: %6.4f \n\n', fval);
