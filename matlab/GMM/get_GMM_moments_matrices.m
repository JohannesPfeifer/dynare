function [DynareResults]=get_GMM_moments_matrices(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,DynareResults);
% [DynareResults]=get_GMM_moments_matrices(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,DynareResults);
% Computes the (auto-)correlation from the covariances and writes the
% moments to the results structure
% 
% INPUTS 
%   o xparam1:                  initial value of estimated parameters as returned by set_prior()
%   o DynareDataset:            data after required transformation
%   o DynareOptions             Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o Model                     Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).          
%   o EstimatedParameters:      Matlab's structure describing the estimated_parameters (initialized by dynare, see @ref{estim_params_}).
%   o GMMInfo                   Matlab's structure describing the GMM settings (initialized by dynare, see @ref{bayesopt_}).
%   o DynareResults             Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%  
% OUTPUTS 
%   o DynareResults             Matlab's structure gathering the results
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


%------------------------------------------------------------------------------
% 1. get moments at parameter values and pass decision rules at xparam1 to DynareResults
%------------------------------------------------------------------------------

[fval,moments_difference,modelMoments,exit_flag,ys,trend_coeff,info,Model,DynareOptions,GMMinfo,DynareResults]...
    = GMM_Objective_Function(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,DynareResults);

%------------------------------------------------------------------------------
% 2. compute correlations
%------------------------------------------------------------------------------

DynareResults.gmm.unconditionalmoments=compute_GMM_correlation_matrices(DynareResults.gmm.unconditionalmoments,DynareOptions.gmm.autolag);
