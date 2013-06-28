function [moments_difference]=get_GMM_ObjectFun_moments(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,DynareResults)
% [moments_difference]=get_GMM_objectFun_moments(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,DynareResults)
% Returns the moments difference as the first output argument for computation of the Standard Errors via numerical differentiation
% INPUTS 
%   o xparam1:        initial value of estimated parameters as returned by set_prior()
%   o DynareDataset:            data after required transformation
%   o DynareOptions             Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o Model                     Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).          
%   o EstimatedParameters:      Matlab's structure describing the estimated_parameters (initialized by dynare, see @ref{estim_params_}).
%   o GMMInfo                   Matlab's structure describing the GMM settings (initialized by dynare, see @ref{bayesopt_}).
%   o DynareResults             Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%  
% OUTPUTS 
%   o moments_difference        [numMom x 1] vector with difference between data and model moments 
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

[fval,moments_difference,modelMoments,exit_flag]...
    = GMM_Objective_Function(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,DynareResults);