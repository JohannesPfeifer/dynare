function [Wopt] = get_Optimal_Weighting_Matrix_GMM_SMM(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMM_SMM_info,DynareResults,BoundsInfo,GMM_SMM_indicator)
% [Wopt] = get_Optimal_Weighting_Matrix_GMM_SMM(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMM_SMM_info,DynareResults,BoundsInfo,GMM_SMM_indicator)
% This function computes the optimal weigthing matrix by a Bartlett kernel with maximum lag qlag

% INPUTS 
%   o xparam1:                  initial value of estimated parameters as returned by set_prior()
%   o DynareDataset:            data after required transformation
%   o DynareOptions             Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o Model                     Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).          
%   o EstimatedParameters:      Matlab's structure describing the estimated_parameters (initialized by dynare, see @ref{estim_params_}).
%   o GMM_SMM_Info              Matlab's structure describing the GMM settings (initialized by dynare, see @ref{bayesopt_}).
%   o DynareResults             Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%   o BoundsInfo                Matlab's structure containing prior bounds
%   o GMM_SMM_indicator         string indicating SMM or GMM
%  
% OUTPUTS 
%   o Wopt                      [numMom x numMom] optimal weighting matrix
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

qLag=DynareOptions.(lower(GMM_SMM_indicator)).qLag;
% We compute the h-function for all observations
T = DynareDataset.nobs;

% Evaluating the objective function to get modelMoments
if strcmp('GMM',GMM_SMM_indicator)
    [fval,info,exit_flag,moments_difference,modelMoments] = GMM_Objective_Function(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMM_SMM_info,BoundsInfo,DynareResults);
    % centered around theoretical moments   
    hFunc = DynareResults.(lower(GMM_SMM_indicator)).datamoments.m_data - repmat(modelMoments',T,1);
elseif strcmp('SMM',GMM_SMM_indicator)
    [fval,info,exit_flag,moments_difference,modelMoments] = SMM_Objective_Function(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMM_SMM_info,BoundsInfo,DynareResults);
    % centered around data moments   
    hFunc = DynareResults.(lower(GMM_SMM_indicator)).datamoments.m_data - repmat(mean(DynareResults.(lower(GMM_SMM_indicator)).datamoments.m_data),T,1);
end

% The required correlation matrices
numMom=GMM_SMM_info.numMom;
GAMA_array = zeros(numMom,numMom,qLag);
GAMA0 = CorrMatrix(hFunc,T,numMom,0);
if qLag > 0
    for ii=1:qLag
        GAMA_array(:,:,ii) = CorrMatrix(hFunc,T,numMom,ii);
    end
end

% The estimate of S
S = GAMA0;
if qLag > 0
    for ii=1:qLag
        S = S + (1-ii/(qLag+1))*(GAMA_array(:,:,ii) + GAMA_array(:,:,ii)');
    end
end

Wopt = S\eye(size(S,1));

try 
    chol(Wopt);
catch err
    if DynareOptions.(lower(GMM_SMM_indicator)).recursive_estimation
        fprintf(2,'\n%s Error: The optimal weighting matrix is not positive definite.\n',GMM_SMM_indicator)
        fprintf(2,'Check whether your model implies stochastic singularity.\n')    
        fprintf(2,'I continue the recursive %s estimation with an identity weighting matrix.\n',GMM_SMM_indicator)
        Wopt=eye(size(Wopt));
    else
        error('%s Error: The optimal weighting matrix is not positive definite. Check whether your model implies stochastic singularity\n',GMM_SMM_indicator)
    end
end
end

% The correlation matrix
function GAMAcorr = CorrMatrix(hFunc,T,numMom,v)
GAMAcorr = zeros(numMom,numMom);
for t=1+v:T
    GAMAcorr = GAMAcorr + hFunc(t-v,:)'*hFunc(t,:);
end
GAMAcorr = GAMAcorr/T;    
end