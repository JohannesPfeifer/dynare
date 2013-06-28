function [W, Wopt] = get_Optimal_Weighting_Matrix_SMM(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,SMMinfo,DynareResults)
% [Wopt] = get_Optimal_Weighting_Matrix_SMM(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,SMMinfo,DynareResults)
% This function computes the optimal weigthing matrix by a Bartlett kernel with maximum lag qlag
% By Martin Andreasen

% INPUTS 
%   o xparam1:                  initial value of estimated parameters as returned by set_prior()
%   o DynareDataset:            data after required transformation
%   o DynareOptions             Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o Model                     Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).          
%   o EstimatedParameters:      Matlab's structure describing the estimated_parameters (initialized by dynare, see @ref{estim_params_}).
%   o SMMInfo                   Matlab's structure describing the SMM settings (initialized by dynare, see @ref{bayesopt_}).
%   o DynareResults             Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%  
% OUTPUTS 
%   o Wopt                      [numMom x numMom] optimal weighting matrix
%   o W                         [numMom x numMom] matrix actually used for
%                               weighting (depends on DynareOptions.smm.optimal_weighting)
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

global objective_function_penalty_base

qLag=DynareOptions.smm.qLag;

% Evaluating the objective function to get modelMoments
[fval,moments_difference,modelMoments] = SMM_Objective_Function(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,SMMinfo,DynareResults);

% We compute the h-function for all observations
T = DynareDataset.info.ntobs;
hFunc = DynareResults.smm.datamoments.m_data - repmat(modelMoments',T,1);

% The required correlation matrices
numMom=SMMinfo.numMom;
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
if DynareOptions.smm.optimal_weighting==1
    W = Wopt;
else
    W = DynareResults.smm.W;
end

try 
    chol(Wopt);
catch err
    if DynareOptions.smm.recursive_estimation
        fprintf(2,'\nSMM Error: The optimal weighting matrix is not positive definite.\n')    
        fprintf(2,'Check whether your model implies stochastic singularity.\n')    
        fprintf(2,'I continue the recursive SMM estimation with an identity weighting matrix.\n')
        Wopt=eye(size(Wopt));
        if DynareOptions.smm.optimal_weighting==1
            W = Wopt;
        else
            W = DynareResults.smm.W;
        end
        new_fval=moments_difference'*Wopt*moments_difference;
        if new_fval>objective_function_penalty_base;
           objective_function_penalty_base=1.1*new_fval;
        end
    else
        error('SMM Error: The optimal weighting matrix is not positive definite. Check whether your model implies stochastic singularity\n')    
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
