function [unconditionalmoments Gamma_y] = get_Moments_pruned_state_space(dr,Model,DynareOptions,control_indices,HigherMoments)
% [unconditionalmoments Gamma_y] = get_Moments_pruned_state_space(dr,Model,DynareOptions,control_indices,HigherMoments)
% Returns the unconditional moments of the pruned state space solution
% INPUTS 
%   o dr :                      Dynare decision rules structure
%   o Model                     Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).          
%   o DynareOptions   :         Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o control_indices :         indices of control variables for which to build state space
%   o HigherMoments:            Structure describing higher moments
%  
% OUTPUTS 
%   o unconditionalmoments      Structure storing only unconditional moments
%   o Gamma_y                   [cell] Matlab cell of nar+1 arrays, where nar is the order of the autocorrelation function.
%                                      Gamma_y{1}       [double]  Covariance matrix.
%                                      Gamma_y{i+1}     [double]  Autocorrelation function (for i=1,...,options_.nar).
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

state_indices=[Model.nstatic+1:Model.nstatic+Model.nspred];

if nargin<4
    control_indices=dr.inv_order_var;
end
if nargin<5
    HigherMoments.vectorMom3 = zeros(1,Model.exo_nbr);
    HigherMoments.vectorMom4 = ones(1,Model.exo_nbr)*3;
    if DynareOptions.order==3
        HigherMoments.vectorMom5 = zeros(1,Model.exo_nbr); 
        HigherMoments.vectorMom6 = ones(1,Model.exo_nbr)*15;
    end
end
% check if memory is sufficient
if DynareOptions.order == 3
    nx= Model.nspred+Model.exo_nbr;
    size_kronecker_matrix=nx^6*8;
    [user,sys] = memory;
    if size_kronecker_matrix> DynareOptions.max_memory_share*sys.PhysicalMemory.Available
        fprintf('\nSTOCH_SIMUL: The matrix required to compute analytical moments at\n') 
        fprintf('STOCH_SIMUL: order 3 is too big. It requires at least %3.1f gigabytes\n',size_kronecker_matrix/1024^3)
        fprintf('STOCH_SIMUL: of free memory. You can try to increase options_.max_memory_share\n')
        fprintf('STOCH_SIMUL: or use simulated moments by using the periods option.\n')
        error('Not enough memory to perform computation of analytical moments at order 3.')
    end
end


%------------------------------------------------------------------------------
% 1. Make sure state-space in linear innovations is set up, including
%    updating when control variables have changed
%------------------------------------------------------------------------------

% Transformation of the approximated solution
dr = Dynare_Unfold_Matrices(Model,DynareOptions,dr);

% We set up the alternative state space representation and use only selected endogenous variables 
dr = State_Space_LinearInov(Model,dr,control_indices,state_indices);

%------------------------------------------------------------------------------
% 2. Compute the moments at the current parameters
%------------------------------------------------------------------------------
[unconditionalmoments] = Get_Pruned_Unconditional_Moments(dr,DynareOptions,HigherMoments,DynareOptions.ar);
[unconditionalmoments] = compute_GMM_correlation_matrices(unconditionalmoments,DynareOptions.ar);

%------------------------------------------------------------------------------
% 3. Write the moments into the Gamma_y-structure documented in the manual
%------------------------------------------------------------------------------

if nargout ==2
  Gamma_y=cell(DynareOptions.ar+1,1);
  Gamma_y{1}=unconditionalmoments.Var_y;
  for ii=1:DynareOptions.ar
    Gamma_y{ii+1}=unconditionalmoments.Corr_y(:,:,ii);
  end
end