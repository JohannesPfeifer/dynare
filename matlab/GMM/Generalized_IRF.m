function [GIRF] = Generalized_IRF(dr,Model,DynareOptions,shocksize,control_indices,vectorMom3,xf,xs)
% [GIRF] = Generalized_IRF(dr,Model,DynareOptions,shocksize,control_indices,vectorMom3,xf,xs)
% This function sets up the state space linear in innovations and calls the function computing 
% the closed-form expressions for the impulse response functions using the pruning method 
% when using the following definition of the impulse response function.
%
% INPUTS 
%   o dr_                       Matlab's decision rule structure
%   o Model                     Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).          
%   o DynareOptions             Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o shocksize     [double]    (p*1) vector of shocks (measured in standard deviations).
%   o control_indices [double]  (ny*1) vector of indices of requested
%                               endogenous variables in decision rules (i.e. M_.endo_names(dr.inv_order_var,:))
%   o vectorMom3    [double]    (p*1) vector of third moments
%   o xf            [double]    (p*1) vector of first order state at which
%                               to compute IRF
%   o xs            [double]    (p*1) vector of second order state at which
%                               to compute IRF
%
% OUTPUT:
%   o GIRF                  [structure] strucutre containing the GIRFs, including
%        GIRF.y             impulse response function for y = g(x,sig) (sum of all parts)
%
%        GIRF.parts.yf      impulse responses at first order
%        GIRF.parts.ys      impulse responses at second order
%        GIRF.parts.yrd     impulse responses at thrid order
%
%        GIRF.x             impulse response function for x = h(x,sig) (sum of all parts)
%
%        GIRF.parts.xf      impulse responses at first order
%        GIRF.parts.xs      impulse responses at second order
%        GIRF.parts.xrd     impulse responses at third order
%
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

if nargin<5
    control_indices=dr.inv_order_var(1:Model.orig_endo_nbr); % variables in matrices are in order_var ordering and need to be mapped to declaration order using inv_order_var    
end
if nargin<6
    vectorMom3=zeros(1,length(shocksize));
end
if nargin<7
    xf=[];
end
if nargin<8
    xs=[];
end

if DynareOptions.order>1 && ~DynareOptions.pruning
    error('GIRFs require the option pruning')
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
% 2. Compute Generalized IRFs for state-space at parameters
%------------------------------------------------------------------------------
[GIRF] = Generalized_IRF_core(dr,DynareOptions,shocksize,vectorMom3,xf,xs); 