function [dr] = State_Space_LinearInov(DynareModel,dr,control_indices,state_indices)
% This function sets up an alternative state space representation for a model solved 
% up to third order where innovations only enter linearly.
% That is we transform the solution form
% z_t = f(x_t-1,u_t,sig)
% to 
% y_t   = g(x_t-1,u_t,sig)
% x_t   = h(x_t-1,u_t,sig)
% u_t   = eps_t
% where v_t = [x_t-1,u_t]
%
% INPUTS 
%   o DynareModel               Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).          
%   o dr :                      Dynare decision rules structure
%   o control_indices :         indices of control variables for which to build state space
%   o state_indices :           indices the state variables in the augmented state space
%  
% OUTPUTS 
%   o dr :                      Dynare decision rules structure
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
% Based on code by Martin M. Andreasen. 

nv = DynareModel.nspred + DynareModel.exo_nbr;
nu = DynareModel.exo_nbr;
nx = DynareModel.nspred;            % Number of predetermined variables in the state equation.

% We setup the function h. The dimensions are
dr.h_s0    = zeros(nv,1); %the steady state level
dr.h_v      = zeros(nv,nv);
dr.h_vv     = zeros(nv,nv,nv);
dr.h_ss     = zeros(nv,1);
dr.h_vvv    = zeros(nv,nv,nv,nv);
dr.h_ssv    = zeros(nv,nv);
dr.h_sss    = zeros(nv,1);

% Setting eta
dr.eta     = zeros(nv,nu);
dr.eta(nx+1:nv,1:nu) = dr.sigma;

% We construct the h function
% Order of variables DynareModel.endo_names(dr.order_var(state_indices),:)
dr.h_s0(1:nx,:)      = dr.f_s0(state_indices,1);
dr.h_v(1:nx,:)       = dr.f_v(state_indices,:);
dr.h_vv(1:nx,:,:)    = dr.f_vv(state_indices,:,:);            
dr.h_ss(1:nx,:)      = dr.f_ss(state_indices,:);
dr.h_vvv(1:nx,:,:,:) = dr.f_vvv(state_indices,:,:,:);            
dr.h_ssv(1:nx,:)     = dr.f_ssv(state_indices,:);    
dr.h_sss(1:nx,:)     = dr.f_sss(state_indices,:);    
% We construct the g function

dr.g_s0     = dr.f_s0(control_indices,:);
dr.g_v      = dr.f_v(control_indices,:);
dr.g_vv     = dr.f_vv(control_indices,:,:);
dr.g_ss     = dr.f_ss(control_indices,1);
dr.g_vvv    = dr.f_vvv(control_indices,:,:,:);
dr.g_ssv    = dr.f_ssv(control_indices,:);
dr.g_sss    = dr.f_sss(control_indices,1);
