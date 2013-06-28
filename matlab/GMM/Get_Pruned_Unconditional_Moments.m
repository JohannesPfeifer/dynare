function [unconditionalmoments] = Get_Pruned_Unconditional_Moments(dr,DynareOptions,HigherMoments,lags)
% [unconditionalmoments] = Get_Pruned_Unconditional_Moments(dr,DynareOptions,HigherMoments)
% Gets the unconditional moments
% 
% INPUTS 
%   o dr :                Dynare decision rules structure
%   o DynareOptions   :   Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o HigherMoments:      Structure describing higher moments
%  
% OUTPUTS 
%   o unconditionalmoments      Structure storing only unconditionalmoments
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


% Compute first and second moments based on first or second order
% approximation
if DynareOptions.order == 1
    unconditionalmoments=Unconditional_Moments_1st_Lyap(dr,DynareOptions,lags);
else
    unconditionalmoments = Unconditional_Moments_2nd_Lyap(dr,DynareOptions,HigherMoments,lags);
end
% Compute first and second moments based on third order approximation
if DynareOptions.order == 3
    unconditionalmoments = Unconditional_Moments_3rd_Lyap(dr,DynareOptions,HigherMoments,unconditionalmoments,lags);
end
