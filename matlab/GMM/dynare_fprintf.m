function [] = dynare_fprintf(verbose,varargin)
% This function sets up the state space linear in innovations and calls the function computing 
% the closed-form expressions for the impulse response functions using the pruning method 
% when using the following definition of the impulse response function.
%
% INPUTS 
%   o verbose                   Dummy variable for verbosity
%   o nargin                    Input to fprintf
%
% OUTPUT: none
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

if verbose
   fprintf(varargin{1,1},varargin{2:end}); 
end