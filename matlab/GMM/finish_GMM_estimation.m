function [M_, options_, oo_] = finish_GMM_estimation(M_, options_, oo_)

% function [M_, options_, oo_] = finish_gmm_estimation(M_, options_, oo_)
% performs tidying up tasks after GMM estimation 
%
% INPUTS
%   M_:             Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).
%   options_:       Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   oo_:            Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%
% OUTPUTS
%   M_:             Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).
%   options_:       Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   oo_:            Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).

% SPECIAL REQUIREMENTS
%   none

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


warning('on','MATLAB:singularMatrix');
%restore old options
options_.irf=options_.gmm.old_irf;
options_.order=options_.gmm.old_order;
options_.gmm=rmfield(options_.gmm,'old_irf');
options_.gmm=rmfield(options_.gmm,'old_order');
