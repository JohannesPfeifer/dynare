function [M_, options_, oo_] = finish_GMM_SMM_estimation(M_, options_, oo_,case_string)
% function [M_, options_, oo_] = finish_GMM_SMM_estimation(M_, options_, oo_)
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


warning('on','MATLAB:singularMatrix');
%restore old options
options_.order=options_.(lower(case_string)).old_order;
options_.(lower(case_string))=rmfield(options_.(lower(case_string)),'old_order');