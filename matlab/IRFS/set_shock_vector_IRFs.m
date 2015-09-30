function shock_vector=set_shock_vector_IRFs(shocks,shock_size,M_)
% function  shock_vector=set_shock_vector_IRFs(shocks,shock_size,M_)
% given shock names and sizes, sets the shock vector for IRFS 
%
% INPUTS
%   shocks:             [n by 1] character array, where n is the number of shocks set
%   shock_size:         [n by k] matrix where n is the number of shocks set
%                       and k the number of different plots to be plotted
%   M_:                 Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).
% OUTPUTS
%   shock_vector:       [exo_nbr by k] matrix where k is the number of different plots to be plotted
%   SS:                 Covariance matrix

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

if size(shocks,1)~=size(shock_size,1)
    error('Shock names and shock sizes are not conformable')
end
n_total_irfs=size(shock_size,2);
shock_vector=zeros(M_.exo_nbr,n_total_irfs);
for n_irf=1:n_total_irfs
    for ii = 1:size(shocks,1)
      varname = deblank(shocks(ii,:));
      for jj=1:M_.exo_nbr
         if strcmp(varname,deblank(M_.exo_names(jj,:)))
            shock_vector(jj,n_irf)=shock_size(ii,n_irf);   
         end
      end
    end
end
end