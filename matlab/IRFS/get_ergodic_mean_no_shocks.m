function [ergodicmean_no_shocks, y1st_start, y2nd_start, y3rd_start]= get_ergodic_mean_no_shocks(M_,oo_,options_)
% [ergodicmean_no_shocks, y1st_start, y2nd_start, y3rd_start]= get_ergodic_mean_no_shocks(M_,oo_,options_)
% computes ergodic mean in absence of shocks  
%
% INPUTS
%   M_:                     Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).
%   oo_:                    Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%   options_:               Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%
% OUTPUTS
%   ergodicmean_no_shocks:  ergodic mean in absence of shocks
%   y1st_start:             starting value for linear term of pruned state space
%   y2nd_start:             starting value for quadratic term of pruned state space
%   y3rd_start:             starting value for cubic term of pruned state space

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

y1st_start=[];
y2nd_start=[];
y3rd_start=[];
ergodicmean_no_shocks=[];

if ~options_.irf_opt.ergodic_mean_irf % if not requested, return empty matrices
return
end

ex1 = zeros(options_.irf+options_.irf_opt.EM.drop,M_.exo_nbr);
dynare_fprintf(options_.verbosity,'\nComputing the ergodic mean in the absence of shocks. Using %d periods.\n',options_.irf_opt.EM.drop)
if options_.order==2 && options_.pruning % in case of pruning, get starting values for terms at each order
  [out_noshock, y1st]= simult_(oo_.dr.ys,oo_.dr,ex1,options_.order);
  y1st_start=y1st(:,end);
elseif options_.order==3 && options_.pruning
  [out_noshock, y1st, y2nd, y3rd]= simult_(oo_.dr.ys,oo_.dr,ex1,options_.order);
  y1st_start=y1st(:,end);
  y2nd_start=y2nd(:,end);
  y3rd_start=y3rd(:,end);
else % if no pruning, just simulate
  [out_noshock]= simult_(oo_.dr.ys,oo_.dr,ex1,options_.order);  
end
ergodicmean_no_shocks=out_noshock(:,end,1);
% check for convergence
abs_change_EM=max(abs(ergodicmean_no_shocks-out_noshock(:,end-500)));
% iterate if not convergence
iter=1;      
while abs_change_EM >options_.irf_opt.EM.tolf && iter<20
    dynare_fprintf(options_.verbosity,'\nConvergence not achieved. Maximum absolute change in function values over the last\n')
    dynare_fprintf(options_.verbosity,'500 draws: %2.1e. \n',abs_change_EM)
    dynare_fprintf(options_.verbosity,'Add another %d periods.\n',options_.irf_opt.EM.drop)
    if options_.order==2 && options_.pruning
      [out_noshock, y1st]= simult_(ergodicmean_no_shocks,oo_.dr,ex1,options_.order,y1st_start);
      y1st_start=y1st(:,end);
    elseif options_.order==3 && options_.pruning
      [out_noshock, y1st, y2nd, y3rd]= simult_(ergodicmean_no_shocks,oo_.dr,ex1,options_.order,y1st_start,y2nd_start,y3rd_start);
      y1st_start=y1st(:,end);
      y2nd_start=y2nd(:,end);
      y3rd_start=y3rd(:,end);
    else % if no pruning, just simulate
      [out_noshock]= simult_(oo_.dr.ys,oo_.dr,ex1,options_.order);  
    end
    ergodicmean_no_shocks=out_noshock(:,end); 
    abs_change_EM=max(abs(ergodicmean_no_shocks-out_noshock(:,end-500)));
    iter=iter+1;
end
