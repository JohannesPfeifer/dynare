function [y, IRF_type, IRF_save_title]=get_IRFs(shock_vector,M_,oo_,options_,iter,n_irfs,ergodicmean_no_shocks,y1st_start,y2nd_start,y3rd_start)
% [y, IRF_type, IRF_save_title]=get_IRFs(M_,oo_,options_,i)
% computes ergodic mean in absence of shocks  
%
% INPUTS
%   shock_vector:           [exo_nbr by 1] shock vector for IRFs
%   M_:                     Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).
%   oo_:                    Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%   options_:               Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   iter:                   Number of current IRF
%   n_irfs:                 Number of total IRFs
%   ergodicmean_no_shocks:   ergodic mean in the absence of shocks

% OUTPUTS
%   y:                      matrix of IRFs
%   IRF_type:               string indicating the type of IRF for figure title
%   IRF_save_title:         string indicating the type of IRF for saving

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

persistent time

if iter==1
time=zeros(n_irfs,1);
end

if options_.irf_opt.generalized_irf
    dynare_fprintf(options_.verbosity,'Computing Generalized Impulse Responses. Progress: %d of %d\n',iter,n_irfs)
    tic_ID=tic;
    [GIRF] = Generalized_IRF(oo_.dr,M_,options_,shock_vector);
    time(iter)=toc(tic_ID);
    if options_.verbosity
        time_per_run=mean(time)/60; %compute average time for one run
        if iter~=n_irfs %if not last run
            dynare_fprintf(options_.verbosity,'Estimated time remaining: %4.1f minutes\n',time_per_run*(n_irfs-iter));
        end
    end
    y=GIRF.y;
    IRF_type='GIRF:';
    IRF_save_title='GIRF';
elseif options_.irf_opt.ergodic_mean_irf
    ex2=zeros(options_.irf,M_.exo_nbr);
    ex2(1,:)=shock_vector';
    if options_.order==2 && options_.pruning
      [out_withshock]= simult_(ergodicmean_no_shocks,oo_.dr,ex2,options_.order,y1st_start);
    elseif options_.order==3 && options_.pruning
      [out_withshock]= simult_(ergodicmean_no_shocks,oo_.dr,ex2,options_.order,y1st_start,y2nd_start,y3rd_start);
    else %no pruning
      [out_withshock]= simult_(ergodicmean_no_shocks,oo_.dr,ex2,options_.order);        
    end
    y = (out_withshock(:,2:end) - ergodicmean_no_shocks*ones(1,options_.irf));
    IRF_type='EM IRF:';
    IRF_save_title='EM_IRF';
else
    dynare_fprintf(options_.verbosity,'Computing Impulse Responses. Progress: %d of %d\n',iter,n_irfs)
    tic_ID=tic;
    y=irf(oo_.dr,shock_vector, options_.irf, options_.drop, ...
          options_.replic, options_.order);
    time(iter)=toc(tic_ID);
    if options_.verbosity
        time_per_run=mean(time)/60; %compute average time for one run
        if iter~=n_irfs %if not last run
            dynare_fprintf(options_.verbosity,'Estimated time remaining: %4.1f minutes\n',time_per_run*(n_irfs-iter));
        end
    end
    IRF_type='IRF:';
    IRF_save_title='IRF';
end

if options_.irf_opt.percent
   y=y*100;
end