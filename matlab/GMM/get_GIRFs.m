function [GIRF]=get_GIRFs(DynareResults,Model,DynareOptions,var_list)
% [GIRF] = get_GIRFs(DynareResults,Model,DynareOptions,var_list)
% This function sets up the state space linear in innovations and calls the function computing 
% the closed-form expressions for the impulse response functions using the pruning method 
% when using the following definition of the impulse response function.
%
% INPUTS 
%   o DynareResults             Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%   o Model                     Matlab's structure describing the Model (initialized by dynare, see @ref{Model}).          
%   o DynareOptions             Matlab's structure describing the options (initialized by dynare, see @ref{DynareOptions}).
%   o var_list                  List of variables for which to plot IRFs.
%
% OUTPUT:
%   o GIRF                  [structure] strucutre containing the GIRFs with
%                           fields of the form endogenousvariable_shock
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

[i_var,nvar] = varlist_indices(var_list,Model.endo_names);
irf_shocks_indx = getIrfShocksIndx();
tit(Model.exo_names_orig_ord,:) = Model.exo_names;
for i=irf_shocks_indx
    shockvector=zeros(Model.exo_nbr,1);
    shockvector(i,1)=1; %due to different state space, 1 corresponds to standard deviation shock
    [GIRF_temp] = Generalized_IRF(DynareResults.dr,Model,DynareOptions,shockvector);
    y=GIRF_temp.y;
    irfs   = [];
    mylist = [];
    for j = 1:nvar
        eval(['GIRF.' deblank(Model.endo_names(i_var(j),:)) '_' ...
              deblank(Model.exo_names(i,:)) ' = y(i_var(j),:);']);
        if max(y(i_var(j),:)) - min(y(i_var(j),:)) > 1e-10
            irfs  = cat(1,irfs,y(i_var(j),:));
            if isempty(mylist)
                mylist = deblank(var_list(j,:));
            else
                mylist = char(mylist,deblank(var_list(j,:)));
            end
        end
    end
    if DynareOptions.nograph == 0
        CheckPath('graphs',Model.dname);
        number_of_plots_to_draw = size(irfs,1);
        [nbplt,nr,nc,lr,lc,nstar] = pltorg(number_of_plots_to_draw);
        if nbplt == 0
        elseif nbplt == 1
                hh = dyn_figure(DynareOptions,'Name',['GIRF: Orthogonalized shock to' ...
                                    ' ' tit(i,:)]);
            for j = 1:number_of_plots_to_draw
                subplot(nr,nc,j);
                plot(1:DynareOptions.irf,transpose(irfs(j,:)),'-k','linewidth',1);
                hold on
                plot([1 DynareOptions.irf],[0 0],'-r','linewidth',0.5);
                hold off
                xlim([1 DynareOptions.irf]);
                title(deblank(mylist(j,:)),'Interpreter','none');
            end
            dyn_saveas(hh,[Model.fname ,filesep,'graphs', filesep, 'GIRF_' deblank(tit(i,:))],DynareOptions);
        else
            for fig = 1:nbplt-1
                    hh = dyn_figure(DynareOptions,'Name',['GIRF: Orthogonalized shock to ' tit(i,:) ...
                                        ' figure ' int2str(fig)]);
                for plt = 1:nstar
                    subplot(nr,nc,plt);
                    plot(1:DynareOptions.irf,transpose(irfs((fig-1)*nstar+plt,:)),'-k','linewidth',1);
                    hold on
                    plot([1 DynareOptions.irf],[0 0],'-r','linewidth',0.5);
                    hold off
                    xlim([1 DynareOptions.irf]);
                    title(deblank(mylist((fig-1)*nstar+plt,:)),'Interpreter','none');
                end
                dyn_saveas(hh,[Model.fname ,filesep,'graphs', filesep, 'GIRF_' deblank(tit(i,:)) int2str(fig)],DynareOptions);
            end
            hh = dyn_figure(DynareOptions,'Name',['GIRF: Orthogonalized shock to ' tit(i,:) ' figure ' int2str(nbplt) '.']);
            m = 0;
            for plt = 1:number_of_plots_to_draw-(nbplt-1)*nstar;
                m = m+1;
                subplot(lr,lc,m);
                plot(1:DynareOptions.irf,transpose(irfs((nbplt-1)*nstar+plt,:)),'-k','linewidth',1);
                hold on
                plot([1 DynareOptions.irf],[0 0],'-r','linewidth',0.5);
                hold off
                xlim([1 DynareOptions.irf]);
                title(deblank(mylist((nbplt-1)*nstar+plt,:)),'Interpreter','none');
            end
            dyn_saveas(hh,[Model.fname ,filesep,'graphs', filesep, 'GIRF_' deblank(tit(i,:)) int2str(nbplt) ],DynareOptions);
        end
    end
end

