function [gamma_y, var_mean, var_variance, autocorr]=disp_th_moments_order3(dr,M_,options_,i_var)
% [gamma_y, var_mean, var_variance, autocorr]=disp_th_moments_order3(dr,M_,options_,i_var)
% Display theoretical moments of variables based on (third order) pruned
% state-space
%
% INPUTS:
% dr :                      Dynare decision rules structure
% M_ :                      Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).          
% options_   :              Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
% i_var :                   Index of requested variables in policy rules, i.e. in dr.order_var
% 
%
% OUTPUTS: 
% gamma_y                   [cell] Matlab cell of nar+1 arrays, where nar is the order of the autocorrelation function.
%                                      gamma_y{1}       [double]  Covariance matrix.
%                                      gamma_y{i+1}     [double]  Autocorrelation function (for i=1,...,options_.ar).
% var_mean                  [vector] Unconditional mean
% var_variance              [vector] Unconditional covariance matrix
% autocorr                  [cell] Cell storing the theoretical autocorrelation
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


nvar = length(i_var);
[unconditionalmoments, gamma_y]=get_Moments_pruned_state_space(dr,M_,options_,dr.inv_order_var(i_var));

m = unconditionalmoments.E_y;

i1 = find(abs(diag(gamma_y{1})) > 1e-12);
s2 = diag(gamma_y{1});
sd = sqrt(s2);

z = [ m sd s2 ];
var_mean = m;
var_variance = gamma_y{1};

if ~options_.noprint %options_.nomoments == 0
    title='THEORETICAL MOMENTS BASED ON PRUNED STATE SPACE';
    if options_.hp_filter
        error('Theoretical HP-filtered  moments not available')
    end
    headers=char('VARIABLE','MEAN','STD. DEV.','VARIANCE');
    labels = deblank(M_.endo_names(i_var,:));
    lh = size(labels,2)+2;
    dyntable(title,headers,labels,z,lh,11,4);
end

if length(i1) == 0
    disp(' ')
    disp('All endogenous are constant or non stationary, not displaying correlations and auto-correlations')
    disp(' ')
    return;
end

if options_.nocorr == 0 % && size(stationary_vars, 1) > 0
    corr = gamma_y{1}(i1,i1)./(sd(i1)*sd(i1)');
    if ~options_.noprint,
        disp(' ')
        title='MATRIX OF CORRELATIONS BASED ON PRUNED STATE SPACE';            
        if options_.hp_filter
            error('Theoretical HP-filtered moments not available')
        end
        labels = deblank(M_.endo_names(i_var(i1),:));
        headers = char('Variables',labels);
        lh = size(labels,2)+2;
        dyntable(title,headers,labels,corr,lh,8,4);
    end
end
if options_.ar > 0 %&& size(stationary_vars, 1) > 0
    z=[];
    for i=1:options_.ar
        autocorr{i} = gamma_y{i+1};
        z(:,i) = diag(gamma_y{i+1}(i1,i1));
    end
    if ~options_.noprint,      
        disp(' ')    
        title='COEFFICIENTS OF AUTOCORRELATION BASED ON PRUNED STATE SPACE';            
        if options_.hp_filter        
            error('Theoretical HP-filtered  moments not available')
        end      
        labels = deblank(M_.endo_names(i_var(i1),:));      
        headers = char('Order ',int2str([1:options_.ar]'));
        lh = size(labels,2)+2;
        dyntable(title,headers,labels,z,lh,8,4);
    end  
end
