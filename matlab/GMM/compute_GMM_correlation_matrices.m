function [unconditionalmoments]=compute_GMM_correlation_matrices(unconditionalmoments,autolags)
% [DynareResults]=compute_GMM_SMM_correlation_matrices(xparam1,DynareDataset,DynareOptions,Model,EstimatedParameters,GMMinfo,DynareResults);
% Computes the (auto-)correlation from the covariances 
% 
% INPUTS 
%   o unconditionalmoments      Structure storing only unconditionalmoments
%   o autolags                  lags for autocorrelation
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

ny=size(unconditionalmoments.Var_y,1);
nv=size(unconditionalmoments.Var_v,1);

autolag_max=max(autolags);

prod_var=repmat(sqrt(diag(unconditionalmoments.Var_y)),1,ny).*...
        repmat(sqrt(diag(unconditionalmoments.Var_y)'),ny,1);
unconditionalmoments.Corr_y = unconditionalmoments.autoCov_y./...
    repmat(prod_var,[1 1 autolag_max]); 

%set NaNs due to 0 variances to 0
unconditionalmoments.Corr_y(repmat(prod_var==0,[1 1 autolag_max]))=0; 

prod_var=repmat(sqrt(diag(unconditionalmoments.Var_v)),1,nv).*...
        repmat(sqrt(diag(unconditionalmoments.Var_v)'),nv,1);
unconditionalmoments.Corr_v = unconditionalmoments.autoCov_v./...
    repmat(prod_var,[1 1 autolag_max]); 

%set NaNs due to 0 variances to 0
unconditionalmoments.Corr_v(repmat(prod_var==0,[1 1 autolag_max]))=0; 
