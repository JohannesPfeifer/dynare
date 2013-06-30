function [inv_chol_s]=get_inv_chol_covar_mat(Sigma_e,diag_dummy)
% function  [chol_s]=get_chol_covar_mat(Sigma_e,)
% compute inverse of lower triangular cholesky decomposition accounting for singularity
%
% INPUTS
%   Sigma_e:            Covariance matrix
%   diag_dummy:         dummy indicating if only diagonal should be taken
% OUTPUTS
%   chol_s:             Cholesky of covariance matrix
%
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
exo_nbr=length(Sigma_e);
if diag_dummy
    SS=diag(diag(Sigma_e))+1e-14*eye(exo_nbr);
else
    SS=Sigma_e+1e-14*eye(size(Sigma_e));
end
chol_diag_s = transpose(chol(SS));  
inv_chol_s=eye(exo_nbr)/chol_diag_s;
singular_entries=find(diag(Sigma_e)==0); %make sure IRFs are not printed if singular (1e-7 criterion does not bite if multiple>1)
if ~isempty(singular_entries)
    inv_chol_s(singular_entries,:)=0;
    inv_chol_s(:,singular_entries)=0;
end                
