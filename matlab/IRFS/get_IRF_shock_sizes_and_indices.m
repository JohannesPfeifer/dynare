function [cs, SS, irf_shocks_indx]=get_IRF_shock_sizes_and_indices(M_,options_)
% function  [cs, SS]=get_IRF_shock_sizes(M_,options_)
% gets shock sizes for IRFS 
%
% INPUTS
%   M_:                 Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).
%   options_:           Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   irf_shocks_indx:    indices of shock vectors in cs
% OUTPUTS
%   cs:                 matrix with IRF shocks in columns
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

if ~isempty(options_.irf_opt.irf_shocks) %if shock size specified
    if options_.irf_opt.stderr  % if specified in terms of standard deviations
        if options_.irf_opt.nonorthogonal %if no orthogonalization
            SS(M_.exo_names_orig_ord,M_.exo_names_orig_ord)=M_.Sigma_e+1e-14*eye(M_.exo_nbr);
            cs = transpose(chol(SS));
            cs = options_.irf_opt.irf_shocks'*cs; 
        else %orthogonalization
            cs=options_.irf_opt.irf_shocks'.*(sqrt(diag(M_.Sigma_e))*ones(1,M_.exo_nbr));
        end        
    else % if specified in absolute terms       
        if options_.irf_opt.nonorthogonal %if no orthogonalization
            cs=options_.irf_opt.irf_shocks; 
        else %orthogonalization
            %compute correlation matrix
            Correlation_matrix = diag(1./sqrt(diag(M_.Sigma_e)))*M_.Sigma_e*diag(1./sqrt(diag(M_.Sigma_e)));
            % Remove NaNs appearing because of variances calibrated to zero.
            if any(isnan(Correlation_matrix))
                zero_variance_idx = find(~diag(M_.Sigma_e));
                for i=1:length(zero_variance_idx)
                    Correlation_matrix(zero_variance_idx(i),:) = 0;
                    Correlation_matrix(:,zero_variance_idx(i)) = 0;
                    Correlation_matrix(zero_variance_idx(i),zero_variance_idx(i))=1;
                end        
            end
            cs=options_.irf_opt.irf_shocks'.*Correlation_matrix;
        end
    end
    irf_shocks_indx = (1:length(cs));
else  %if shock size not specified  
    SS(M_.exo_names_orig_ord,M_.exo_names_orig_ord)=M_.Sigma_e+1e-14*eye(M_.exo_nbr);
    cs = transpose(chol(SS));
    irf_shocks_indx = getIrfShocksIndx();
end
