function Correlation_matrix=compute_correlation_matrix(Sigma)
% function  Correlation_matrix=compute_correlation_matrix(Sigma)
% computes correlation matrix from covariance matrix
% INPUTS
%   Sigma:       Covariance matrix
%
% OUTPUTS
%   Correlation_matrix: Correlatio matrix
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

Correlation_matrix = diag(1./sqrt(diag(Sigma)))*Sigma*diag(1./sqrt(diag(Sigma)));
% Remove NaNs appearing because of variances calibrated to zero.
if any(isnan(Correlation_matrix))
    zero_variance_idx = find(~diag(Sigma));
    for i=1:length(zero_variance_idx)
        Correlation_matrix(zero_variance_idx(i),:) = 0;
        Correlation_matrix(:,zero_variance_idx(i)) = 0;
        Correlation_matrix(zero_variance_idx(i),zero_variance_idx(i))=1;
    end        
end
