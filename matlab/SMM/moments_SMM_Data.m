function [dataMoments, E_y, E_yy, autoE_yy, m_data]= moments_SMM_Data(data,DynareOptions)
% [dataMoments E_y E_yy autoE_yy m_data]= moments_GMM_Data(data,DynareOptions)
% This function computes the following empirical moments from data
%  - E[y]
%  - E[y(i)*y(j)]       for i=1:ny and j=i,ny
%  - E[y(i)_t*y(i)_t-k] for i=1:ny and k=1,2,...autoLoag 
%
% INPUTS
%   data        [T x ny] matrix             data set
%   numMom      scalar                      number of selected moments
%   DynareOptions                           Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
% 

%   dataMoments [numMom x 1] vector         selected moments
%   E_y         [ny x 1] vector             (un)centered first moments
%   E_yy        [ny x ny] matrix            (un)centered contemporaneous second moments
%   autoE_yy    [ny x ny x nLags] matrix    (un)centered lagged second moments
%   m_data      [T x numMom] matrix         empirical moments at each point in time 
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


autoLagsIdx=DynareOptions.gmm.autolag;
centeredmoments=DynareOptions.gmm.centeredmoments;

% Number of observations (T) and number of observables (ny)
[T,ny] = size(data);

% The means
E_y       = sum(data,1)'/T;
E_yy      = data'*data/T;
maxLag     = max(autoLagsIdx);
autoE_yy  = zeros(ny,ny,maxLag);
for ii=1:maxLag
    autoE_yy(:,:,ii) = data(1+ii:T,:)'*data(1:T-ii,:)/T;
end

% We collect the moments we need for estimation
[dataMoments, numMom, numLags, firstmoment_selector]= collect_Moments(E_y,E_yy,autoE_yy,DynareOptions);

if nargout>4
    m_data = zeros(T,numMom);
    num_first_mom= ny-sum(~firstmoment_selector);
    % For Ey
    idxMom=num_first_mom;
    m_data(:,1:num_first_mom)= data(:,firstmoment_selector==1);
    % For Eyy
    for i=1:ny
        for j=i:ny
            idxMom = idxMom + 1;
            m_data(:,idxMom) = data(:,i).*data(:,j);
        end
    end

    % For autoEyy
    autoEyy  = zeros(ny,ny,maxLag);
    for i=1:maxLag
        for j=1:ny
            autoEyy(1+i:T,j,i) = data(1+i:T,j).*data(1:T-i,j);
        end
    end
    for i=1:numLags
        for j=1:ny
            idxMom = idxMom + 1;
            m_data(:,idxMom) = autoEyy(:,j,autoLagsIdx(1,i));
        end
    end   
end
end

