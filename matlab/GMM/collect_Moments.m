function [moments, numMom, numLags, firstmoment_selector]= collect_Moments(E_y,E_yy,autoE_yy,DynareOptions)
% function  [moments, numMom, numLags]= collect_Moments(E_y,E_yy,autoE_yy,DynareOptions)
% Selecting the moments for GMM estimation 
%  
% INPUTS
%   E_y         [ny x 1] vector             (un)centered first moments
%   E_yy        [ny x ny] matrix            (un)centered contemporaneous second moments
%   autoE_yy    [ny x ny x nLags] matrix    (un)centered lagged second moments
%   DynareOptions                           Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%  
% OUTPUTS
%   moments     [numMom x 1] vector         selected moments
%   numMom      scalar                      number of selected moments
%   numLags     scalar                      number of selected lags
%   firstmoment_selector [ny x 1] vector    selector vector for first
%                                           moments
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
uncenteredmoments=~DynareOptions.gmm.centeredmoments;
if uncenteredmoments
    if ~isempty(DynareOptions.gmm.firstmoment_selector)
        firstmoment_selector=DynareOptions.gmm.firstmoment_selector;
    else
        firstmoment_selector=ones(size(E_y));
    end
else
  firstmoment_selector=zeros(size(E_y));  
end
ny           = size(E_y,1);
numLags      = size(autoLagsIdx,2);
num_first_mom= ny-sum(~firstmoment_selector);
numMom       = num_first_mom+ ny*(ny+1)/2 + ny*numLags;
moments      = zeros(numMom,1);
moments(1:num_first_mom,1)= E_y(firstmoment_selector==1,1);
temp=ones(size(E_yy));
temp=vec(tril(temp));
Eyy_vec=vec(E_yy);
moments(num_first_mom+1:num_first_mom+ny*(ny+1)/2,1)=Eyy_vec(temp==1) ; %move along rows instead of columns

idxMom=num_first_mom+ny*(ny+1)/2;
for i=1:numLags
    moments(idxMom+1:idxMom+ny,1) = diag(autoE_yy(:,:,autoLagsIdx(1,i)));
    idxMom = idxMom + ny;
end

end

