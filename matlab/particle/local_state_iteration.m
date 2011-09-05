function [y,y_] = local_state_iteration(yhat,epsilon,ghx,ghu,constant,ghxx,ghuu,ghxu,yhat_,ss)

%@info:
%! @deftypefn {Function File} {@var{y}, @var{y_} =} local_state_iteration (@var{yhat},@var{epsilon}, @var{ghx}, @var{ghu}, @var{constant}, @var{ghxx}, @var{ghuu}, @var{ghxu}, @var{yhat_}, @var{ss})
%! @anchor{particle/local_state_iteration}
%! @sp 1
%! Given an initial condition (y) and an innovation (epsilon), this routines computes the next value of the state variables if the
%! model is approximated by an order two taylor expansion around the deterministic steady state. Depending on the number of input/output
%! argument the pruning algorithm advocated by C. Sims is used or not.
%! @sp 2
%! @strong{Inputs}
%! @sp 1
%! @table @ @var
%! @item yhat
%! n*1 vector of doubles, initial condition, where n is the number of state variables.
%! @item epsilon
%! q*1 vector of doubles, structural innovations.
%! @item ghx
%! n*n matrix of doubles, is a subset of dr.ghx where we only consider the lines corresponding to the state variables.
%! @item ghu
%! n*q matrix of doubles, is a subset of dr.ghu where we only consider the lines corresponding to the state variables.
%! @item constant
%! n*1 vector of doubles, deterministic steady state plus second order correction for the state variables.
%! @item ghxx
%! n*n² matrix of doubles, subset of dr.ghxx where we only consider the lines corresponding to the state variables.
%! @item ghuu
%! n*q² matrix of doubles, subset of dr.ghuu where we only consider the lines corresponding to the state variables.
%! @item ghxu
%! n*(nq) matrix of doubles, subset of dr.ghxu where we only consider the lines corresponding to the state variables.
%! @item yhat_
%! n*1 vector of doubles, second initial condition for pruning version.
%! @item ss
%! n*1 vector of doubles, steady state for the union of the states and observed variables.
%! @end table
%! @sp 2
%! @strong{Outputs}
%! @sp 1
%! @table @ @var
%! @item y
%! n*1 vector of doubles, next values for the state variables.
%! @item y_
%! n*1 vector of doubles, update of the latent variables needed for the pruning version (first order update).
%! @end table
%! @sp 2
%! @strong{Remarks}
%! @sp 1
%! [1] If the function has 10 input arguments then it must have 2 output arguments (pruning version).
%! @sp 1
%! [2] If the function has 08 input arguments then it must have 1 output argument.
%! @sp 2
%! @strong{This function is called by:}
%! @sp 2
%! @strong{This function calls:}
%!
%!
%! @end deftypefn
%@eod:

% Copyright (C) 2011 Dynare Team
% stephane DOT adjemian AT univ DASH lemans DOT fr
% frederic DOT karame AT univ DASH evry DOT fr
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

number_of_threads = 1;

if nargin==8
    pruning = 0;
    if nargout>1
        error('local_state_iteration:: Numbers of input and output argument are inconsistent!')
    end
elseif nargin==10
    pruning = 1;
    if nargout~=2
        error('local_state_iteration:: Numbers of input and output argument are inconsistent!')
    end
else
    error('local_state_iteration:: Wrong number of input arguments!')
end

switch pruning
  case 0
    for i =1:size(yhat,2)
        y(:,i) = constant + ghx*yhat(:,i) + ghu*epsilon(:,i) ...
                 + A_times_B_kronecker_C(.5*ghxx,yhat(:,i),number_of_threads)  ...
                 + A_times_B_kronecker_C(.5*ghuu,epsilon(:,i),number_of_threads) ...
                 + A_times_B_kronecker_C(ghxu,yhat(:,i),epsilon(:,i),number_of_threads);
    end
  case 1
    for i =1:size(yhat,2)
        y(:,i) = constant + ghx*yhat(:,i) + ghu*epsilon(:,i) ...
                 + A_times_B_kronecker_C(.5*ghxx,yhat_(:,i),number_of_threads)  ...
                 + A_times_B_kronecker_C(.5*ghuu,epsilon(:,i),number_of_threads) ...
                 + A_times_B_kronecker_C(ghxu,yhat_(:,i),epsilon(:,i),number_of_threads);
    end
    y_ = ghx*yhat_+ghu*epsilon;
    y_ = bsxfun(@plus,y_,ss);
end

%@test:1
%$ addpath ../matlab
%$
%$ n = 2;
%$ q = 3;
%$
%$ yhat = zeros(n,1);
%$ epsilon = zeros(q,1);
%$ ghx = rand(n,n);
%$ ghu = rand(n,q);
%$ constant = ones(n,1);
%$ ghxx = rand(n,n*n);
%$ ghuu = rand(n,q*q);
%$ ghxu = rand(n,n*q);
%$ yhat_ = zeros(n,1);
%$ ss = ones(n,1);
%$
%$ % Call the tested routine.
%$ y1 = local_state_iteration(yhat,epsilon,ghx,ghu,constant,ghxx,ghuu,ghxu);
%$ [y2,y2_] = local_state_iteration(yhat,epsilon,ghx,ghu,constant,ghxx,ghuu,ghxu,yhat_,ss);
%$
%$ % Check the results.
%$ t(1) = dyn_assert(y1,ones(n,1));
%$ t(2) = dyn_assert(y2,ones(n,1));
%$ t(3) = dyn_assert(y2_,ones(n,1));
%$ T = all(t);
%@eof:1

%@test:2
%$ addpath ../matlab
%$ old_path = pwd;
%$ cd([fileparts(which('dynare')) '/../tests/']);
%$ global oo_
%$ dynare('dsge_base2');
%$ load dsge_base2
%$ cd(old_path);
%$ dr = oo_.dr;
%$ clear('oo_','options_','M_');
%$ delete([fileparts(which('dynare')) '/../tests/dsge_base2.mat']);
%$ istates = dr.nstatic+(1:dr.npred);
%$ n = dr.npred;
%$ q = size(dr.ghu,2);
%$ yhat = zeros(n,1);
%$ epsilon = zeros(q,1);
%$ ghx = dr.ghx(istates,:);ghx
%$ ghu = dr.ghu(istates,:);ghu
%$ constant = dr.ys(istates,:)+dr.ghs2(istates,:);
%$ ghxx = dr.ghxx(istates,:);ghxx
%$ ghuu = dr.ghuu(istates,:);ghuu
%$ ghxu = dr.ghxu(istates,:);ghxu
%$ yhat_ = zeros(n,1);
%$ ss = dr.ys(istates,:);
%$
%$ % Call the tested routine.
%$ y1 = local_state_iteration(yhat,epsilon,ghx,ghu,constant,ghxx,ghuu,ghxu);
%$ [y2,y2_] = local_state_iteration(yhat,epsilon,ghx,ghu,constant,ghxx,ghuu,ghxu,yhat_,ss);
%$
%$ % Check the results.
%$ t(1) = 1;%dyn_assert(y1,ones(n,1));
%$ t(2) = 1;%dyn_assert(y2,ones(n,1));
%$ t(3) = 1;%dyn_assert(y2_,ones(n,1));
%$ T = all(t);
%@eof:2