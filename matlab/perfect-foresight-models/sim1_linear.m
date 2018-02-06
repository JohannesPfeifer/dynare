function [endogenousvariables, info] = sim1_linear(endogenousvariables, exogenousvariables, steadystate_y, steadystate_x, M, options)

% Solves a linear approximation of a perfect foresight model using sparse matrix.
%
% INPUTS
% - endogenousvariables [double] N*T array, paths for the endogenous variables (initial guess).
% - exogenousvariables  [double] T*M array, paths for the exogenous variables.
% - steadystate_y       [double] N*1 array, steady state for the endogenous variables.
% - steadystate_x       [double] M*1 array, steady state for the exogenous variables.
% - M                   [struct] contains a description of the model.
% - options             [struct] contains various options.
%
% OUTPUTS
% - endogenousvariables [double] N*T array, paths for the endogenous variables (solution of the perfect foresight model).
% - info                [struct] contains informations about the results.
%
% NOTATIONS
% - N is the number of endogenous variables.
% - M is the number of innovations.
% - T is the number of periods (including initial and/or terminal conditions).
%
% REMARKS
% - The structure `M` describing the structure of the model, must contain the
% following informations:
%  + lead_lag_incidence, incidence matrix (given by the preprocessor).
%  + endo_nbr, number of endogenous variables (including aux. variables).
%  + exo_nbr, number of innovations.
%  + maximum_lag,
%  + maximum_endo_lag,
%  + params, values of model's parameters.
%  + fname, name of the model.
%  + NNZDerivatives, number of non zero elements in the jacobian of the dynamic model.
% - The structure `options`, must contain the following options:
%  + verbosity, controls the quantity of information displayed.
%  + periods, the number of periods in the perfect foresight model.
%  + debug.
% - The steady state of the exogenous variables is required because we need
% to center the variables around the deterministic steady state to solve the
% perfect foresight model.

% Copyright (C) 2015-2017 Dynare Team
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

verbose = options.verbosity;

lead_lag_incidence = M.lead_lag_incidence;

ny = M.endo_nbr;
nx = M.exo_nbr;

maximum_lag = M.maximum_lag;
max_lag = M.maximum_endo_lag;

nyp = nnz(lead_lag_incidence(1,:));
ny0 = nnz(lead_lag_incidence(2,:));
nyf = nnz(lead_lag_incidence(3,:));

nd = nyp+ny0+nyf; % size of y (first argument passed to the dynamic file).

periods = options.periods;

params = M.params;

% Indices in A.
ip   = find(lead_lag_incidence(1,:)');
ic   = find(lead_lag_incidence(2,:)');
in   = find(lead_lag_incidence(3,:)');
icn  = find(lead_lag_incidence(2:3,:)');
ipcn = find(lead_lag_incidence');

% Indices in y.
jp  = nonzeros(lead_lag_incidence(1,:)');
jc  = nonzeros(lead_lag_incidence(2,:)');
jn  = nonzeros(lead_lag_incidence(3,:)');
jpc = [jp; jc];
jcn = [jc; jn];

jexog = transpose(nd+(1:nx));
jendo = transpose(1:nd);

i_upd = maximum_lag*ny+(1:periods*ny);

% Center the endogenous and exogenous variables around the deterministic steady state.
endogenousvariables = bsxfun(@minus, endogenousvariables, steadystate_y);
exogenousvariables = bsxfun(@minus, exogenousvariables, transpose(steadystate_x));

Y = endogenousvariables(:);

if verbose
    skipline()
    printline(80)
    disp('MODEL SIMULATION:')
    skipline()
end

dynamicmodel = str2func([M.fname,'_dynamic']);

z = steadystate_y([ip; ic; in]);
x = repmat(transpose(steadystate_x), 1+M.maximum_exo_lag+M.maximum_exo_lead, 1);

% Evaluate the Jacobian of the dynamic model at the deterministic steady state.
[d1, jacobian] = dynamicmodel(z, x, params, steadystate_y, M.maximum_exo_lag+1);

% Check that the dynamic model was evaluated at the steady state.
if max(abs(d1))>options.solve_tolf
    error('Jacobian is not evaluated at the steady state!')
end

[r0,c0,v0] = find(jacobian(:,jc));
[rT,cT,vT] = find(jacobian(:,jpc));
[r1,c1,v1] = find(jacobian(:,jcn));
[rr,cc,vv] = find(jacobian(:,jendo));

iv0 = 1:length(v0);
ivT = 1:length(vT);
iv1 = 1:length(v1);
iv  = 1:length(vv);

% Initialize the vector of residuals.
res = zeros(periods*ny, 1);

% Initialize the sparse Jacobian.
iA = zeros(periods*M.NNZDerivatives(1), 3);

h2 = clock;
i_rows = (1:ny)';
i_cols_A = ipcn;
i_cols = ipcn+(maximum_lag-1)*ny;
m = 0;
for it = (maximum_lag+1):(maximum_lag+periods)
    if isequal(it, maximum_lag+periods) && isequal(it, maximum_lag+1)
        nv = length(v0);
        iA(iv0+m,:) = [i_rows(r0),ic(c0),v0];
    elseif isequal(it, maximum_lag+periods)
        nv = length(vT);
        iA(ivT+m,:) = [i_rows(rT), i_cols_A(jpc(cT)), vT];
    elseif isequal(it, maximum_lag+1)
        nv = length(v1);
        iA(iv1+m,:) = [i_rows(r1), icn(c1), v1];
    else
        nv = length(vv);
        iA(iv+m,:) = [i_rows(rr),i_cols_A(cc),vv];
    end
    z(jendo) = Y(i_cols);
    z(jexog) = transpose(exogenousvariables(it,:));
    res(i_rows) = jacobian*z;
    m = m + nv;
    i_rows = i_rows + ny;
    i_cols = i_cols + ny;
    if it > maximum_lag+1
        i_cols_A = i_cols_A + ny;
    end
end

% Evaluation of the maximum residual at the initial guess (steady state for the endogenous variables).
err = max(abs(res));

if options.debug
    fprintf('\nLargest absolute residual at iteration %d: %10.3f\n', 1, err);
    if any(isnan(res)) || any(isinf(res)) || any(isnan(Y)) || any(isinf(Y))
        fprintf('\nWARNING: NaN or Inf detected in the residuals or endogenous variables.\n');
    end
    if ~isreal(res) || ~isreal(Y)
        fprintf('\nWARNING: Imaginary parts detected in the residuals or endogenous variables.\n');
    end
    skipline()
end

iA = iA(1:m,:);
A = sparse(iA(:,1), iA(:,2), iA(:,3), periods*ny, periods*ny);

% Try to update the vector of endogenous variables.
try
    Y(i_upd) =  Y(i_upd) - A\res;
catch
    % Normally, because the model is linear, the solution of the perfect foresight model should
    % be obtained in one Newton step. This is not the case if the model is singular.
    info.status = false;
    info.error = NaN;
    info.iterations = 1;
    if verbose
        skipline()
        disp('Singularity problem! The jacobian matrix of the stacked model cannot be inverted.')
    end
    return
end

i_cols = ipcn+(maximum_lag-1)*ny;
i_rows = (1:ny)';
for it = (maximum_lag+1):(maximum_lag+periods)
    z(jendo) = Y(i_cols);
    z(jexog) = transpose(exogenousvariables(it,:));
    m = m + nv;
    res(i_rows) = jacobian*z;
    i_rows = i_rows + ny;
    i_cols = i_cols + ny;
end

ERR = max(abs(res));

if verbose
    fprintf('Iter: %s,\t Initial err. = %s,\t err. = %s,\t time = %s\n', num2str(1), num2str(err), num2str(ERR), num2str(etime(clock,h2)));
    printline(80);
end

if any(isnan(res)) || any(isinf(res)) || any(isnan(Y)) || any(isinf(Y)) || ~isreal(res) || ~isreal(Y)
    info.status = false;% NaN or Inf occurred
    info.error = ERR;
    info.iterations = 1;
    endogenousvariables = reshape(Y, ny, periods+maximum_lag+M.maximum_lead);
    if verbose
        skipline()
        if ~isreal(res) || ~isreal(Y)
            disp('Simulation terminated with imaginary parts in the residuals or endogenous variables.')
        else
            disp('Simulation terminated with NaN or Inf in the residuals or endogenous variables.')
        end
        disp('There is most likely something wrong with your model. Try model_diagnostics or another simulation method.')
    end
else
    info.status = true;% Convergency obtained.
    info.error = ERR;
    info.iterations = 1;
    endogenousvariables = bsxfun(@plus, reshape(Y, ny, periods+maximum_lag+M.maximum_lead), steadystate_y);
end

if verbose
    skipline();
end