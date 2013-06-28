function [dataset_,xparam1, M_, options_, oo_, estim_params_,GMMinfo_] = initialize_GMM_estimation(var_list_, dname, M_, options_, oo_, estim_params_)

% function [dataset_,xparam1, M_, options_, oo_, estim_params_,GMMinfo_] = initialize_GMM_estimation(var_list_, dname, M_, options_, oo_, estim_params_)
% performs initialization tasks before GMM estimation 
%
% INPUTS
%   var_list_:      selected endogenous variables vector
%   dname:          alternative directory name
%   M_:             Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).
%   options_:       Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   oo_:            Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%   estim_params_:  Matlab's structure describing the estimated_parameters (initialized by dynare, see @ref{estim_params_}).
%
% OUTPUTS
%   dataset_:       data after required transformation
%   xparam1:        initial value of estimated parameters as returned by set_prior()
%   M_:             Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).
%   options_:       Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   oo_:            Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%   estim_params_:  Matlab's structure describing the estimated_parameters (initialized by dynare, see @ref{estim_params_}).
%   GMMinfo_:       Matlab's structure describing the GMM parameter options (initialized by dynare, see @ref{GMMinfo_}).

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


global objective_function_penalty_base

% Set the "size" of penalty.
objective_function_penalty_base = 1e9;


%save old options
options_.gmm.old_irf=options_.irf;
options_.gmm.old_order=options_.order;

% Decide if a DSGE or DSGE-VAR has to be estimated.
if ~isempty(strmatch('dsge_prior_weight',M_.param_names))
    error('GMM does not work with DSGE-VAR')
end
var_list_ = check_list_of_variables(options_, M_, var_list_);
options_.varlist = var_list_;

%set approximation order to GMM approximation order
options_.order=options_.gmm.order;

% If options_.lik_init == 1 set by default options_.qz_criterium to 1-1e-6
%  and check options_.qz_criterium < 1-eps if options_.lik_init == 1
% Else set by default options_.qz_criterium to 1+1e-6
if isempty(options_.qz_criterium)
    options_.qz_criterium = 1-1e-6;
end

% If uncentered moments are requested, demean the data 
% the data are prefiltered then there must not be constants in the
% measurement equation of the DSGE model or in the DSGE-VAR model.
if options_.gmm.centeredmoments
    options_.prefilter = 1;
    options_.noconstant = 1; %needed for smoother
end

% Set the name of the directory where (intermediary) results will be saved.
if isempty(dname)
    M_.dname = M_.fname;
else
    M_.dname = dname;
end

% Set the number of observed variables.
n_varobs = size(options_.varobs,1);

% Set priors over the estimated parameters.
if ~isempty(estim_params_)
    [xparam1,estim_params_,GMMinfo_,lb,ub,M_] = set_prior(estim_params_,M_,options_);
    if any(GMMinfo_.pshape > 0) % prior specified
        if any(setdiff([0;GMMinfo_.pshape],[0,3]))
            if ~options_.gmm.use_prior
                fprintf('\nPriors were specified, but the use_prior-option was not set.\n')
                fprintf('I set use_prior to 1. Conducting GMM with Prior.\n')
                options_.gmm.use_prior=1;
            end
                fprintf('\nNon-normal priors specified. GMM with priors uses a Laplace type of approximation.\n')
                fprintf('Only the prior mean and standard deviation are relevant, all other shape information, except for the parameter bounds, is ignored.\n\n')
        end
        if any(isinf(GMMinfo_.p2))
            inf_var_pars=GMMinfo_.name(isinf(GMMinfo_.p2));
            disp_string=[inf_var_pars{1,:}];
            for ii=2:size(inf_var_pars,1)
                disp_string=[disp_string,', ',inf_var_pars{ii,:}];
            end
            fprintf('The parameter(s) %s have infinite prior variance. This implies a flat prior\n',disp_string)
            fprintf('I disable the matrix singularity warning\n')
            warning('off','MATLAB:singularMatrix');
        end
        bounds(:,1) = lb;
        bounds(:,2) = ub;        
    else
        % No priors are declared so Dynare will estimate the model by
        % maximum likelihood with inequality constraints for the parameters.
        bounds(:,1) = lb;
        bounds(:,2) = ub;
    end
    % Test if initial values of the estimated parameters are all between
    % the prior lower and upper bounds.
    if any(xparam1 < bounds(:,1)) || any(xparam1 > bounds(:,2))
        outside_bound_vars=GMMinfo_.name([find(xparam1 < bounds(:,1)); find(xparam1 > bounds(:,2))],:);
        disp_string=[outside_bound_vars{1,:}];
        for ii=2:size(outside_bound_vars,1)
            disp_string=[disp_string,', ',outside_bound_vars{ii,:}];
        end
        error(['Initial value(s) of ', disp_string ,' are outside parameter bounds. Potentially, you should set prior_trunc=0. If you used the mode_file-option, check whether your mode-file is consistent with the priors.'])
    end
    lb = bounds(:,1);
    ub = bounds(:,2);
    GMMinfo_.lb = lb;
    GMMinfo_.ub = ub;
else% If estim_params_ is empty (e.g. when running the smoother on a calibrated model)
    error('GMM: the ''estimated_params'' block is mandatory')
end

% Is there a linear trend in the measurement equation?
if ~isfield(options_,'trend_coeffs') % No!
    GMMinfo_.with_trend = 0;
else% Yes!
    GMMinfo_.with_trend = 1;
    GMMinfo_.trend_coeff = {};
    trend_coeffs = options_.trend_coeffs;
    nt = length(trend_coeffs);
    for i=1:n_varobs
        if i > length(trend_coeffs)
            GMMinfo_.trend_coeff{i} = '0';
        else
            GMMinfo_.trend_coeff{i} = trend_coeffs{i};
        end
    end
    error('GMM does not allow for trend in data')
end

% Get informations about the variables of the model.
options_.k_order_solver=1;
GMMinfo_.nv=M_.nspred + M_.exo_nbr;
GMMinfo_.nstatic = M_.nstatic;          % Number of static variables.
GMMinfo_.nspred = M_.nspred;             % Number of predetermined variables.
GMMinfo_.nu = M_.exo_nbr;
GMMinfo_.nx = M_.nspred;            % Number of predetermined variables in the state equation.
GMMinfo_.nz = M_.endo_nbr; % Number of control variables + state variables
GMMinfo_.vectorMom3 = zeros(1,GMMinfo_.nu);
GMMinfo_.vectorMom4 = ones(1,GMMinfo_.nu)*3;
if options_.gmm.order==3
    GMMinfo_.vectorMom5 = zeros(1,GMMinfo_.nu); 
    GMMinfo_.vectorMom6 = ones(1,GMMinfo_.nu)*15;
end

%initialize state space including inv_order_var 
oo_.dr = set_state_space(oo_.dr,M_,options_);

% Test if observed variables are declared.
if isempty(options_.varobs)
    error('VAROBS is missing')
else
  varsindex=[];
  for ii = 1:size(options_.varobs,1)
    varname = deblank(options_.varobs(ii,:));
    for jj=1:M_.orig_endo_nbr
      if strcmp(varname,deblank(M_.endo_names(jj,:)))
        varsindex=[varsindex; jj];
      end
    end
  end
  GMMinfo_.control_indices=oo_.dr.inv_order_var(varsindex);% variables in matrices are in order_var ordering and need to be mapped to declaration order using inv_order_var    
end
GMMinfo_.state_indices=[M_.nstatic+1:M_.nstatic+M_.nspred]';
oo_.gmm.y_label=M_.endo_names(varsindex,:);
oo_.gmm.v_label=M_.endo_names(dr.order_var(GMMinfo_.state_indices),:);
GMMinfo_.ny=length(GMMinfo_.control_indices);

% Test if the data file is declared.
if isempty(options_.datafile)
        error('datafile option is missing')
end

% Load and transform data.
transformation = [];
if options_.loglinear && ~options_.logdata
    transformation = @log;
end
xls.sheet = options_.xls_sheet;
xls.range = options_.xls_range;

if ~isfield(options_,'nobs')
    options_.nobs = [];
end

dataset_ = initialize_dataset(options_.datafile,options_.varobs,options_.first_obs,options_.nobs,transformation,options_.prefilter,xls);

options_.nobs = dataset_.info.ntobs;

if max(options_.gmm.autolag)>options_.nobs+1
    error('GMM Error: Data set is too short to compute second moments');
end

fprintf('\n---------------------------------------------------\n')
fprintf('Conducting GMM estimation at order %u\n', options_.gmm.order)

if options_.gmm.use_prior
    fprintf('Using GMM with priors \n');
end

if options_.gmm.centeredmoments
    fprintf('Using centered moments\n')
else
    fprintf('Using uncentered moments\n')
end

%% check if selector matrix is correct
if options_.gmm.centeredmoments && ~isempty(options_.gmm.firstmoment_selector) % if centered but specified, ignore it
        fprintf('Centered moments requested. First moment selector is ignored\n')
        options_.gmm.firstmoment_selector=zeros(GMMinfo_.ny,1);    
elseif ~options_.gmm.centeredmoments && ~isempty(options_.gmm.firstmoment_selector)  % if not uncentered and specified, check it
    firstmoment_selector=options_.gmm.firstmoment_selector;
    if length(firstmoment_selector)~=GMMinfo_.ny
        error('Number of entries in the selector matrix is not equal to the number of observables')
    end
    if ~isempty(setdiff([0;1],[0;1;unique(firstmoment_selector)])) %second 0,1 makes sure that error does not happen for all 1 and 0
        error('Selector Matrix may only contain zeros and ones')
    end
    n_first_mom=sum(firstmoment_selector==1);
    if n_first_mom>0
        first_moment_vars=M_.endo_names(varsindex(firstmoment_selector==1),:);
        first_moment_var_string=[first_moment_vars(1,:)];
        for ii=2:n_first_mom
            first_moment_var_string=[first_moment_var_string,', ',first_moment_vars(ii,:)];
        end
        fprintf('Using the first moments of: %s\n', first_moment_var_string);
    else
        fprintf('Using no first moments\n');
    end   
else %if not specified, set it
    if options_.gmm.centeredmoments
        options_.gmm.firstmoment_selector=zeros(GMMinfo_.ny,1);
    else 
        options_.gmm.firstmoment_selector=ones(GMMinfo_.ny,1);
        fprintf('Using the first moments of all variables\n');
    end
end

n_auto_cov=length(options_.gmm.autolag);
if n_auto_cov>0
    autocov_string=[num2str(options_.gmm.autolag(1))];
    for ii=2:n_auto_cov
        autocov_string=[autocov_string,', ',num2str(options_.gmm.autolag(ii))];
    end
    fprintf('Using auto-covariances at order: %s \n', autocov_string);
else
    fprintf('Using no auto-covariances\n');
end   
    
if options_.gmm.optimal_weighting
    fprintf('Using optimal weighting matrix with Newey-West standard errors of lag order: %d\n\n', options_.gmm.qLag);
else
    fprintf('Using diagonal weighting matrix\n\n');
end
    
[momentstomatch, E_y, E_yy, autoE_yy, m_data] = moments_GMM_Data(dataset_.data',options_);
GMMinfo_.numMom = size(momentstomatch,1); %Number of moments
oo_.gmm.datamoments.momentstomatch=momentstomatch;
oo_.gmm.datamoments.E_y=E_y;
oo_.gmm.datamoments.E_yy=E_yy;
oo_.gmm.datamoments.autoE_yy=autoE_yy;
oo_.gmm.datamoments.m_data=m_data;

% Setting the initial weighting matrix
oo_.gmm.W = eye(size(momentstomatch,1));

% setting noconstant option
% steadystate_check_flag = 1;

% M = M_;
% nvx = estim_params_.nvx;
% ncx = estim_params_.ncx;
% nvn = estim_params_.nvn;
% ncn = estim_params_.ncn;
% if estim_params_.np,
%   M.params(estim_params_.param_vals(:,1)) = xparam1(nvx+ncx+nvn+ncn+1:end);
% end;
% [oo_.steady_state, params] = evaluate_steady_state(oo_.steady_state,M,options_,oo_,steadystate_check_flag);

% if all(abs(oo_.steady_state(GMMinfo_.mfys))<1e-9)
%     options_.noconstant = 1;
% else
%     options_.noconstant = 0;
% end


