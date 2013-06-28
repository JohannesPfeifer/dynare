function [dataset_,xparam1, M_, options_, oo_, estim_params_,SMMinfo_] = initialize_SMM_estimation(var_list_, dname, M_, options_, oo_, estim_params_)

% function [dataset_,xparam1, M_, options_, oo_, estim_params_,SMMinfo_] = initialize_SMM_estimation(var_list_, dname, M_, options_, oo_, estim_params_)
% performs initialization tasks before SMM estimation 
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
%   SMMinfo_:       Matlab's structure describing the SMM parameter options (initialized by dynare, see @ref{SMMinfo_}).

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
options_.smm.old_irf=options_.irf;
options_.smm.old_order=options_.order;

if options_.gmm.order>1 && ~options_.pruning
    fprintf('SMM only works with pruning. Set pruning option to 1.\n')
    options_.options_.pruning=1;
end

% Decide if a DSGE or DSGE-VAR has to be estimated.
if ~isempty(strmatch('dsge_prior_weight',M_.param_names))
    error('SMM does not work with DSGE-VAR')
end
var_list_ = check_list_of_variables(options_, M_, var_list_);
options_.varlist = var_list_;

%set approximation order to SMM approximation order
options_.order=options_.smm.order;

% If options_.lik_init == 1 set by default options_.qz_criterium to 1-1e-6
%  and check options_.qz_criterium < 1-eps if options_.lik_init == 1
% Else set by default options_.qz_criterium to 1+1e-6
if isempty(options_.qz_criterium)
    options_.qz_criterium = 1-1e-6;
end

% If uncentered moments are requested, demean the data 
% the data are prefiltered then there must not be constants in the
% measurement equation of the DSGE model or in the DSGE-VAR model.
if options_.smm.centeredmoments
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
    [xparam1,estim_params_,SMMinfo_,lb,ub,M_] = set_prior(estim_params_,M_,options_);
    if any(SMMinfo_.pshape > 0) % prior specified
        if any(setdiff([0;SMMinfo_.pshape],[0,3]))
            if ~options_.smm.use_prior
                fprintf('\nPriors were specified, but the use_prior-option was not set.\n')
                fprintf('I set use_prior to 1. Conducting SMM with Prior.\n')
                options_.smm.use_prior=1;
            end
                fprintf('\nNon-normal priors specified. SMM with priors uses a Laplace type of approximation.\n')
                fprintf('Only the prior mean and standard deviation are relevant, all other shape information, except for the parameter bounds, is ignored.\n\n')
        end
        if any(isinf(SMMinfo_.p2))
            inf_var_pars=SMMinfo_.name(isinf(SMMinfo_.p2));
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
        outside_bound_vars=SMMinfo_.name([find(xparam1 < bounds(:,1)); find(xparam1 > bounds(:,2))],:);
        disp_string=[outside_bound_vars{1,:}];
        for ii=2:size(outside_bound_vars,1)
            disp_string=[disp_string,', ',outside_bound_vars{ii,:}];
        end
        error(['Initial value(s) of ', disp_string ,' are outside parameter bounds. Potentially, you should set prior_trunc=0. If you used the mode_file-option, check whether your mode-file is consistent with the priors.'])
    end
    lb = bounds(:,1);
    ub = bounds(:,2);
    SMMinfo_.lb = lb;
    SMMinfo_.ub = ub;
else% If estim_params_ is empty (e.g. when running the smoother on a calibrated model)
    error('SMM: the ''estimated_params'' block is mandatory')
end

% Is there a linear trend in the measurement equation?
if ~isfield(options_,'trend_coeffs') % No!
    SMMinfo_.with_trend = 0;
else% Yes!
    SMMinfo_.with_trend = 1;
    SMMinfo_.trend_coeff = {};
    trend_coeffs = options_.trend_coeffs;
    nt = length(trend_coeffs);
    for i=1:n_varobs
        if i > length(trend_coeffs)
            SMMinfo_.trend_coeff{i} = '0';
        else
            SMMinfo_.trend_coeff{i} = trend_coeffs{i};
        end
    end
    error('SMM does not allow for trend in data')
end

% Get informations about the variables of the model.
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
  SMMinfo_.varsindex=varsindex;    
end
oo_.smm.y_label=M_.endo_names(varsindex,:);

SMMinfo_.ny=length(SMMinfo_.varsindex);

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

if options_.smm.simulation_multiple<1
    fprintf('The simulation horizon is shorter than the data. Set the multiple to 2.\n')
    options_.smm.simulation_multiple=2;
end

options_.smm.long=round(options_.smm.simulation_multiple*options_.nobs);
oo_.smm.variance_correction_factor=(1+1/options_.smm.simulation_multiple);

%% draw shocks for SMM
smmstream = RandStream('mt19937ar','Seed',options_.smm.seed);
% eliminate shocks with 0 variance
temp_shocks=randn(smmstream,options_.smm.long+options_.smm.drop,M_.exo_nbr);
if options_.smm.bounded_support==1
    temp_shocks(temp_shocks>2)=2;
    temp_shocks(temp_shocks<-2)=-2;
end
oo_.smm.shock_series=temp_shocks;


if max(options_.smm.autolag)>options_.nobs+1
    error('SMM Error: Data set is too short to compute second moments');
end

fprintf('\n---------------------------------------------------\n')
fprintf('Conducting SMM estimation at order %u\n', options_.smm.order)
fprintf('The simulation horizon is %u periods.\n', options_.smm.long)

if options_.smm.use_prior
    fprintf('Using SMM with priors \n');
end

if options_.smm.centeredmoments
    fprintf('Using centered moments\n')
else
    fprintf('Using uncentered moments\n')
end

%% check if selector matrix is correct
if options_.smm.centeredmoments && ~isempty(options_.smm.firstmoment_selector) % if centered but specified, ignore it
        fprintf('Centered moments requested. First moment selector is ignored\n')
        options_.smm.firstmoment_selector=zeros(SMMinfo_.ny,1);    
elseif ~options_.smm.centeredmoments && ~isempty(options_.smm.firstmoment_selector)  % if not uncentered and specified, check it
    firstmoment_selector=options_.smm.firstmoment_selector;
    if length(firstmoment_selector)~=SMMinfo_.ny
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
    if options_.smm.centeredmoments
        options_.smm.firstmoment_selector=zeros(SMMinfo_.ny,1);
    else 
        options_.smm.firstmoment_selector=ones(SMMinfo_.ny,1);
        fprintf('Using the first moments of all variables\n');
    end
end

n_auto_cov=length(options_.smm.autolag);
if n_auto_cov>0
    autocov_string=[num2str(options_.smm.autolag(1))];
    for ii=2:n_auto_cov
        autocov_string=[autocov_string,', ',num2str(options_.smm.autolag(ii))];
    end
    fprintf('Using auto-covariances at order: %s \n', autocov_string);
else
    fprintf('Using no auto-covariances\n');
end   
    
if options_.smm.optimal_weighting
    fprintf('Using optimal weighting matrix with Newey-West standard errors of lag order: %d\n\n', options_.smm.qLag);
else
    fprintf('Using diagonal weighting matrix\n\n');
end
    
[momentstomatch, E_y, E_yy, autoE_yy, m_data] = moments_SMM_Data(dataset_.data',options_);
SMMinfo_.numMom = size(momentstomatch,1); %Number of moments
oo_.smm.datamoments.momentstomatch=momentstomatch;
oo_.smm.datamoments.E_y=E_y;
oo_.smm.datamoments.E_yy=E_yy;
oo_.smm.datamoments.autoE_yy=autoE_yy;
oo_.smm.datamoments.m_data=m_data;

% Setting the initial weighting matrix
oo_.smm.W = eye(size(momentstomatch,1));


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

% if all(abs(oo_.steady_state(SMMinfo_.mfys))<1e-9)
%     options_.noconstant = 1;
% else
%     options_.noconstant = 0;
% end


