function [dataset_,dataset_info,xparam1, M_, options_, oo_, estim_params_,bayestopt_,bounds] = initialize_GMM_estimation(var_list_, M_, options_, oo_, estim_params_, GMM_SMM_indicator)

% function [dataset_,dataset_info,xparam1, M_, options_, oo_, estim_params_,bayestopt_] = initialize_GMM_estimation(var_list_, M_, options_, oo_, estim_params_)
% performs initialization tasks before GMM estimation 
%
% INPUTS
%   var_list_:      selected endogenous variables vector
%   M_:             Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).
%   options_:       Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   oo_:            Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%   estim_params_:  Matlab's structure describing the estimated_parameters (initialized by dynare, see @ref{estim_params_}).
%   GMM_SMM_indicator         string indicating SMM or GMM
%
% OUTPUTS
%   dataset_:       data after required transformation
%   dataset_info:   Various informations about the dataset (descriptive statistics and missing observations).
%   xparam1:        initial value of estimated parameters as returned by set_prior()
%   M_:             Matlab's structure describing the Model (initialized by dynare, see @ref{M_}).
%   options_:       Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   oo_:            Matlab's structure gathering the results (initialized by dynare, see @ref{oo_}).
%   estim_params_:  Matlab's structure describing the estimated_parameters (initialized by dynare, see @ref{estim_params_}).
%   bayestopt_:     Matlab's structure describing the GMM parameter options (initialized by dynare, see @ref{bayestopt_}).
%   bounds:         structure containing prior bounds

% SPECIAL REQUIREMENTS
%   none

% Copyright (C) 2013-2017 Dynare Team
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


%save old options
options_.(lower(GMM_SMM_indicator)).old_irf=options_.irf;
options_.(lower(GMM_SMM_indicator)).old_order=options_.order;

if options_.(lower(GMM_SMM_indicator)).order>1 && ~options_.pruning
    fprintf('%s at higher order only works with pruning. Set pruning option to 1.\n',GMM_SMM_indicator)
    options_.pruning=1;
end

%set approximation order to GMM/SMM approximation order
options_.order=options_.(lower(GMM_SMM_indicator)).order;

% Get the list of the endogenous variables for which posterior statistics wil be computed
var_list_ = check_list_of_variables(options_, M_, var_list_);
options_.varlist = var_list_;

%set qz_criterion
options_=select_qz_criterium_value(options_);

% If uncentered moments are requested, demean the data 
if options_.(lower(GMM_SMM_indicator)).centeredmoments
    options_.prefilter = 1;
end

% Set priors over the estimated parameters.
if ~isempty(estim_params_) && ~(isfield(estim_params_,'nvx') && (size(estim_params_.var_exo,1)+size(estim_params_.var_endo,1)+size(estim_params_.corrx,1)+size(estim_params_.corrn,1)+size(estim_params_.param_vals,1))==0)
    [xparam1,estim_params_,bayestopt_,lb,ub,M_] = set_prior(estim_params_,M_,options_);
    if ~isempty(bayestopt_) && any(bayestopt_.pshape==0) && any(bayestopt_.pshape~=0)
        error('Estimation must be either fully classical or fully Bayesian. Maybe you forgot to specify a prior distribution.')
    end
    if any(bayestopt_.pshape > 0) % prior specified
        if any(setdiff([0;bayestopt_.pshape],[0,3]))
            if ~options_.(lower(GMM_SMM_indicator)).penalized_estimator
                fprintf('\nPriors were specified, but the penalized_estimator-option was not set.\n')
                fprintf('I set penalized_estimator to 1. Conducting %s with penalty.\n',GMM_SMM_indicator)
                options_.(lower(GMM_SMM_indicator)).penalized_estimator=1;
            end
            fprintf('\nNon-normal priors specified. %s with penalty uses a Laplace type of approximation.\n',GMM_SMM_indicator)
            fprintf('Only the prior mean and standard deviation are relevant, all other shape information, except for the parameter bounds, is ignored.\n\n')
        end
        if any(isinf(bayestopt_.p2))
            inf_var_pars=bayestopt_.name(isinf(bayestopt_.p2));
            disp_string=[inf_var_pars{1,:}];
            for ii=2:size(inf_var_pars,1)
                disp_string=[disp_string,', ',inf_var_pars{ii,:}];
            end
            fprintf('The parameter(s) %s have infinite prior variance. This implies a flat prior\n',disp_string)
            fprintf('I disable the matrix singularity warning\n')
            warning('off','MATLAB:singularMatrix');
        end
    end
end

%check for calibrated covariances before updating parameters
if ~isempty(estim_params_) && ~(isfield(estim_params_,'nvx') && sum(estim_params_.nvx+estim_params_.nvn+estim_params_.ncx+estim_params_.ncn+estim_params_.np)==0)
    estim_params_=check_for_calibrated_covariances(xparam1,estim_params_,M_);
end

%%read out calibration that was set in mod-file and can be used for initialization
xparam1_calib=get_all_parameters(estim_params_,M_); %get calibrated parameters
if ~any(isnan(xparam1_calib)) %all estimated parameters are calibrated
    estim_params_.full_calibration_detected=1;
else
    estim_params_.full_calibration_detected=0;
end
if options_.use_calibration_initialization %set calibration as starting values
    if ~isempty(bayestopt_) && all(bayestopt_.pshape==0) && any(all(isnan([xparam1_calib xparam1]),2))
        error('Estimation: When using the use_calibration option with %s without prior, the parameters must be properly initialized.',GMM_SMM_indicator)
    else
        [xparam1,estim_params_]=do_parameter_initialization(estim_params_,xparam1_calib,xparam1); %get explicitly initialized parameters that have precedence to calibrated values
    end
end

if ~isempty(bayestopt_) && all(bayestopt_.pshape==0) && any(isnan(xparam1))
    error('%s without penalty requires all estimated parameters to be initialized, either in an estimated_params or estimated_params_init-block ',GMM_SMM_indicator)
end

if ~isempty(estim_params_) && ~(all(strcmp(fieldnames(estim_params_),'full_calibration_detected'))  || (isfield(estim_params_,'nvx') && sum(estim_params_.nvx+estim_params_.nvn+estim_params_.ncx+estim_params_.ncn+estim_params_.np)==0))
    if ~isempty(bayestopt_) && any(bayestopt_.pshape > 0)
        % Plot prior densities.
        if ~options_.nograph && options_.plot_priors
            plot_priors(bayestopt_,M_,estim_params_,options_)
        end
        % Set prior bounds
        bounds = prior_bounds(bayestopt_, options_.prior_trunc);
        bounds.lb = max(bounds.lb,lb);
        bounds.ub = min(bounds.ub,ub);
    else  % estimated parameters but no declared priors
          % No priors are declared so Dynare will estimate the model by
          % maximum likelihood with inequality constraints for the parameters.
        bounds.lb = lb;
        bounds.ub = ub;
    end
    % Test if initial values of the estimated parameters are all between the prior lower and upper bounds.
    if options_.use_calibration_initialization
        try
            check_prior_bounds(xparam1,bounds,M_,estim_params_,options_,bayestopt_)
        catch
            e = lasterror();
            fprintf('Cannot use parameter values from calibration as they violate the prior bounds.')
            rethrow(e);
        end
    else
        check_prior_bounds(xparam1,bounds,M_,estim_params_,options_,bayestopt_)
    end
end


% storing prior parameters in results
oo_.prior.mean = bayestopt_.p1;
oo_.prior.mode = bayestopt_.p5;
oo_.prior.variance = diag(bayestopt_.p2.^2);
oo_.prior.hyperparameters.first = bayestopt_.p6;
oo_.prior.hyperparameters.second = bayestopt_.p7;

% Is there a linear trend in the measurement equation?
if ~isfield(options_,'trend_coeffs') % No!
    bayestopt_.with_trend = 0;
else% Yes!
    bayestopt_.with_trend = 1;
    bayestopt_.trend_coeff = {};
    for i=1:options_.number_of_observed_variables
        if i > length(options_.trend_coeffs)
            bayestopt_.trend_coeff{i} = '0';
        else
            bayestopt_.trend_coeff{i} = options_.trend_coeffs{i};
        end
    end
    error('SMM does not allow for trend in data',GMM_SMM_indicator)
end

if strcmp('GMM',GMM_SMM_indicator)
    % Get informations about the variables of the model.
    options_.k_order_solver=1;
    bayestopt_.nv=M_.nspred + M_.exo_nbr;
    bayestopt_.nstatic = M_.nstatic;          % Number of static variables.
    bayestopt_.nspred = M_.nspred;             % Number of predetermined variables.
    bayestopt_.nu = M_.exo_nbr;
    bayestopt_.nx = M_.nspred;            % Number of predetermined variables in the state equation.
    bayestopt_.nz = M_.endo_nbr; % Number of control variables + state variables
    bayestopt_.vectorMom3 = zeros(1,bayestopt_.nu);
    bayestopt_.vectorMom4 = ones(1,bayestopt_.nu)*3;
    if options_.gmm.order==3
        bayestopt_.vectorMom5 = zeros(1,bayestopt_.nu);
        bayestopt_.vectorMom6 = ones(1,bayestopt_.nu)*15;
    end
end

%initialize state space including inv_order_var 
oo_.dr = set_state_space(oo_.dr,M_,options_);

if ~isfield(options_,'varobs')
    error('VAROBS statement is missing!')
else
    % Set the number of observed variables.
    options_.number_of_observed_variables = length(options_.varobs);
    % Check that each declared observed variable is also an endogenous variable.
    for i = 1:options_.number_of_observed_variables
        id = strmatch(options_.varobs{i}, M_.endo_names, 'exact');
        if isempty(id)
            error(['Unknown variable (' options_.varobs{i} ')!'])
        end
    end
    % Check that a variable is not declared as observed more than once.
    if length(unique(options_.varobs))<length(options_.varobs)
        for i = 1:options_.number_of_observed_variables
            if length(strmatch(options_.varobs{i},options_.varobs,'exact'))>1
                error(['A variable cannot be declared as observed more than once (' options_.varobs{i} ')!'])
            end
        end
    end
    varsindex=[];
    for ii = 1:size(options_.varobs,2)
        varname = deblank(options_.varobs(1,ii));
        for jj=1:M_.orig_endo_nbr
            if strcmp(varname,deblank(M_.endo_names(jj,:)))
                varsindex=[varsindex; jj];
            end
        end
    end
    if strcmp('GMM',GMM_SMM_indicator)
        bayestopt_.control_indices=oo_.dr.inv_order_var(varsindex);% variables in matrices are in order_var ordering and need to be mapped to declaration order using inv_order_var
    elseif strcmp('SMM',GMM_SMM_indicator)
        bayestopt_.varsindex=varsindex;
    end
end

if strcmp('GMM',GMM_SMM_indicator)
    bayestopt_.state_indices=[M_.nstatic+1:M_.nstatic+M_.nspred]';
    oo_.gmm.y_label=M_.endo_names(varsindex,:);
    oo_.gmm.v_label=M_.endo_names(oo_.dr.order_var(bayestopt_.state_indices),:);
    bayestopt_.ny=length(bayestopt_.control_indices);
elseif strcmp('SMM',GMM_SMM_indicator)
    oo_.smm.y_label=M_.endo_names(varsindex,:);    
    bayestopt_.ny=length(bayestopt_.varsindex);
end

% Build the dataset
if ~isempty(options_.datafile)
    [pathstr,name,ext] = fileparts(options_.datafile);
    if strcmp(name,M_.fname)
        error('Data-file and mod-file are not allowed to have the same name. Please change the name of the data file.')
    end
end

if isnan(options_.first_obs)
    options_.first_obs=1;
end
[dataset_, dataset_info, newdatainterfaceflag] = makedataset(options_);

%set options for old interface from the ones for new interface
if ~isempty(dataset_)
    options_.nobs = dataset_.nobs;
end

if strcmp('GMM',GMM_SMM_indicator)
    if max(options_.gmm.autolag)>options_.nobs+1
        error('GMM Error: Data set is too short to compute second moments');
    end
elseif strcmp('SMM',GMM_SMM_indicator)
    if options_.smm.simulation_multiple<1
        fprintf('The simulation horizon is shorter than the data. Set the multiple to 2.\n')
        options_.smm.simulation_multiple=2;
    end    
    options_.smm.long=round(options_.smm.simulation_multiple*options_.nobs);
    oo_.smm.variance_correction_factor=(1+1/options_.smm.simulation_multiple);
    %% draw shocks for SMM
    smmstream = RandStream('mt19937ar','Seed',options_.smm.seed);
    temp_shocks=randn(smmstream,options_.smm.long+options_.smm.drop,M_.exo_nbr);
    if options_.smm.bounded_support==1
        temp_shocks(temp_shocks>2)=2;
        temp_shocks(temp_shocks<-2)=-2;
    end
    oo_.smm.shock_series=temp_shocks;    
    if max(options_.smm.autolag)>options_.nobs+1
        error('SMM Error: Data set is too short to compute second moments');
    end
end

fprintf('\n---------------------------------------------------\n')
fprintf('Conducting %s estimation at order %u\n',GMM_SMM_indicator,options_.(lower(GMM_SMM_indicator)).order)

if options_.(lower(GMM_SMM_indicator)).penalized_estimator
    fprintf('Using %s with priors \n',GMM_SMM_indicator);
end

if options_.(lower(GMM_SMM_indicator)).centeredmoments
    fprintf('Using centered moments\n')
else
    fprintf('Using uncentered moments\n')
end

% setting steadystate_check_flag option
if options_.steadystate.nocheck
    steadystate_check_flag = 0;
else
    steadystate_check_flag = 1;
end

%% check steady state
M = M_;
nvx = estim_params_.nvx;
ncx = estim_params_.ncx;
nvn = estim_params_.nvn;
ncn = estim_params_.ncn;
if estim_params_.np
    M.params(estim_params_.param_vals(:,1)) = xparam1(nvx+ncx+nvn+ncn+1:end);
end
[oo_.steady_state, params,info] = evaluate_steady_state(oo_.steady_state,M,options_,oo_,steadystate_check_flag);

if info(1)
    fprintf('\ninitialize_GMM_SMM_estimation:: The steady state at the initial parameters cannot be computed.\n')
    print_info(info, 0, options_);
end

% bayestopt_.mfys: position of observables in oo_.dr.ys (declaration order)
var_obs_index_dr = [];
k1 = [];
for i=1:options_.number_of_observed_variables
    var_obs_index_dr = [var_obs_index_dr; strmatch(options_.varobs{i},M_.endo_names(oo_.dr.order_var,:),'exact')];
    k1 = [k1; strmatch(options_.varobs{i},M_.endo_names, 'exact')];
end
bayestopt_.mfys = k1;

% If the steady state of the observed variables is non zero, set noconstant equal 0 ()
if (~options_.loglinear && all(abs(oo_.steady_state(bayestopt_.mfys))<1e-9)) || (options_.loglinear && all(abs(log(oo_.steady_state(bayestopt_.mfys)))<1e-9))
    options_.noconstant = 1;
else
    options_.noconstant = 0;
    % If the data are prefiltered then there must not be constants in the
    % measurement equation
    if options_.prefilter
        skipline()
        disp('You have specified the option "prefilter" to demean your data but the')
        disp('steady state of of the observed variables is non zero.')
        disp('Either change the measurement equations, by centering the observed')
        disp('variables in the model block, or drop the prefiltering.')
        error('The option "prefilter" is inconsistent with the non-zero mean measurement equations in the model.')
    end
end

%% get the non-zero rows and columns of Sigma_e
Sigma_e_non_zero_rows=find(~all(M_.Sigma_e==0,1));
Sigma_e_non_zero_columns=find(~all(M_.Sigma_e==0,2));
if ~isequal(Sigma_e_non_zero_rows,Sigma_e_non_zero_columns')
    error('Structual error matrix not symmetric')
end
if isfield(estim_params_,'var_exo') && ~isempty(estim_params_.var_exo)
    estim_params_.Sigma_e_entries_to_check_for_positive_definiteness=union(Sigma_e_non_zero_rows,estim_params_.var_exo(:,1));
else
    estim_params_.Sigma_e_entries_to_check_for_positive_definiteness=Sigma_e_non_zero_rows;
end

%% check if selector matrix is correct
if options_.(lower(GMM_SMM_indicator)).centeredmoments && ~isempty(options_.(lower(GMM_SMM_indicator)).firstmoment_selector) % if centered but specified, ignore it
        fprintf('Centered moments requested. First moment selector is ignored\n')
        options_.(lower(GMM_SMM_indicator)).firstmoment_selector=zeros(bayestopt_.ny,1);    
elseif ~options_.(lower(GMM_SMM_indicator)).centeredmoments && ~isempty(options_.(lower(GMM_SMM_indicator)).firstmoment_selector)  % if not uncentered and specified, check it
    firstmoment_selector=options_.(lower(GMM_SMM_indicator)).firstmoment_selector;
    if length(firstmoment_selector)~=bayestopt_.ny
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
    if options_.(lower(GMM_SMM_indicator)).centeredmoments
        options_.(lower(GMM_SMM_indicator)).firstmoment_selector=zeros(bayestopt_.ny,1);
    else 
        options_.(lower(GMM_SMM_indicator)).firstmoment_selector=ones(bayestopt_.ny,1);
        fprintf('Using the first moments of all variables\n');
    end
end

n_auto_cov=length(options_.(lower(GMM_SMM_indicator)).autolag);
if n_auto_cov>0
    autocov_string=[num2str(options_.(lower(GMM_SMM_indicator)).autolag(1))];
    for ii=2:n_auto_cov
        autocov_string=[autocov_string,', ',num2str(options_.(lower(GMM_SMM_indicator)).autolag(ii))];
    end
    fprintf('Using auto-covariances at order: %s \n', autocov_string);
else
    fprintf('Using no auto-covariances\n');
end   
    

    
[momentstomatch, E_y, E_yy, autoE_yy, m_data] = moments_GMM_SMM_Data(dataset_.data,options_);
bayestopt_.numMom = size(momentstomatch,1); %Number of moments
oo_.(lower(GMM_SMM_indicator)).datamoments.momentstomatch=momentstomatch;
oo_.(lower(GMM_SMM_indicator)).datamoments.E_y=E_y;
oo_.(lower(GMM_SMM_indicator)).datamoments.E_yy=E_yy;
oo_.(lower(GMM_SMM_indicator)).datamoments.autoE_yy=autoE_yy;
oo_.(lower(GMM_SMM_indicator)).datamoments.m_data=m_data;

switch options_.(lower(GMM_SMM_indicator)).weighting_matrix
    case 'optimal'
        fprintf('Using optimal weighting matrix with Newey-West standard errors of lag order: %d\n\n', options_.(lower(GMM_SMM_indicator)).qLag);
        if strcmp('GMM_SMM_indicator','GMM')
            % Setting the initial weighting matrix
            oo_.(lower(GMM_SMM_indicator)).W = eye(size(momentstomatch,1));
        elseif strcmp('GMM_SMM_indicator','SMM')
            oo_.(lower(GMM_SMM_indicator)).W=get_Optimal_Weighting_Matrix_GMM_SMM(xparam1,dataset_,options_,M_,estim_params_,bayestopt_,oo_,bounds,case_string);
        end
    case 'diagonal'        
        if strcmp('GMM_SMM_indicator','GMM')
            % Setting the initial weighting matrix
            oo_.(lower(GMM_SMM_indicator)).W = eye(size(momentstomatch,1));
        elseif strcmp('GMM_SMM_indicator','SMM')
            oo_.(lower(GMM_SMM_indicator)).W=diag(get_Optimal_Weighting_Matrix_GMM_SMM(xparam1,dataset_,options_,M_,estim_params_,bayestopt_,oo_,bounds,case_string));
        end
    case 'identity_matrix' %Use identity
        fprintf('Using identity weighting matrix\n');
        % Setting the initial weighting matrix
        oo_.(lower(GMM_SMM_indicator)).W = eye(size(momentstomatch,1));        
    otherwise %user specified matrix in file
        fprintf('Using user-specified weighting matrix\n');
        try
            load(options_.(lower(GMM_SMM_indicator)).weighting_matrix,'weighting_matrix')
            W=weighting_matrix;
        catch
            error(['No matrix named ''weighting_matrix'' could be found in ',options_.(lower(GMM_SMM_indicator)).weighting_matrix,'.mat'])
        end
        [nrow, ncol]=size(W);
        if ~isequal(nrow,ncol) && ~isequal(nrow,nx) %check if square and right size
            error(['jumping_covariweighting_matrixance matrix must be square and have ',num2str(size(momentstomatch,1)),' rows and columns'])
        end
        try %check for positive definiteness
            chol(W);
            hsd = sqrt(diag(W));
            invhess = inv(W./(hsd*hsd'))./(hsd*hsd');
        catch
            error(['Specified weighting_matrix is not positive definite'])
        end
end
