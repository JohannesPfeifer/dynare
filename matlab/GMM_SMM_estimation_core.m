function [M_,oo_,estim_params_,bayestopt_,dataset_,dataset_info]=GMM_SMM_estimation_core(var_list_,M_,options_,oo_,estim_params_,bayestopt_,dataset_,dataset_info,case_string)
% function [M_,oo_,estim_params_,bayestopt_,dataset_,dataset_info]=GMM_SMM_estimation_core(var_list_,M_,options_,oo_,estim_params_,bayestopt_,dataset_,dataset_info,case_string)
% Estimates the model using GMM or SMM
%
% INPUTS
%   var_list_:          selected endogenous variables vector
%   M_:                 [structure] decribing the model
%   options_:           [structure] describing the options
%   oo__                [structure] storing the results
%   estim_params_:      [structure] characterizing parameters to be estimated
%   bayestopt_:         [structure] describing the priors
%   dataset_:           [dseries] object storing the dataset
%   dataset_info:       [structure] storing informations about the sample.
%
% OUTPUTS
%   M_:                 [structure] decribing the model
%   oo__                [structure] storing the results
%   estim_params_:      [structure] characterizing parameters to be estimated
%   bayestopt_:         [structure] describing the priors
%   dataset_:           [dseries] object storing the dataset
%   dataset_info:       [structure] storing informations about the sample.
%
% SPECIAL REQUIREMENTS
%   none

% Copyright (C) 2017 Dynare Team
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


if strcmp(case_string,'GMM')
    objective_function = str2func('GMM_Objective_Function');
elseif strcmp(case_string,'SMM')
    objective_function = str2func('SMM_Objective_Function');
end

[dataset_,dataset_info,xparam1, M_, options_, oo_, estim_params_,bayestopt_,bounds] = initialize_GMM_SMM_estimation(var_list_, M_, options_, oo_, estim_params_,case_string);
%% Set the names of the priors.
pnames = ['     '; 'beta '; 'gamm '; 'norm '; 'invg '; 'unif '; 'invg2'; '     '; 'weibl'];

oo_ = initial_GMM_SMM_estimation_checks(objective_function,xparam1,dataset_,M_,estim_params_,options_,bayestopt_,bounds,oo_,case_string);
if options_.(lower(case_string)).recursive_estimation
    gmm_smm_orders=[1:options_.(lower(case_string)).order];
    fprintf('\nDoing Recursive %s Approximation, Starting at Order 1.\n\n',case_string);
else
    gmm_smm_orders=options_.(lower(case_string)).order;
end
options_.order=gmm_smm_orders(1);
[xparam1, fval, exitflag, hh, options_] = dynare_minimize_objective(objective_function,xparam1,options_.mode_compute,options_,[bounds.lb bounds.ub],bayestopt_.name,bayestopt_,[],dataset_,options_,M_,estim_params_,bayestopt_,bounds,oo_);
fprintf('\nObjective function at miminum with %s: %f.\n',options_.(lower(case_string)).weighting_matrix,fval)
if options_.(lower(case_string)).verbose
    oo_=display_estimation_results_table(xparam1,NaN(size(xparam1)),M_,options_,estim_params_,bayestopt_,oo_,pnames,case_string,[(lower(case_string)),'_temp']);
end
if strcmp('GMM','case_string')
    if ~(strcmp('optimal',options_.(lower(case_string)).weighting_matrix) || strcmp('diagonal',options_.(lower(case_string)).weighting_matrix))
    %no optimal matrix selected, do not repeat
        if length(gmm_smm_orders)==1
            gmm_smm_orders=[];
        end
    end
elseif strcmp('SMM','case_string')
    if length(gmm_smm_orders)==1
        %nothing to do anymore
        gmm_smm_orders=[];
    else
        %first iteration already done
        gmm_smm_orders(1)=[];
    end
end
for ii=gmm_smm_orders
    %%compute new weighting matrix at old order
    if strcmp('case_string','GMM')
        switch options_.(lower(case_string)).weighting_matrix
            case 'optimal'
                fprintf('\nRepeat optimization with optimal weighting matrix at order %u\n',ii);
                oo_.(lower(case_string)).W=get_Optimal_Weighting_Matrix_GMM_SMM(xparam1,dataset_,options_,M_,estim_params_,bayestopt_,oo_,bounds,case_string);
            case 'diagonal'
                oo_.(lower(case_string)).W=diag(get_Optimal_Weighting_Matrix_GMM_SMM(xparam1,dataset_,options_,M_,estim_params_,bayestopt_,oo_,bounds,case_string));
        end
    end
    options_.order=ii;%set order
    [xparam1, fval, exitflag, hh, options_] = dynare_minimize_objective(objective_function,xparam1,options_.mode_compute,options_,[bounds.lb bounds.ub],bayestopt_.name,bayestopt_,[],dataset_,options_,M_,estim_params_,bayestopt_,bounds,oo_);
    if strcmp(case_string,'SMM')
        oo_=get_SMM_moments_matrices(xparam1,dataset_,options_,M_,estim_params_,bayestopt_,oo_,bounds);
    elseif strcmp(case_string,'GMM')
        oo_=get_GMM_moments_matrices(xparam1,dataset_,options_,M_,estim_params_,bayestopt_,oo_,bounds);
    end
    fprintf('\n%s Approximation Order: %d \n',case_string,options_.order)
    fprintf('Objective function at miminum: %u\n',fval)
    if options_.(lower(case_string)).verbose && ~(ii==max(gmm_smm_orders)) % if verbose and not last run
        oo_.(lower(case_string)).SE = get_Standard_Errors_GMM_SMM(xparam1,dataset_,options_,M_,estim_params_,bayestopt_,oo_,bounds,case_string);
        oo_=display_estimation_results_table(xparam1,oo_.(lower(case_string)).SE,M_,options_,estim_params_,bayestopt_,oo_,pnames,case_string,case_string);
    end
end
if strcmp('case_string','GMM') && strcmp('optimal',options_.(lower(case_string)).weighting_matrix)
    %optimal one was used
    oo_.(lower(case_string)).Wopt=oo_.(lower(case_string)).W;
else
    %compute optimal weighting matrix for Standard error computation
    oo_.(lower(case_string)).Wopt=get_Optimal_Weighting_Matrix_GMM_SMM(xparam1,dataset_,options_,M_,estim_params_,bayestopt_,oo_,bounds,case_string);
end
oo_.(lower(case_string)).SE = get_Standard_Errors_GMM_SMM(xparam1,dataset_,options_,M_,estim_params_,bayestopt_,oo_,bounds,case_string);
oo_=display_estimation_results_table(xparam1,oo_.(lower(case_string)).SE,M_,options_,estim_params_,bayestopt_,oo_,pnames,case_string,case_string);
if options_.irf
    if isempty(var_list_)
        oo_.(lower(case_string)).GIRF=get_GIRFs(oo_,M_,options_,M_.endo_names);
    else
        oo_.(lower(case_string)).GIRF=get_GIRFs(oo_,M_,options_,var_list_);
    end
end
M_ = set_all_parameters(xparam1,estim_params_,M_);
oo_.(lower(case_string)).J_test = compute_J_statistic_GMM_SMM(xparam1,dataset_,options_,M_,estim_params_,bayestopt_,oo_,bounds,case_string);
[M_, options_, oo_] = finish_GMM_SMM_estimation(M_, options_, oo_,case_string); %currrently not used as we do not return options_