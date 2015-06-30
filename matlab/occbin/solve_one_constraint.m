function [zdatalinear_ zdatapiecewise_ zdatass_ oobase_ Mbase_  ] = ...
    solve_one_constraint(Mbase_, oobase_, options)

% solve_one_constraint [zdatalinear zdatapiecewise zdatass oo base M
% base] = solve one constraint(modnam, modnamstar, constraint, constraint
% relax, shockssequence, irfshock, nperiods, maxiter, init, Mbase_,
% oobase_, options);
% 
% Inputs: 
% modnam: name of .mod file for the reference regime (excludes the .mod extension).
% modnamstar: name of .mod file for the alternative regime (excludes the .mod exten- sion).
% constraint: the constraint (see notes 1 and 2 below). When the condition in constraint evaluates to true, the solution switches from the reference to the alternative regime.
% constraint relax: when the condition in constraint relax evaluates to true, the solution returns to the reference regime.
% shockssequence: a sequence of unforeseen shocks under which one wants to solve the model (size T×nshocks).
% irfshock: label for innovation for IRFs, from Dynare .mod file (one or more of the ?varexo?).
% nperiods: simulation horizon (can be longer than the sequence of shocks defined in shockssequence; must be long enough to ensure convergence back to the reference model at the end of the simulation horizon and may need to be varied depending on the sequence of shocks).
% maxiter: maximum number of iterations allowed for the solution algorithm (20 if not specified).
% init:	the initial position for the vector of state variables, in deviation from steady state (if not specified, the default is steady state). The ordering follows the definition order in the .mod files.
%
% Outputs:
% zdatalinear: an array containing paths for all endogenous variables ignoring the occasionally binding constraint (the linear solution), in deviation from steady state. Each column is a variable, the order is the definition order in the .mod files.
% zdatapiecewise: an array containing paths for all endogenous variables satisfying the occasionally binding constraint (the occbin/piecewise solution), in deviation from steady state. Each column is a variable, the order is the definition order in the .mod files.
% zdatass: theinitialpositionforthevectorofstatevariables,indeviationfromsteady state (if not specified, the default is a vectors of zero implying that the initial conditions coincide with the steady state). The ordering follows the definition order in the .mod files.
% oobase,Mbase: structures produced by Dynare for the reference model ? see Dynare User Guide.

% Log of changes:
% 6/17/2013 -- Luca added a trailing underscore to local variables in an
% attempt to avoid conflicts with parameter names defined in the .mod files
% to be processed.
% 6/17/2013 -- Luca replaced external .m file setss.m
% 5/31/2015 -- Michel brought file in Dynare. Further changes recorded in Git


zdatalinear_ = [];
zdatapiecewise_ = [];
zdatass_ = [];

errlist_ = [];


% get constraints
[ieq_base,ieq_alt,constr_base,constr_alt] = ...
    get_occbin_constraints(Mbase_,oobase_.steady_state,options.ramsey_policy)
z = repmat(oobase_.steady_state,3,1);
x = zeros(1,Mbase_.exo_nbr);
[residuals,jacobia_] = evaluate_model(z,x,Mbase_,oobase_.steady_state);
dr = set_state_space(oobase_.dr,Mbase_,options);
dr.ys = oobase_.steady_state;

[dr,info] = dyn_first_order_solver(jacobia_(ieq_base,:),Mbase_,dr,options,0);

% Create steady state values of the variables if needed for processing the constraint
for i=1:Mbase_.endo_nbr
   eval([deblank(Mbase_.endo_names(i,:)) '_ss = dr.ys(i); ']);
end

% parse the .mod file for the alternative regime
%eval(['dynare ',modnamstar_,' noclearall nolog '])
%oostar_ = oo_;
%Mstar_ = M_;


nvars_ = Mbase_.endo_nbr;
zdatass_ = dr.ys;

% get the matrices holding the first derivatives for the model
% each regime is treated separately
[cof_, Jbarmat_] = get_coef(jacobia_(ieq_base,:),Mbase_);
[cofstar_, Jstarbarmat_] = get_coef(jacobia_(ieq_alt,:),Mbase_);
Dstartbarmat_ = residuals(ieq_alt);

%[hm1_,h_,hl1_,Jbarmat_] = get_deriv(Mbase_,zdatass_);
%cof_ = [hm1_,h_,hl1_];

%[hm1_,h_,hl1_,Jstarbarmat_,resid_] = get_deriv(Mstar_,zdatass_);
%cofstar_ = [hm1_,h_,hl1_];
%Dstartbarmat_ = resid_;

if isfield(Mbase_,'nfwrd')
    % the latest Dynare distributions have moved nstatic and nfwrd
    [decrulea_,decruleb_]=get_pq(dr,Mbase_.nstatic,Mbase_.nfwrd);
else
    [decrulea_,decruleb_]=get_pq(dr,dr.nstatic,dr.nfwrd);
end

endog_ = Mbase_.endo_names;
exog_ =  Mbase_.exo_names;


% processes the constraints specified in the call to this function
% uppend a suffix to each endogenous variable
%constraint_difference_ = process_constraint(constraint_,'_difference',Mbase_.endo_names,0);

%constraint_relax_difference_ = process_constraint(constraint_relax_,'_difference',Mbase_.endo_names,0);
shockssequence_ = zeros(60,1);
shockssequence_(10) = 0.02; 
shockssequence_(30) = -0.02; 

irfshock_ =char('erra');      % label for innovation for IRFs

nshocks_ = size(shockssequence_,1);

% if necessary, set default values for optional arguments
if ~exist('init_')
    init_ = zeros(nvars_,1);
end

if ~exist('maxiter_')
    maxiter_ = 20;
end

if ~exist('nperiods_')
    nperiods_ = 100;
end


% set some initial conditions and loop through the shocks 
% period by period
init_orig_ = init_;
zdatapiecewise_ = zeros(nperiods_,nvars_);
wishlist_ = endog_;
nwishes_ = size(wishlist_,1);
violvecbool_ = zeros(nperiods_,1);


for ishock_ = 1:nshocks_
    
    changes_=1;
    iter_ = 0;
    
    
    while (changes_ & iter_<maxiter_)
        iter_ = iter_ +1;
        
        % analyze when each regime starts based on current guess
        [regime regimestart]=map_regime(violvecbool_);
        
        
        
        % get the hypothesized piece wise linear solution
        [zdatalinear_]=mkdatap_anticipated(nperiods_,decrulea_,decruleb_,...
            cof_,Jbarmat_,cofstar_,Jstarbarmat_,Dstartbarmat_,...
            regime,regimestart,violvecbool_,...
            endog_,exog_,irfshock_,shockssequence_(ishock_,:),init_);
        
        %for i_indx_=1:nwishes_
        %    eval([deblank(wishlist_(i_indx_,:)),'_difference=zdatalinear_(:,i_indx_);']);
        %end
        
        %disp(zdatalinear_)
        %pause
        
        newviolvecbool_ = test_constraint(zdatalinear_(1:end-1,:),constr_base);       
        relaxconstraint_ = test_constraint(zdatalinear_(1:end-1,:),constr_alt);       
    
        %newviolvecbool_ = eval(constraint_difference_);
        %relaxconstraint_ = eval(constraint_relax_difference_);
        
        
        
        % check if changes to the hypothesis of the duration for each
        % regime
        if (max(newviolvecbool_-violvecbool_>0)) | sum(relaxconstraint_(find(violvecbool_==1))>0)
            changes_ = 1;
        else
            changes_ = 0;
        end
        
        
        
        violvecbool_ = (violvecbool_|newviolvecbool_)-(relaxconstraint_ & violvecbool_);
                
        
    end
    
    init_ = zdatalinear_(1,:);
    zdatapiecewise_(ishock_,:)=init_;
    init_= init_';
    
    % reset violvecbool_ for next period's shock -- this resetting is 
    % consistent with expecting no additional shocks
    violvecbool_=[violvecbool_(2:end);0];
    
end

% if necessary, fill in the rest of the path with the remainder of the 
% last IRF computed.
zdatapiecewise_(ishock_+1:end,:)=zdatalinear_(2:nperiods_-ishock_+1,:);

% get the linear responses
zdatalinear_ = mkdata(max(nperiods_,size(shockssequence_,1)),...
                  decrulea_,decruleb_,endog_,exog_,...
                  wishlist_,irfshock_,shockssequence_,init_orig_);

if changes_ ==1
    display('Did not converge -- increase maxiter_')
end
