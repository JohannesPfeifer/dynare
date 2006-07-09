% Copyright (C) 2001 Michel Juillard
%
function disp_th_moments(dr,var_list)
  global M_ oo_ options_
  
  nvar = size(var_list,1);
  if nvar == 0
    nvar = length(dr.order_var);
    ivar = [1:nvar]';
  else
    ivar=zeros(nvar,1);
    for i=1:nvar
      i_tmp = strmatch(var_list(i,:),M_.endo_names,'exact');
      if isempty(i_tmp)
      	error (['One of the variable specified does not exist']) ;
      else
	ivar(i) = i_tmp;
      end
    end
  end
  
  [oo_.gamma_y,ivar] = th_autocovariances(dr,ivar);
  m = dr.ys(ivar);

  
  i1 = find(abs(diag(oo_.gamma_y{1})) > 1e-12);
  s2 = diag(oo_.gamma_y{1});
  sd = sqrt(s2);
  if options_.order == 2
    m = m+oo_.gamma_y{options_.ar+3};
  end
  
  z = [ m sd s2 ];
  oo_.mean = m;
  oo_.var = oo_.gamma_y{1};
  
  lh = size(deblank(M_.endo_names(ivar,:)),2)+2;
  if options_.nomoments == 0
    title='THEORETICAL MOMENTS';
    if options_.hp_filter == 1
      title = [title ' (HP filter, lambda = ' int2str(options_.hp_filter) ')'];
    end
    headers=strvcat('VARIABLE','MEAN','STD. DEV.','VARIANCE');
    table(title,headers,deblank(M_.endo_names(ivar,:)),z,lh,11,4);
    if M_.exo_nbr > 1
      disp(' ')
      title='VARIANCE DECOMPOSITION (in percent)';
      if options_.hp_filter == 1
	title = [title ' (HP filter, lambda = ' ...
		 int2str(options_.hp_filter) ')'];
      end
      headers = M_.exo_names;
      headers(M_.exo_names_orig_ord,:) = headers;
      headers = strvcat(' ',headers);
      table(title,headers,deblank(M_.endo_names(ivar(i1),:)),100*oo_.gamma_y{options_.ar+2}(i1,:), ...
	    lh,8,2);
    end
  end
  
  if options_.nocorr == 0
    disp(' ')
    title='MATRIX OF CORRELATIONS';
    if options_.hp_filter == 1
      title = [title ' (HP filter, lambda = ' int2str(options_.hp_filter) ')'];
    end
    labels = deblank(M_.endo_names(ivar,:));
    headers = strvcat('Variables',labels(i1,:));
    corr = oo_.gamma_y{1}(i1,i1)./(sd(i1)*sd(i1)');
    table(title,headers,labels(i1,:),corr,lh,8,4);
  end
  
  if options_.ar > 0
    disp(' ')
    title='COEFFICIENTS OF AUTOCORRELATION';
    if options_.hp_filter == 1
      title = [title ' (HP filter, lambda = ' int2str(options_.hp_filter) ')'];
    end
    labels = deblank(M_.endo_names(ivar(i1),:));
    headers = strvcat('Order ',int2str([1:options_.ar]'));
    z=[];
    for i=1:options_.ar
      oo_.autocorr{i} = oo_.gamma_y{i+1};
      z(:,i) = diag(oo_.gamma_y{i+1}(i1,i1));
    end
    table(title,headers,labels,z,0,8,4);
  end
  
% 10/09/02 MJ 
% 10/18/02 MJ added th_autocovariances() and provided for lags on several
% periods
% 10/30/02 MJ added correlations and autocorrelations, uses table()
%             oo_.gamma_y is now a cell array.
% 02/18/03 MJ added subtitles for HP filter
% 05/01/03 MJ corrected options_.hp_filter
% 05/21/03 MJ variance decomposition: test M_.exo_nbr > 1
% 05/21/03 MJ displays only variables with positive variance
