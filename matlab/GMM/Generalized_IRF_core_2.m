function [GIRF] = Generalized_IRF_core(dr,options_,shockSize,vectorMom3,xf,xs)
% [GIRF] = Generalized_IRF_core(dr,options_,shockSize,vectorMom3,xf,xs)
% This function computes closed-form expressions for the impulse response
% functions using the pruning method when using the following definition 
% of the impulse response function.
%
% impulse(x_t+l) = E_t[w_t|eps_t+1=eps_t+1+shockSize, eps_t+2, eps_t+3, ..., eps_t+l]
%                 -E_t[w_t|eps_t+1=eps_t+1          , eps_t+2, eps_t+3, ..., eps_t+l]
%
% Note that this definition is the one behind the 
% so-called Generalized Impulse Reponse Function (GIRF). The definition is
% based on Andreasen (2012), European Economic Review
%
% INPUTS 
%   o dr_                       Matlab's decision rule structure
%   o options_                  Matlab's structure describing the options (initialized by dynare, see @ref{options_}).
%   o shocksize     [double]   (p*1) vector of shocks (measured in standard deviations).
%   o vectorMom3    [double]   (p*1) vector of third moments
%   o xf            [double]   (p*1) vector of first order state at which
%                               to compute IRF
%   o xs            [double]   (p*1) vector of second order state at which
%                               to compute IRF
%
% OUTPUT:
%   o GIRF                  [structure] strucutre containing the GIRFs, including
%        GIRF.y             impulse response function for y = g(x,sig) (sum of all parts)
%
%        GIRF.parts.yf      impulse responses at first order
%        GIRF.parts.ys      impulse responses at second order
%        GIRF.parts.yrd     impulse responses at thrid order
%
%        GIRF.x             impulse response function for x = h(x,sig) (sum of all parts)
%
%        GIRF.parts.xf      impulse responses at first order
%        GIRF.parts.xs      impulse responses at second order
%        GIRF.parts.xrd     impulse responses at third order
%
% NOTE: if xf = [] and xs = [], then we use the default setting and apply
% the mean values of xf and xs.
%
% Based on code by Martin M. Andreasen, 2012
%
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

orderApp=options_.order;
IRFlength=options_.irf;
% read out decision rule coefficients
eta     = dr.eta;
sig     = dr.sig;

if ~isfield(dr,'h_v')
    error('State Space with linear innovations has not been set up')
end
hx      = dr.h_v;
gx      = dr.g_v;

% The dimension of the model
[ny,nx] = size(gx);
ne      = size(eta,2);
GIRFy   = zeros(ny,IRFlength,3);
GIRFx   = zeros(nx,IRFlength,3);

% read out decision rule coefficients
if orderApp>1
    hxx     = dr.h_vv; 
    gxx     = dr.g_vv;
    hss     = dr.h_ss;

    % Defining matrices
    HHxxtil  = 1/2*reshape(hxx,nx,nx^2);
    GGxxtil  = 1/2*reshape(gxx,ny,nx^2);   
end

if orderApp>2
    hxxx    = dr.h_vvv;
    gxxx    = dr.g_vvv;
    hssx    = dr.h_ssv;
    gssx    = dr.g_ssv;
    
    % Defining matrices
    HHxxxtil = 1/6*reshape(hxxx,nx,nx^3);
    GGxxxtil = 1/6*reshape(gxxx,ny,nx^3);
end


% Checking on shockSize
[n1,n2] = size(shockSize);
% ensuring that we have a column vector
if n2 > n1
    shockSize = shockSize';
end
if size(shockSize,1) ~= ne
    error('lenght of shockSize must be equal to ne');
end

%% Computing the mean values for xf and xs
sigeta     = sig*eta;
eta_eta    = eta*eta';
AA         = (eye(nx)-hx)\eye(nx); %inv(eye(nx)-hx);
vec_var_xf = (eye(nx^2)-kron(hx,hx))\reshape(sig^2*eta_eta,nx^2,1); 
var_xf     = reshape(vec_var_xf,nx,nx);
mean_xfxf  = var_xf;
if orderApp>1
    mean_xs    = AA*(0.5*reshape(hxx,nx,nx^2,1)*reshape(mean_xfxf,nx^2,1)+0.5*hss*sig^2);
end
if isempty(xf)
    xf = zeros(nx,1);
end
if isempty(xs) & orderApp>1
    xs = mean_xs;
end

%% Impulse response functions at first order
GIRFy1st = zeros(ny,IRFlength);
GIRFxf   = zeros(nx,IRFlength);
hx_powers = zeros(nx,nx,1+IRFlength);
hx_powers(:,:,1)=eye(nx);
for k=2:IRFlength+1
    hx_powers(:,:,k)     = hx^(k-1);
end

for k=1:IRFlength
    GIRFxf(:,k)   = squeeze(hx_powers(:,:,k))*sigeta*shockSize;
    GIRFy1st(:,k) = gx*GIRFxf(:,k);
end
% Saving output
GIRFy(:,:,1) = GIRFy1st;
GIRFx(:,:,1) = GIRFxf;

GIRF.parts.xf=GIRFxf;
GIRF.y=GIRFy;
GIRF.x=GIRFx;
GIRF.parts.yf=GIRFy1st;

if orderApp == 1
    GIRF.y=squeeze(GIRFy(:,:,1));
    GIRF.x=squeeze(GIRFx(:,:,1));
    return
end

%% Impulse response functions at second order
GIRFy2nd = zeros(ny,IRFlength);
GIRFxs   = zeros(nx,IRFlength);
GIRFxfxf = zeros(nx^2,IRFlength);
consQ    = kron(sigeta*shockSize,sigeta*shockSize);
% Computing GIRFxfxf
h = waitbar(0/IRFlength,'Computing GIRF for x1st^2');
for k=1:IRFlength
    GIRFxfxf(:,k) = kron(squeeze(hx_powers(:,:,k+1))*xf,squeeze(hx_powers(:,:,k))*sigeta*shockSize) + kron(squeeze(hx_powers(:,:,k))*sigeta*shockSize,squeeze(hx_powers(:,:,k+1))*xf) + ...
            kron(squeeze(hx_powers(:,:,k)),squeeze(hx_powers(:,:,k)))*consQ;
    waitbar(k/IRFlength,h);
end
close(h);
% Computing GIRFxs and GIRFy2nd
h = waitbar(0/IRFlength,'Computing GIRF for x2nd');
for k=1:IRFlength
    if k > 1
        GIRFxs(:,k) = hx*GIRFxs(:,k-1) + HHxxtil*GIRFxfxf(:,k-1);
    end
    GIRFy2nd(:,k) = gx*(GIRFxf(:,k)+GIRFxs(:,k)) + GGxxtil*GIRFxfxf(:,k);
    waitbar(k/IRFlength,h);
end
close(h);
% Saving output 
GIRFy(:,:,2) = GIRFy2nd;
GIRFx(:,:,2) = GIRFxf+GIRFxs;
GIRF.parts.xs=GIRFxs;
GIRF.parts.ys=GIRFy2nd;

if orderApp == 2
    GIRF.y=squeeze(GIRFy(:,:,2));
    GIRF.x=squeeze(GIRFx(:,:,2));
    return
end

%% Impulse response functions a third order
GIRFy3rd   = zeros(ny,IRFlength);
GIRFxrd    = zeros(nx,IRFlength);
GIRFxfxs   = zeros(nx^2,IRFlength);
GIRFxfxfxf = zeros(nx^3,IRFlength);
xf_xf      = kron(xf,xf);

etaEps_etaEps = zeros(nx^2,1);
idx = 0;
for gama2=1:nx
    for gama1=1:nx
        idx = idx + 1;
        etaEps_etaEps(idx,1) = eta(gama2,:)*eta(gama1,:)'*sig^2;
    end
end

% Computing GIRFxfxfxf
h = waitbar(0/IRFlength,'Computing GIRF for x1st^3');
for k=1:IRFlength
    term1 = zeros(nx^3,1);
    term2 = zeros(nx^3,1);
    term3 = zeros(nx^3,1);
    hx_km1= squeeze(hx_powers(:,:,k));
    hxlm1 = sig*hx_km1;
    hx_km1_eta_nu=hx_km1*sigeta*shockSize;
    for j=1:k
        omegaj = zeros(nx^3,1);
        hxlmj = sig*squeeze(hx_powers(:,:,k-j+1));
        idx = 0;
        for gama3=1:nx
            for gama2=1:nx
                for gama1=1:nx
                    idx = idx + 1;
                    %for phi1=1:ne
                    %    omegaj(idx,1) = omegaj(idx,1) + hxlmj(gama3,:)*sigeta(:,phi1)...
                    %        *hxlm1(gama2,:)*sigeta*shockSize...
                    %        *hxlmj(gama1,:)*sigeta(:,phi1);
                    %end
                    omegaj(idx,1) = sig^2*hxlmj(gama3,:)*eta_eta*hxlmj(gama1,:)'...
                            *hxlm1(gama2,:)*sigeta*shockSize;
                end
            end
        end
        term1 = term1 + omegaj(:,1);
        hx_kmj_hx_kmj_eta_eta=kron(hx^(k-j),hx^(k-j))*etaEps_etaEps;
        term2 = term2 + kron(hx_kmj_hx_kmj_eta_eta,hx_km1_eta_nu);
        term3 = term3 + kron(hx_km1_eta_nu,hx_kmj_hx_kmj_eta_eta);
    end
    hx_k_xf=squeeze(hx_powers(:,:,k+1))*xf;
    GIRFxfxfxf(:,k) = kron(hx_k_xf,kron((hx_k_xf+hx_km1_eta_nu),hx_km1_eta_nu)) ...
        + kron(hx_k_xf,kron(hx_km1_eta_nu,hx_k_xf)) ...
        + kron(hx_km1_eta_nu,kron((hx_k_xf+hx_km1_eta_nu),hx_k_xf)) ...
        + kron(hx_km1_eta_nu,kron(hx_k_xf,hx_km1_eta_nu)) ... 
        + term1 ...
        + term2 ...
        + term3 ...
        + kron(hx_km1_eta_nu,kron(hx_km1_eta_nu,hx_km1_eta_nu)); 
    waitbar(k/IRFlength,h);
end
close(h) 

% Computing GIRFxsxf
hx_hx     = kron(hx,hx);
hx_hsstil = kron(hx,0.5*hss*sig^2);
hx_HHxxtil= kron(hx,HHxxtil);
h = waitbar(0/IRFlength,'Computing GIRF for x1st*x2nd');
for k=1:IRFlength
    if k == 1
       GIRFxfxs(:,k) = kron(sigeta*shockSize,hx*xs+HHxxtil*xf_xf+0.5*hss*sig^2);
    elseif k > 1 
       GIRFxfxs(:,k) =  hx_hx*GIRFxfxs(:,k-1) + hx_HHxxtil*GIRFxfxfxf(:,k-1) + hx_hsstil*GIRFxf(:,k-1);
    end
    waitbar(k/IRFlength,h);    
end
close(h) 

% Computing GIRFxrd and GIRFy3rd
h = waitbar(0/IRFlength,'Computing GIRF for x3rd');
for k=1:IRFlength
    if k > 1
        GIRFxrd(:,k) = hx*GIRFxrd(:,k-1) + 2*HHxxtil*GIRFxfxs(:,k-1) + 3/6*hssx*sig^2*GIRFxf(:,k-1) + ...
            HHxxxtil*GIRFxfxfxf(:,k-1);
    end
    GIRFy3rd(:,k) = gx*(GIRFxf(:,k) + GIRFxs(:,k) + GIRFxrd(:,k)) ...
        + GGxxtil*(GIRFxfxf(:,k) + 2*GIRFxfxs(:,k)) ...
        + GGxxxtil*GIRFxfxfxf(:,k) + 3/6*gssx*sig^2*GIRFxf(:,k);
        waitbar(k/IRFlength,h);    
end
close(h) 

% Saving output
GIRFy(:,:,3) = GIRFy3rd;
GIRFx(:,:,3) = GIRFxf + GIRFxs + GIRFxrd;

GIRF.parts.xf=GIRFxf;
GIRF.parts.xs=GIRFxs;
GIRF.parts.xrd=GIRFxrd;
GIRF.y=squeeze(GIRFy(:,:,3));
GIRF.x=squeeze(GIRFx(:,:,3));
GIRF.parts.yf=GIRFy1st;
GIRF.parts.ys=GIRFy2nd;
GIRF.parts.yrd=GIRFy3rd;
