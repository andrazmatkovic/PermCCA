function [U, V, R, R_bootCI] = cross_validate_CCA(X_test, Y_test, A_train, B_train, nboot, alpha)
%
% Apply CCA weights from training set to test set.
%
% INPUTS
% ======
%
% --X_test  test dataset - left side (observations x variables)
% --Y_test  test dataset - right side (observations x variables)
% --A_train CCA weights computed on trainig set - left side (variables x CC modes)
% --B_train CCA weights computed on trainig set - right side (variables x CC modes)
% --nboot   number of permutations for bootstrap [1000]
% --alpha   alpha for bootstrap CI for R [0.05]
%
% OUTPUTS
% =======
%
% U
%   cross validated CCA scores (left side); U=X*A; (observations x CC modes)
% V
%   cross validated CCA scores (right side); V=Y*B; (observations x CC modes)
% R
%   cross validated canonical correlations
% R_bootCI
%   bootstrap CI for R
%   
%

% ~~~~~~~~~~~~~~~
%
% 2021-05-03 Matkovic, Andraz
%            Initial version.

if nargin < 5 || isempty(nboot); nboot = 1000; end
if nargin < 6 || isempty(alpha); alpha = 0.05; end


U = X_test * A_train;
V = Y_test * B_train;

R = diag(corr(U, V));

if nargout > 3
    bootci_R(U,V,nboot,alpha)
end

function [ci, bootstats] = bootci_R(U,V,nboot,alpha)
%

    
[N,P] = size(U);

bootstats = NaN(nboot,P);
for i=1:nboot
    s = randsample(N,N,true); % sample with replacement
    bootstats(i,:) = diag(corr(U(s,:), V(s,:))); 
end

lower = quantile(bootstats, alpha/2)';
upper = quantile(bootstats, 1-alpha/2)';

ci = [lower upper];
