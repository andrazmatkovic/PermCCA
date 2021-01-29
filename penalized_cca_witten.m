function [U,V,A,B,R,penaltyX,penaltyY,cv] = penalized_cca_witten(X,Y,K,penaltyX,penaltyY,kfold,max_iter,niter)
%
% Fit regularized CCA using penalized matrix decomposition (Witten,
% 2020). Aim of CCA is to find maximally correlated pairs canonical vectors
% u_i, v_i for u = 1 ... k such that U = X*A and V = Y*B, subject to 
% U*U = I and V*V = I, where u_i and v_i vectors represent columns of U and 
% V, respectively.
%
% k-fold cross validation can be used to optimize parameters. Grid or random search is 
% used to determine the combination of penalties which yield highest first canonical
% correlation.
%
% Most of the code is directly translated from R code in PMA package 
% (https://github.com/bnaras/PMA).
%
%
% INPUTS
% ======
% 
% --X           N x P data matrix
% --Y           N x Q data matrix
% --K           how many components to fit; by default it fits all possible 
%               canonical modes, but this can be lowered to save time, 
%               because usually only first few modes capture relevant
%               covariaton between X and Y
%               [min(P,Q)]
% --penaltyX    penalty for X; if a vector of multiple values is supplied, cross
%               validation is performed; must be within range 0-1 [0]   
% --penaltyY    penalty for Y; if a vector of multiple values is supplied, cross
%               validation is performed; must be within range 0-1 [0] 
% --kfold       number of folds for cross validation [5]
% --max_iter    if max_iter ~= 0, random search is performed instead of
%               grid search, such that only max_iter parameter combinations
%               are tested [0]
% --niter       number of iterations [15]
%
% OUTPUTS
% =======
%
%   A
%       canonical weights for X (P x K, where K = min(P,Q) or defined by user)
%   B
%       canonical weights for Y (Q x K)
%   U
%       canonical scores / variables for X (N x K)
%   V
%       canonical scores / variables for Y (N x K)
%   R
%       vector of canonical correlations
%   penaltyX
%       penalty for X (best penalty in case cross-validation is performed)
%   penaltyY
%       penalty for Y (best penalty in case cross-validation is performed)
%
% SOURCES
% =======
%
% Witten, D. M., Tibshirani, R., & Hastie, T. (2009). A penalized matrix decomposition, 
% with applications to sparse principal components and canonical correlation analysis. 
% Biostatistics, 10(3), 515-534. https://doi.org/10.1093/biostatistics/kxp008
%
% Witten, D., Tibshirani, R., Gross, S., Narasimhan, B., & Witten, M. D. (2020). 
% Package ‘pma’. Genetics and Molecular Biology, 8(1), 28. https://github.com/bnaras/PMA


%	~~~~~~~~~~~~~~~~~~
%
% Changelog
% 2021-01-29 Andraz Matkovic
%            Initial version.

narginchk(2,8);

rng(10);

X = zscore(X);
Y = zscore(Y);

N = size(X,1);
P = size(X,2);
Q = size(Y,2);

if nargin < 3 || isempty(K) || K > min(P,Q); K        = min(P,Q); end
if nargin < 4 || isempty(penaltyX);          penaltyX = 0;        end
if nargin < 5 || isempty(penaltyY);          penaltyY = 0;        end
if nargin < 6 || isempty(kfold);             kfold    = 5;        end
if nargin < 7 || isempty(max_iter);          max_iter = 0;        end
if nargin < 8 || isempty(niter);             niter    = 15;       end

A = NaN(P,K);
B = NaN(Q,K);
V = NaN(N,K); 
U = V;

cv = struct();

XY = X'*Y / N;
[~, ~, B_init] = svds(XY,K);

if length(penaltyX) > 1 || length(penaltyY) > 1 % cross validate
  cv.grid = make_grid(penaltyX, penaltyY);
  grid_idx = 1:numel(cv.grid);
  if max_iter > 0 && max_iter < numel(cv.grid) % random search
      grid_idx = randperm(numel(cv.grid));
      grid_idx = grid_idx(1:max_iter);
      grid_idx = sort(grid_idx);  
  end
  
  cv.critgrid = NaN(size(cv.grid));
  
  for i=grid_idx % do not change to parfor before comparing the results between non-parallelized and parallelized forms
      tmp_penalties = cv.grid(i);
      penaltyX_tmp = tmp_penalties{1}(1);
      penaltyY_tmp = tmp_penalties{1}(2);
      
      folds   = reshape(randperm(N-mod(N,kfold)),[],kfold)';
      cv_crit = NaN(kfold,1);
      for j=1:kfold
          
          testidx = folds(j,:)';
          trainidx = folds(1:kfold ~= j,:); trainidx = trainidx(:);
          [A_tmp, B_tmp, ~] = main_algorithm(X(trainidx,:),Y(trainidx,:),B_init(:,1:K),penaltyX_tmp,penaltyY_tmp,1, niter);

          % using maximal first canonical correlation as a criterium for CV
          cv_crit(j) = corr(X(testidx,:)*A_tmp,Y(testidx,:)*B_tmp);
      end
      cv.critgrid(i) = mean(cv_crit);
  end
  
   % save best parameters
   [~, max_i] = max(cv.critgrid(:));
   best_penalties = cv.grid(max_i);
   penaltyX = best_penalties{1}(1);
   penaltyY = best_penalties{1}(2);
   
end


[A, B, ~] = main_algorithm(X,Y,B_init(:,1:K),penaltyX,penaltyY,K, niter);

U = X*A;
V = Y*B;
R = abs(diag(corr(U,V)))';


end


function [A_big, B_big, D] = main_algorithm(X,Y,B,penaltyX,penaltyY,K, niter)

B_init = B(sum(Y.^2,1) ~= 0,:);

X_res = X;
Y_res = Y;

%X_res = X(sum(X.^2,2) ~= 0,:);
%Y_res = Y(sum(Y.^2,2) ~= 0,:);

for k=1:K
    [Ai, Bi, Di] = innerloop(X_res, Y_res, B_init(:,k), penaltyX, penaltyY, niter);

    X_res = [X_res; sqrt(Di) * Ai'];
    Y_res = [Y_res; -sqrt(Di) * Bi'];

    if k == 1
        A = Ai;
        B = Bi;
        D = Di;
    else
        D = [D Di];
        A = [A Ai];
        B = [B Bi];
    end

end

A_big = A;
B_big = B;

%A_big = zeros(size(X,2), K);
%A_big(sum(X.^2,1) ~= 0,:) = A;

%B_big = zeros(size(Y,2), K);
%B_big(sum(Y.^2,1) ~= 0,:) = B;

end

function [A, B, D] = innerloop(X,Y,B,penaltyX,penaltyY,niter)

B_old = rand(size(B,1),1);
ncolX = size(X,2);
ncolY = size(Y,2);
A     = rand(ncolX,1);

for i = 1:niter
   if sum(isnan(A)) > 0 || sum(isnan(B)) > 0
      B = zeros(0, size(B,1));
      B_old = B;
   end
   if sum(abs(B_old - B)) > (1e-6)
       
      argA = X'*Y*B;
      lamA = binary_search(argA, penaltyX * sqrt(ncolX));
      sA = soft_threshold(argA, lamA);
      A = sA / l2norm(sA);
      
      % update V
      B_old = B;
      argB = A'*X'*Y;
      lamB = binary_search(argB, penaltyY * sqrt(ncolY));
      sB = soft_threshold(argB, lamB)';
      B = sB / l2norm(sB);
   end
end

D = sum((X*A)'*(Y*B));
if sum(isnan(A)) > 0 || sum(isnan(B)) > 0
    A = zeros(ncolX,1);
    B = zeros(ncolY,1);
    D = 0;
end



end



function [error] = MSE(XY_hat,XY)

error = norm(XY_hat - XY);
end


function [grid] = make_grid(a,b)
% Make a grid of numbers from vectors a and b.
%
% --a   vector
% --b   vector

grid = cell(numel(a),numel(b));
for i=1:numel(a)
    for j=1:numel(a)
        grid(i,j) = {[a(i) b(j)]};
    end
end

end

function [a] = soft_threshold(a, threshold)
% soft-thresholding function

d = abs(a) - threshold;
d(d < 0) = 0;
a = sign(a) .* d;

end


function [out] = binary_search(argu,sumabs)
if l2norm(argu) == 0 || sum(abs(argu/l2norm(argu))) <= sumabs
    out = 0;
    return
end
lam1 = 0;
lam2 = max(abs(argu)) - (1e-5);
iter = 1;
while iter < 150
    su = soft_threshold(argu,(lam1+lam2)/2);
    if sum(abs(su/l2norm(su))) < sumabs
        lam2 = (lam1+lam2) / 2;
    else
        lam1 = (lam1+lam2) / 2;
    end
    if (lam2-lam1) < 1e-6 
        out = (lam1+lam2)/2;
        return
    end
    iter = iter+1;
end
warning("Didn't quite converge");
out = (lam1+lam2)/2;

end

function [a] = l2norm(vec)
  a  = norm(vec);
  if a == 0
      a = .05;
  end
end