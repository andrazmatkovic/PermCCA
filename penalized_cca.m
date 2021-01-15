function [U,V,A,B,R,optionsX,optionsY] = penalized_cca(X,Y,K,optionsX,optionsY,kfold,max_iter,tol)
%
% Fit regularized CCA using alternating least squares algorithm (Polajnar,
% 2020). Aim of CCA is to find maximally correlated pairs canonical vectors
% u_i, v_i for u = 1 ... k such that U = X*A and V = Y*B, subject to 
% U*U = I and V*V = I, where u_i and v_i vectors represent columns of U and 
% V, respectively.
%
% glmnet function is used for regularization, therefore elastic net, LASSO
% and ridge regularization are available through the use of mixing
% parameter (e.g. optionsX.alpha = 1 means that LASSO will be used for X
% variables). Regularization parameters can be fit for every component 
% separately and also separately for each set of variables.
%
% k-fold cross validation can be used to optimize parameters. Note that
% cross validation is performed for each component separately. Computation
% time can be reduced using parameter K, which determines how many
% components will be estimated. Usually only first few components are
% relevant, therefore computational time can be reduced significantly.
%
% Lambda can be optimized using grid or random search (default: grid), see
% max_iter parameter.
%
% Dependns on glmnet for Matlab.
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
% --optionsX    struct; arguments lambda and alpha passed to glmnet or 
%               Matlab's lasso function for X; e.g.
%
%               Default:
%
%                   optionsX.lambda = 0; 
%                   optionsX.alpha = 1;
%
%               - if length(optionsX) == 1, same value is used for all
%                 components, otherwise different values can be set for each
%                 component;
%               - if length(optionsX) > 1 && length(optionsX) < K, last value
%                 is recycled for subsequent components;
%               - if multiple values of options.lambda are supplied cross
%                 validation is performed and best fitting lambda is selected
%
% --optionsY    struct; arguments passed to glmnet for Y; same as for
%               optionsX
% --kfold       number of folds for cross validation [5]
% --max_iter    if max_iter ~= 0, random search is performed instead of
%               grid search, such that only max_iter parameter combinations
%               are tested [0]
% --tol         converge tolerance [0.00001]; this relates to the
%               difference in canonical correlation at which estimation of
%               weights can be stopped
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
%   optionsX
%       structure with regularization parameters for X
%   optionsY
%       structure with regularization parameters for Y
%
% EXAMPLE
% =======
%
% In this example LASSO regularization will be used for all components and
% only first 3 components will be estimated. Grid search and 5-fold cross 
% validation will be used to optimize hyperparameter lambda::
%
%   options = struct();
%   options(1).lambda = [0.001 0.01 0.02 0.05 0.1 0.2 0.3 0.5];
%   options(1).alpha = 1;
%
%   [U,V,A,B,R,optX,optY] = penalized_cca(X,Y,options,options,3,5);
%
% SOURCES
% =======
%
% Polajnar, E. (2019). Using elastic net restricted kernel canonical
%   correlation analysis for cross-language information retrieval.
%   Communications in Statistics-Simulation and Computation, 1-18.
%   doi: 10.1080/03610918.2019.1704420
%
% DEPENDENCY
% ==========
%
% - glmnet is a dependency only if Octave is being used
%
%   Glmnet for Matlab (2013) Qian, J., Hastie, T., Friedman, J., Tibshirani,
%       R. and Simon, N. http://www.stanford.edu/~hastie/glmnet_matlab/
% 


%	~~~~~~~~~~~~~~~~~~
%
% 	Changelog
%   2021-01-12 Andraz Matkovic
%              Initial version.

narginchk(2,8);

rng(10);

X = zscore(X);
Y = zscore(Y);

N = size(X,1);
P = size(X,2);
Q = size(Y,2);

if nargin < 3 || isempty(K) || K > min(P,Q); K        = min(P,Q); end
if nargin < 4 || isempty(optionsX);          optionsX.lambda = 0; optionsX.alpha = 1; end
if nargin < 5 || isempty(optionsY);          optionsY.lambda = 0; optionsY.alpha = 1; end
if ~isfield(optionsX,'alpha');  optionsX.alpha  = 1; end
if ~isfield(optionsY,'alpha');  optionsY.alpha  = 1; end
if ~isfield(optionsX,'lambda'); optionsX.lambda = 0; end
if ~isfield(optionsY,'lambda'); optionsY.lambda = 0; end
if nargin < 6 || isempty(kfold);             kfold    = 5;        end
if nargin < 7 || isempty(max_iter);          max_iter = 0;        end
if nargin < 8 || isempty(tol);               tol      = 0.00001;  end

A = NaN(P,K);
B = NaN(Q,K);
V = NaN(N,K); 
U = V;

% recycle last option if not enough options are provided
if length(optionsX) < K
    for i=length(optionsX):K
        optionsX(i) = optionsX(end);
    end
end
if length(optionsY) < K
    for i=length(optionsY):K
        optionsY(i) = optionsY(end);
    end
end

for k=1:K
    
    if length(optionsX(k).lambda) == 1 && length(optionsY(k).lambda) == 1
        [A, B, U, V] = innerloop(X,Y,A,B,U,V,optionsX(k),optionsY(k),tol,k);
    else
        % cross validate
        grid = combvec(optionsX(k).lambda, optionsY(k).lambda);
        if max_iter > 0 % random search
           grid = grid(:,randperm(max_iter));
           grid = grid(:,1:max_iter);
        end
        
        errorgrid = NaN(size(grid,2),1);
        
        for i=1:size(grid,2) % do not change to parfor before comparing the results between non-parallelized and parallelized forms
            optionsX_tmp = optionsX(k);
            optionsY_tmp = optionsY(k);
            optionsX_tmp.lambda = grid(1,i);
            optionsY_tmp.lambda = grid(2,i);
            
            folds = reshape(randperm(N-mod(N,kfold)),[],kfold)';
            error = NaN(kfold,1);
            for j=1:kfold

                
                trainidx = folds(j,:);
                testidx  = folds(1:kfold ~= j,:); testidx = testidx(:);

                [A_tmp, B_tmp, ~, ~] = innerloop(X(trainidx,:),Y(trainidx,:),A,B,U(trainidx,:),V(trainidx,:),optionsX_tmp,optionsY_tmp,tol,k);
                error(j) = MSPE(X(testidx,:)*A_tmp(:,k),Y(testidx,:)*B_tmp(:,k));
            end
            errorgrid(i) = mean(error);
        end
        
         % save best parameters
         [~, min_i] = min(errorgrid);
         optionsX(k).lambda = grid(1,min_i);
         optionsY(k).lambda = grid(2,min_i);
         
         % fit once again using best parameters
         [A, B, U, V] = innerloop(X,Y,A,B,U,V,optionsX(k),optionsY(k),tol,k);
    end
    
end

R = abs(diag(corr(U,V)))';

end

function [A, B, U, V] = innerloop(X,Y,A,B,U,V,optionsX,optionsY,tol,k)
    
% initialize
critR  = Inf;
Rk_old = Inf;
Q      = size(Y,2);
Bk     = (1/Q) * ones(Q,1);

while critR > tol
    v = Y * Bk;

    % orthogonalize
    if k > 1
       [v, ~] = qr([V(:,1:k-1) v]);
       v = v(:,k);
    end

    % least squares fit
    Ak_tmp = elasticnet(X,v,optionsX);

    u  = X*Ak_tmp;
    Ak = Ak_tmp / sqrt(var(u)); % normalize variance
    Ak(isnan(Ak)) = 0;
    
    % orthogonalize
    u  = X*Ak_tmp;
    if k > 1
       [u,~] = qr([U(:,1:k-1) u]);
       u = u(:,k);
    end

    % least squares fit
    Bk_tmp = elasticnet(Y,u,optionsY);
    
    v  = Y*Bk_tmp;
    Bk = Bk_tmp / sqrt(var(v)); % normalize variance
    Bk(isnan(Bk)) = 0;
    
    uk = X*Ak;
    vk = Y*Bk;

    Rk = corr(uk,vk);

    critR  = abs(Rk - Rk_old);
    Rk_old = Rk;
end

A(:,k) = Ak;
B(:,k) = Bk;
U(:,k) = uk;
V(:,k) = vk;

end

function [error] = MSPE(uk,vk)
% Compute mean squared prediction error.
%
% SOURCE
% ======
%
% Waaijenborg, S., & Zwinderman, A. H. (2007, December). Penalized
%   canonical correlation analysis to quantify the association between gene
%   expression and DNA markers. In BMC proceedings (Vol. 1, No. S1, p. S122).
%   BioMed Central. doi: 10.1186/1753-6561-1-S1-S122

Rk = corr(uk,vk);
N  = size(uk,1);

error = norm((uk - Rk.*vk),2) ./ N;

end

function [B] = elasticnet(X,y,options)

if exist('OCTAVE_VERSION', 'builtin') ~= 0
    B_fit = glmnet(X, v, 'gaussian', options);
    B     = B_fit.beta;
else
    B = lasso(X,y,'Lambda',options.lambda,'Alpha',options.alpha);
end

end