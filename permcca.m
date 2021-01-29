function varargout = permcca(Y,X,nP,Z,W,Sel,partial,statistic,permsetY,permsetX,ncompY,ncompX,K,varargin)
% Permutation inference for canonical correlation
% analysis (CCA).
%
% Usage:
% [pfwer,r,A,B,U,V] = permcca(Y,X,nP,Z,W,Sel,partial,permsetY,permsetX,ncompY,ncompX,K,varargin)
%
%
% Inputs:
% - Y        : Left set of variables, size N by P.
% - X        : Right set of variables, size N by Q.
% - nP       : An integer representing the number of permutations.
%              Default is 1000 permutations.
% - Z        : (Optional) Nuisance variables for both (partial CCA) or left 
%              side (part CCA) only.
% - W        : (Optional) Nuisance variables for the right side only 
%              (bipartial CCA).
% - Sel      : (Optional) Selection matrix or a selection vector, to use 
%              Theil's residuals instead of Huh-Jhun's projection. If
%              specified as a vector, it can be made of integer indices or
%              logicals. The R unselected rows of Z (S of W) must be full
%              rank. Use -1 to randomly select N-R (or N-S) rows.
% - partial  : (Optional) Boolean indicating whether this is partial (true) 
%              or part (false) CCA. Default is true, i.e., partial CCA.
% - statistic: (Optional) which statistic to use for statistical testing:
%              - 'wilks' for Wilns' lambda or
%              - 'roy' for Roy's largest root
%              Default: 'wilks'
% - permsetY : (Optional) Matrix with predefined permutations for left set
%              of variables (permutations should be stored in columns). 
%              Note: number of rows should equal number of selected rows.
% - permsetX : (Optional) Matrix with predefined permutations for right set
%              of variables (permutations should be stored in columns).
%              Note: number of rows should equal number of selected rows.
% - ncompY   : (Optional) Number of components after dimensionality 
%              reduction using SVD on left side.
% - ncompX   : (Optional) Number of components after dimensionality 
%              reduction using SVD on right side.
% - K        : (Optional) How many components to estimate
% - varargin : other arguments passed to penalized_cca.m; note that
%              cross-validation if performed only for initial CCA solution
%
% Outputs:
% - p   : p-values, FWER corrected via closure.
% - r   : Canonical correlations.
% - A   : Canonical coefficients, left side.
% - B   : Canonical coefficients, right side.
% - U   : Canonical variables, left side.
% - V   : Canonical variables, right side.
% - lambdaY : Regularization parameters for left side (Y)
% - lambdaX : Regularization parameters for right side (X)
% - cv  : cross validation results
% - Yr  : Y matrix (rank reduced is SVD was applied)
% - Xr  : X matrix (rank reduced is SVD was applied)
% - At  : Canonical coefficients transformed back to original space (only
%         available in case SVD was applied before CCA)
% - Bt  : Canonical coefficients transformed back to original space (only
%         available in case SVD was applied before CCA)
%
% ___________________________________________
% AM Winkler, O Renaud, SM Smith, TE Nichols
% NIH - Univ. of Geneva - Univ. of Oxford
% Mar/2020

%	~~~~~~~~~~~~~~~~~~
%
% 	Changelog
%   2021-01-13 Andraz Matkovic
%              Added K parameter. Use penalized_cca function instead of svd
%              to estimate component weights.

% Read input arguments
narginchk(2,18)
if nargin < 3 || isempty(nP);      nP = 1000;      end
if nargin < 4;                     Z = [];         end
if nargin < 5;                     W = [];         end
if nargin < 6;                     Sel = [];       end
if nargin < 7 || isempty(statistic); statistic = 'wilks'; end
if nargin < 8 || isempty(partial); partial = true; end
if nargin < 9 || isempty(permsetY)
    permsetY = false;
else
    if size(permsetY,2) < nP
        error("permsetY does not contain enough permutations.")
    end
end
if nargin < 10 || isempty(permsetX)
    permsetX = false;
else
    if size(permsetX,2) < nP
        error("permsetX does not contain enough permutations.")
    end
end
if nargin < 11 || isempty(ncompY); ncompY = false; end
if nargin < 12 || isempty(ncompX); ncompX = false; end
if nargin < 13 || isempty(K);      K      = min(rank(Y),rank(X)); end

Ny = size(Y,1);
Nx = size(X,1);
if Ny ~= Nx
    error('Y and X do not have same number of rows.')
end
N = Ny; clear Ny Nx
I = eye(N);

% Residualise Y wrt Z
if isempty(Z)
    Qz = I;
else
    Z  = center(Z);
    Qz = semiortho(Z,Sel);
end
Y = center(Y);
Y = Qz'*Y;
P = size(Y,1);
R = size(Z,2);

if size(permsetY,1) ~= (N - R)
    error('Number of rows in permsetY is not valid.')
end

% Residualise X wrt W
if isempty(W)
    if partial
        W  = Z;
        Qw = Qz;
    else
        Qw = I;
    end
else
    W  = center(W);
    Qw = semiortho(W,Sel);
end
X = center(X);
X = Qw'*X;
Q = size(X,1);
S = size(W,2);
if size(permsetX,1) ~= (N - S)
    error('Number of rows in permsetS is not valid.')
end

% Dimensionality reduction
if ncompY
    [Uy, Sy, Vy] = svds(Y, ncompY);
    Yr = Uy;
else
    Yr = Y;
end
if ncompX
    [Ux, Sx, Vx] = svds(X, ncompX);
    Xr = Ux;
else
    Xr = X;
end

% Initial CCA
switch statistic
    case 'wilks'
        Kinit = min(rank(Yr),rank(Xr));
    case 'roy'
        Kinit = K;
end
[A,B,r,lambdaY,lambdaX,cv] = cca(Qz*Yr,Qw*Xr,R,S,Kinit,varargin{:});
K = numel(r);
U = Yr*[A null(A')];
V = Xr*[B null(B')];

% First permutation is no permutation
fprintf('Permutation %d/%d ',1,nP);
    
idxY = (1:P);
idxX = (1:Q);

% Initialise counter
cnt = zeros(1,K);
lW  = zeros(1,K);
% For each canonical variable
for k = 1:K
    [lWtmp] = compute_statistic(Qz*U(idxY,k:end),Qw*V(idxX,k:end),R,S,statistic,lambdaY,lambdaX,varargin{3:end});
    lW(k) = lWtmp(1);
end
lW1 = lW;
cnt = cnt + (lW >= lW1);
fprintf('\n');

% For each permutation
parfor p = 2:(nP-1) 
    if permsetY
        idxY = permsetY(:,p);
    else
        idxY = randperm(P);
    end
    if permsetX
        idxX = permsetX(:,p);
    else
        idxX = randperm(Q);
    end
    fprintf('Permutation %d/%d ',p,nP);
    
    % For each canonical variable
    lW  = zeros(1,K);
    for k = 1:K
        [lWtmp] = compute_statistic(Qz*U(idxY,k:end),Qw*V(idxX,k:end),R,S,statistic,lambdaY,lambdaX,varargin{3:end});
        lW(k) = lWtmp(1);
    end
    cnt = cnt + (lW >= lW1);
    fprintf('\n');
end

U = Qz*Yr*A;
V = Qw*Xr*B;

punc  = cnt/nP;
varargout{1} = cummax(punc); % pfwer
varargout{2} = r;            % canonical correlations
varargout{3} = A;            % canonical weights (left)
varargout{4} = B;            % canonical weights (right)
varargout{5} = U;            % canonical variables (left)
varargout{6} = V;            % canonical variables (right)
varargout{7} = lambdaY;      % regularization parameter (left)
varargout{8} = lambdaX;      % regularization parameter (right)
varargout{9} = cv;           % cross validation results
varargout{10} = Yr;          % Y matrix (rank reduced is SVD was applied)
varargout{11} = Xr;          % X matrix (rank reduced is SVD was applied)

% transform A and B back to original dimensions
% (this is basically least squares solution: A = pinv(Yr) * U)
if ncompY
    varargout{12} = pinv(Uy * Sy * Vy') * Yr*A;
end
if ncompX
    varargout{13} = pinv(Ux * Sx * Vx') * Xr*B;
end

% =================================================================
function Q = semiortho(Z,Sel)
% Compute a semi-orthogonal matrix according to
% the Huh-Jhun or Theil methods. Note that, due to a
% simplification of HJ, input here is Z, not Rz.
if isempty(Sel)
    % If Sel is empty, do Huh-Jhun
    % HJ here is simplified as in Winkler et al, 2020 (see the Appendix text of the paper)
    [Q,D,~] = svd(null(Z'),'econ');
    Q = Q*D;
else
    % Theil
    [N,R] = size(Z);
    if isvector(Sel)
        % If Sel is a vector of logical or integer indices
        if islogical(Sel)
            Sel = find(Sel);
        end
        if Sel(1) > 0
            % If Sel is a column of indices
            unSel = setdiff(1:N,Sel);
            if rank(Z(unSel,:)) < R
                error('Selected rows of nuisance not full rank')
            end
        else
            % If Sel is -1 or anything else but empty [].
            % Try first with a faster approach
            Sel = true(N,1);
            Zs  = bsxfun(@rdivide,Z,mean(Z,2));
            [~,~,iU] = unique(Zs,'rows');
            nU = max(iU);
            for r = randperm(nU,R)
                idx = find(iU == r);
                idx = idx(randperm(numel(idx)));
                Sel(idx(1)) = false;
            end
            % but it it fails, go with another one
            if rank(Z(~Sel,:)) < R
                foundSel = false;
                while ~ foundSel
                    Sel   = sort(randperm(N,N-R));
                    unSel = setdiff(1:N,Sel);
                    if rank(Z(unSel,:)) == R
                        foundSel = true;
                    end
                end
            end
        end
        S = eye(N);
        S = S(:,Sel);
    else
        % Sel is a matrix proper
        S = Sel;
    end
    Rz = eye(N) - Z*pinv(Z);
    Q = Rz*S*sqrtm(inv(S'*Rz*S));
end

% =================================================================
function [A,B,cc,lambdaY,lambdaX,cv] = cca(Y,X,R,S,K,varargin)
% Compute CCA.
N = size(Y,1);
%[Qy,Ry,iY] = qr(Y,0);
%[Qx,Rx,iX] = qr(X,0);

%K  = min(rank(Y),rank(X));
%[L,D,M] = svds(Qy'*Qx,K);
%cc = min(max(diag(D(:,1:K))',0),1);

[~,~,A,B,cc,lambdaY,lambdaX,cv] = penalized_cca_witten(Y,X,K,varargin{:});

%A  = Ry\L(:,1:K)*sqrt(N-R);
%B  = Rx\M(:,1:K)*sqrt(N-S);
%A(iY,:) = A;
%B(iX,:) = B;

% =================================================================
function X = center(X)
% Mean center a matrix and remove constant columns.
icte = sum(diff(X,1,1).^2,1) == 0;
X = bsxfun(@minus,X,mean(X,1));
X(:,icte) = [];


function [stat] = compute_statistic(Y,X,R,S,statistic,optionsY,optionsX,varargin)
    
switch statistic
    case 'wilks'
        [~,~,rperm] = cca(Y,X,R,S,[],optionsY,optionsX,varargin{3:end});
        stat = -fliplr(cumsum(fliplr(log(1-rperm.^2))));
    case 'roy'
        [~,~,rperm] = cca(Y,X,R,S,1,optionsY,optionsX,varargin{3:end});
        stat = rperm(1)^2;
end
