function [scree, k, pvar] = parallel_analysis(X, nperm, permset, alpha, P2plot, verbose)
% function [scree, k, pvar] = parallel_analysis(X, nperm, permset)
%
% Perform Horn's parallel analysis for PCA to estimate the number of
% "statistically significant" components.
%
% INPUTS
% ======
%
% --X          N x P matrix (N observations, P variables)
% --nperm      number of permutations
% --permset    predefined permutations sets (N x P x nperm array) (optional)
% --alpha      alpha value to use for determination of "significant" components 
%              (one-sided testing is applied) [.05]
% --P2plot     how many components to plot [50]
% --verbose    whether to print information [false]
%
% OUTPUTS
% =======
%
% scree
%   scree plot
% k
%   number of estimated components
% pvar
%   what percent of variance do first k components explain
%
%

% 2020-12-18 Matkovic, Andraz
%            Initial version
%

if nargin < 3 || isempty(permset);  permset  = [];    end
if nargin < 4 || isempty(alpha);    alpha    = 0.05;  end
if nargin < 5 || isempty(P2plot);   P2plot   = 50;    end
if nargin < 6 || isempty(verbose);  verbose = false;  end

N = size(X,1);
P = size(X,2);
%if P > N
%    P = N;
%end

X = bsxfun(@minus, X, mean(X)); % center

%[~, ~, S] = pca(X);
[~, S, ~] = svd(X, 'econ'); % use svd instead of pca to enable computing pca on m < n matrices
S = diag(S) .^2 ./ (N - 1); % transform to eigenvalues

permS = NaN(nperm,min(P,N));

parfor p=1:nperm
    if verbose
        fprintf('... permutation no. %d\n', p)
    end
    permX = X;
    for c=1:P
        if permset
            idx = permset(:,c,p);
        else
            idx = randperm(N);
        end
        
        permX(:,c) = permX(idx,c);

    end

    [~, tmpS, ~] = svd(permX, 'econ');
    permS(p,:) = diag(tmpS) .^2 ./ (N - 1);
    %[~, ~, permS(p,:)] = pca(permX);

end

q = [quantile(permS, alpha); quantile(permS, 1 - alpha)];
m = mean(permS);
k = find((S < q(2,:)'), 1, 'first') - 1;
scree = figure;

if P < P2plot
    P2plot = P;
end

plot(1:min(P,N), S/sum(S), 'b', 1:min(P,N), m/sum(S), 'k', 1:min(P,N), q/sum(S), 'k:', 'LineWidth', 1)
xlim([1 P2plot]);
legend('Raw data eigenvalues', 'Mean eigenvalues of permuted datasets', sprintf('%f %% confidence interval', 1 - alpha*2))
grid on
xlabel('Component')
ylabel('% of explained variance')

pvar = sum(S(1:k) / sum(S)) * 100;
str = sprintf('Number of significant components: %d.\nTogether they explain %.2f %% of variance', k, pvar);
if verbose
    fprintf('... %d significant components explain %.2f %% of variance', k, pvar)
end
text(P2plot/2,S(1)/sum(S)/2,str)
