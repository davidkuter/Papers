function [K_perm, significant_mask, thresholds, real_eigvals] = permutation_test(X_scaled, n_perm, alpha)
% Permutation test for significant PCs using parallel analysis
[n_samples, n_features] = size(X_scaled);

% Eigenvalues of real data
[~,~,latent] = pca(X_scaled);
real_eigvals = latent;

% Permutation null distribution
perm_eigvals = zeros(n_perm, n_features);
rng(42); % reproducibility
for i = 1:n_perm
    X_perm = X_scaled;
    for j = 1:n_features
        X_perm(:,j) = X_perm(randperm(n_samples), j); % shuffle column independently
    end
    [~,~,latent_perm] = pca(X_perm, 'Centered',false,'Algorithm','svd');
    perm_eigvals(i,:) = latent_perm';
end

% Threshold for significance
thresholds = prctile(perm_eigvals, 100*(1-alpha), 1);
significant_mask = real_eigvals' > thresholds;
K_perm = sum(significant_mask);
end

