function [coeff_final, score_final, explained_final, cumulative_variance] = fit_pca(X_scaled, K_final)
% Fit PCA on full dataset and subset to K_final PCs
[coeff_all, score_all, ~, ~, explained_all] = pca(X_scaled,'Centered',false,'Algorithm','svd');  % all PCs
cumulative_variance_all = cumsum(explained_all);

% Subset for final PCs
coeff_final = coeff_all(:,1:K_final);
score_final = score_all(:,1:K_final);
explained_final = explained_all(1:K_final);
cumulative_variance = cumulative_variance_all(1:K_final);
end
