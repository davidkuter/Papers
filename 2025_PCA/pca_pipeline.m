function results = pca_pipeline(T, n_perm, alpha, cumvar_thresh)
% Modular PCA pipeline with loadings table (features x PCs)

if nargin < 2, n_perm = 500; end
if nargin < 3, alpha = 0.05; end
if nargin < 4, cumvar_thresh = 95; end

%% ----------------------
%% Extract numeric data
%% ----------------------
numeric_mask = varfun(@isnumeric, T, 'OutputFormat', 'uniform');
X = T{:, numeric_mask};
feature_names = T.Properties.VariableNames(numeric_mask);

%% ----------------------
%% Standardize
%% ----------------------
[X_scaled, mu, sigma] = standardise_data(X);

%% ----------------------
%% Permutation test
%% ----------------------
[K_perm, sig_mask, thresholds, real_eigvals] = permutation_test(X_scaled, n_perm, alpha);

%% ----------------------
%% Fit PCA on full dataset (all PCs)
%% ----------------------
[coeff_all, score_all, ~, ~, explained_all] = pca(X_scaled);

%% ----------------------
%% Determine K_cumvar
%% ----------------------
K_cumvar = select_k_cumvar(explained_all, cumvar_thresh);

%% ----------------------
%% Final conservative selection
%% ----------------------
K_final = min(K_perm, K_cumvar);

%% ----------------------
%% Subset PCA results for final PCs
%% ----------------------
[coeff_final, score_final, explained_final, cumulative_variance] = fit_pca(X_scaled, K_final);

%% ----------------------
%% Create loadings table: rows = features, columns = PCs
%% ----------------------
PC_names = arrayfun(@(x) sprintf('PC%d', x), 1:K_final, 'UniformOutput', false);
loadings_table = array2table(coeff_final, ...
    'VariableNames', PC_names, ...
    'RowNames', feature_names);

%% ----------------------
%% Collect results
%% ----------------------
results = struct();
results.feature_names = feature_names;
results.K_perm = K_perm;
results.significant_mask = sig_mask;
results.thresholds = thresholds;
results.real_eigvals = real_eigvals;
results.K_cumvar = K_cumvar;
results.cumvar_thresh = cumvar_thresh;
results.K_final = K_final;
results.coeff_final = coeff_final;
results.score_final = score_final;
results.explained_final = explained_final;
results.cumulative_variance = cumulative_variance;
results.mu = mu;
results.sigma = sigma;
results.loadings_table = loadings_table;  % <-- added table

end
