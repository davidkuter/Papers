function [errors_mean, errors_std, K_cv] = cross_validated_pca(X_scaled, n_splits)
[n_samples, n_features] = size(X_scaled);
errors_mean = zeros(n_features,1);
errors_std  = zeros(n_features,1);

cv = cvpartition(n_samples,'KFold',n_splits);

for k = 1:n_features
    fold_errors = zeros(n_splits,1);
    for f = 1:n_splits
        train_idx = training(cv,f);
        test_idx  = test(cv,f);

        [coeff_train,~,~,~,~,mu] = pca(X_scaled(train_idx,:),'NumComponents',k);

        X_test_centered = X_scaled(test_idx,:) - mu;
        X_proj = X_test_centered * coeff_train;
        X_recon = X_proj * coeff_train' + mu;

        % normalize by variance per feature (from training fold)
        var_features = var(X_scaled(train_idx,:), 0, 1);
        fold_errors(f) = mean(mean((X_scaled(test_idx,:) - X_recon).^2 ./ var_features));
    end
    errors_mean(k) = mean(fold_errors);
    errors_std(k) = std(fold_errors);
end

[~, K_cv] = min(errors_mean);
end