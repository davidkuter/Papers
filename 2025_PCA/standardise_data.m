function [X_scaled, mu, sigma] = standardise_data(X)
    % Standardise features to zero mean and unit variance
    mu = mean(X,1);
    sigma = std(X,[],1);
    X_scaled = (X - mu) ./ sigma;
end

