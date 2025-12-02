function K_cumvar = select_k_cumvar(explained_all, cumvar_thresh)
% Determine the number of PCs needed to reach cumulative variance threshold
cumulative_variance = cumsum(explained_all);
K_cumvar = find(cumulative_variance >= cumvar_thresh, 1, 'first');
end
