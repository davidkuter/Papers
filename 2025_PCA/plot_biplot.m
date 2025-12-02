function plot_biplot(scores, coeff, feature_names)
% Plots a biplot of first two PCs
figure;
biplot(coeff(:,1:2), 'Scores', scores(:,1:2), 'VarLabels', feature_names);
title('PCA Biplot (PC1 vs PC2)');
grid on;
end