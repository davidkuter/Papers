function plot_cumvar(cumvar, threshold)
% Plots cumulative variance and threshold line
figure;
plot(cumvar, '-o', 'LineWidth', 2);
xlabel('Number of PCs');
ylabel('Cumulative Variance (%)');
title('Cumulative Variance Plot');
grid on;
yline(threshold, 'r--', sprintf('Threshold = %d%%', threshold));
end