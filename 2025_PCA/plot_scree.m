function plot_scree(explained)
% Classic scree plot with points connected by line
figure;
plot(1:length(explained), explained, '-o', 'LineWidth', 2, 'MarkerSize', 6, ...
    'MarkerFaceColor', [0.2 0.6 0.8]);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Scree Plot');
grid on;
end

