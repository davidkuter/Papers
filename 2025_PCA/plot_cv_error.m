function plot_cv_error(errors_mean, errors_std, K_cv)
% Plot mean ± std reconstruction error vs number of PCs
%
% Inputs:
%   errors_mean : mean CV reconstruction error for each k
%   errors_std  : std deviation across folds
%   K_cv        : PC number with minimum error

    k = 1:length(errors_mean);
    figure;
    errorbar(k, errors_mean, errors_std, 'bo-', 'LineWidth', 2, 'MarkerSize',6);
    hold on;
    yline(errors_mean(K_cv),'r--','LineWidth',2);
    xlabel('Number of PCs');
    ylabel('Mean Squared Reconstruction Error');
    title('Cross-Validated PCA Reconstruction Error');
    legend('CV Error ± SD', sprintf('Minimum Error at PC %d',K_cv),'Location','northeast');
    grid on;
    hold off;
end


