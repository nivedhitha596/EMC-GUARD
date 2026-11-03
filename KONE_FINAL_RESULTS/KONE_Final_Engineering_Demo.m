%% =========================================================================
%  KONE_Final_Engineering_Demo.m
%  EMC-PREDICT + EMC-GUARD  -  Final Engineering Visualisation
%
%  Physics-Informed Configuration-Aware Digital Twin
%  for Predictive EMC Pre-Compliance of Elevator Systems
%
%  Workflow: PREDICT --> OPTIMISE --> EXPERIMENT --> CORRELATE
%
%  This script generates:
%   A. Global sensitivity analysis
%   B. Multi-objective Pareto optimisation
%   C. Model agreement analysis
%   D. Residual analysis
%   E. Engineering response surfaces
%   F. Animated 3D cable geometry visualisation
%
%  IMPORTANT NOTICES
%  -----------------------------------------------------------------
%  1. This script reads only from the frozen Stage 12A/12B data files.
%     Model coefficients are NOT reloaded or modified.
%  2. All results originate from the deterministic MATLAB analytical
%     dataset (216 configurations). No physical measurements are shown.
%  3. The 3D geometry is an illustrative engineering representation of
%     the parameterised model, not an electromagnetic field simulation.
%  4. Training agreement (R^2 ~= 1) reflects exact dataset reproduction
%     by the least-squares model, NOT independent validation accuracy.
%  5. No EMC compliance, safety certification or regulatory approval
%     is claimed or implied by any result shown here.
%  -----------------------------------------------------------------
%
%  Usage: Run this script from MATLAB with path set to project root.
%         All figures are saved to KONE_FINAL_RESULTS\.
% =========================================================================

clear; clc; close all;

%% -- PATH CONFIGURATION ---------------------------------------------------

projectRoot  = 'D:\Downloads\kone\01_MATLAB_Models';
resultsRoot  = 'D:\Downloads\kone\KONE_FINAL_RESULTS';

path12B = fullfile(projectRoot, '06_Stage11A_Results', '08_Stage12B_Results');
path12A = fullfile(projectRoot, '06_Stage11A_Results', '07_Stage12A_ModelFreeze');
path11A = fullfile(projectRoot, '06_Stage11A_Results');

if ~exist(resultsRoot, 'dir')
    mkdir(resultsRoot);
end

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('  EMC-PREDICT + EMC-GUARD  -  Final Engineering Visualisation\n');
fprintf('%s\n\n', repmat('=', 1, 70));

%% -- LOAD ALL DATA --------------------------------------------------------

fprintf('[1/6] Loading data files...\n');

% Stage 12B7 - All 216 configurations
allCfgFile = fullfile(path12B, 'Stage12B7_AllConfigurations.csv');
assert(isfile(allCfgFile), ['Missing: ' allCfgFile]);
allCfg = readtable(allCfgFile);

% Stage 12B7 - Corrected Pareto front (6 configurations)
paretoFile = fullfile(path12B, 'Stage12B7_Corrected_Pareto.csv');
assert(isfile(paretoFile), ['Missing: ' paretoFile]);
paretoCorr = readtable(paretoFile);

% Stage 12B7 - Pareto configurations (9 configurations)
paretoAllFile = fullfile(path12B, 'Stage12B7_ParetoConfigurations.csv');
assert(isfile(paretoAllFile), ['Missing: ' paretoAllFile]);
paretoAll = readtable(paretoAllFile);

% Stage 12B6 - Global sensitivity summary
globalSensFile = fullfile(path12B, 'Stage12B6_GlobalSensitivitySummary.csv');
assert(isfile(globalSensFile), ['Missing: ' globalSensFile]);
globalSens = readtable(globalSensFile);

% Stage 12B6 - Factor level effects
factorLvlFile = fullfile(path12B, 'Stage12B6_FactorLevelEffects.csv');
assert(isfile(factorLvlFile), ['Missing: ' factorLvlFile]);
factorLvl = readtable(factorLvlFile);

% Stage 12B5 - Individual sensitivity sweeps
sepSens    = readtable(fullfile(path12B, 'Stage12B5_Separation_Sensitivity.csv'));
pitchSens  = readtable(fullfile(path12B, 'Stage12B5_Pitch_Sensitivity.csv'));
overSens   = readtable(fullfile(path12B, 'Stage12B5_Overlap_Sensitivity.csv'));
gndSens    = readtable(fullfile(path12B, 'Stage12B5_GroundClearance_Sensitivity.csv'));

% Stage 12B8 - Balanced design ranking
balFile = fullfile(path12B, 'Stage12B8_Balanced_Design_Ranking.csv');
balRank = readtable(balFile);

% Stage 12A - Model predictions vs actuals
predFile = fullfile(path12A, 'Stage12A_Selected_Model_Predictions.csv');
assert(isfile(predFile), ['Missing: ' predFile]);
modelPred = readtable(predFile);

% Stage 12A - Model summary
summFile = fullfile(path12A, 'Stage12A_Selected_Model_Summary.csv');
assert(isfile(summFile), ['Missing: ' summFile]);
modelSumm = readtable(summFile);

% Stage 11A - Spearman correlations and sensitivity indices
spearman  = readtable(fullfile(path11A, 'Stage11A_Spearman_Results.csv'));
sensIdx   = readtable(fullfile(path11A, 'Stage11A_Sensitivity_Indices.csv'));

fprintf('   All data files loaded successfully.\n');
fprintf('   All configurations : %d\n', height(allCfg));
fprintf('   Pareto (corrected) : %d\n', height(paretoCorr));
fprintf('   Pareto (full)      : %d\n', height(paretoAll));

%% -- GLOBAL STYLE SETTINGS ------------------------------------------------

% Colour palette
COL_BLUE    = [0.145  0.400  0.855];   % primary
COL_ORANGE  = [1.000  0.502  0.000];   % Pareto / accent
COL_GREEN   = [0.133  0.631  0.133];
COL_RED     = [0.839  0.153  0.157];
COL_GREY    = [0.502  0.502  0.502];
COL_GOLD    = [0.855  0.647  0.126];
COL_PURPLE  = [0.580  0.404  0.741];

BG_DARK = [0.06 0.09 0.14];

FIG_SIZE = [120 80 900 560];   % position for exported figures

% (styled_figure and styled_axes moved to local functions at end)

FSIZE = 11;

%% =======================================================================
%  A. GLOBAL SENSITIVITY ANALYSIS
% =======================================================================

fprintf('\n[A] Generating global sensitivity analysis...\n');

% -- A1: Average conditional range bar chart (Stage 12B6) -----------------

capSens  = globalSens(strcmp(globalSens.Response, 'Capacitance'), :);
mutSens  = globalSens((strcmp(globalSens.Response, 'MutualInductance') | strcmp(globalSens.Response, 'AbsMutualInductance')), :);

figA1 = styled_figure(FIG_SIZE);
t_A1 = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t_A1, 'A1. Global Sensitivity Analysis - Average Conditional Range', ...
    'FontSize', FSIZE+3, 'FontWeight', 'bold');

ax1 = nexttile;
[sortedCap, iCap] = sort(capSens.AverageConditionalRange, 'descend');
bh = bar(ax1, sortedCap, 'FaceColor', 'flat');
bh.CData = repmat(COL_BLUE, height(capSens), 1);
bh.CData(1,:) = COL_RED;
ax1.XTickLabel = capSens.Factor(iCap);
styled_axes(ax1, 'Capacitance (B2\_SP\_Ground)', 'Factor', ...
    'Avg. Conditional Range (pF/m)', FSIZE);
% Add value labels
for k = 1:length(sortedCap)
    text(ax1, k, sortedCap(k) + 0.05*max(sortedCap), ...
        sprintf('%.3g', sortedCap(k)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

ax2 = nexttile;
[sortedMut, iMut] = sort(mutSens.AverageConditionalRange, 'descend');
bh2 = bar(ax2, sortedMut, 'FaceColor', 'flat');
bh2.CData = repmat(COL_ORANGE, height(mutSens), 1);
bh2.CData(1,:) = COL_RED;
ax2.XTickLabel = mutSens.Factor(iMut);
styled_axes(ax2, 'Abs. Mutual Inductance (B1\_SP\_Overlap)', 'Factor', ...
    'Avg. Conditional Range (nH)', FSIZE);
for k = 1:length(sortedMut)
    text(ax2, k, sortedMut(k) + 0.05*max(sortedMut), ...
        sprintf('%.3g', sortedMut(k)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

annotation('textbox', [0.01 0.01 0.98 0.04], ...
    'String', ['Source: Stage12B6_GlobalSensitivitySummary.csv | ' ...
    'Metric: Average conditional output range across factor levels | ' ...
    'Analytical dataset only'], ...
    'EdgeColor', 'none', 'FontSize', 8, 'Color', [0.4 0.4 0.4], ...
    'HorizontalAlignment', 'center');

saveas(figA1, fullfile(resultsRoot, 'A1_GlobalSensitivity_ConditionalRange.png'));
fprintf('   Saved A1_GlobalSensitivity_ConditionalRange.png\n');

% -- A2: Spearman rank correlation ------------------------------------------

figA2 = styled_figure(FIG_SIZE);
t_A2 = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t_A2, 'A2. Spearman Rank Correlation - Parameter Influence', ...
    'FontSize', FSIZE+3, 'FontWeight', 'bold');

factorNames = spearman.Row;
capCorr     = spearman.Capacitance;
mutCorr     = spearman.AbsMutualInductance;

ax3 = nexttile;
bh3 = barh(ax3, capCorr);
bh3.FaceColor = COL_BLUE;
ax3.YTickLabel = factorNames;
xline(ax3, 0, 'k--', 'LineWidth', 1);
styled_axes(ax3, 'Capacitance: Spearman \rho', ...
    'Spearman Rank Correlation', 'Factor', FSIZE);
xlim(ax3, [-1 1]);

ax4 = nexttile;
bh4 = barh(ax4, mutCorr);
% Colour by sign
colours = zeros(length(mutCorr), 3);
for k = 1:length(mutCorr)
    if mutCorr(k) < 0
        colours(k,:) = COL_RED;
    else
        colours(k,:) = COL_ORANGE;
    end
end
bh4.CData = colours;
bh4.FaceColor = 'flat';
ax4.YTickLabel = factorNames;
xline(ax4, 0, 'k--', 'LineWidth', 1);
styled_axes(ax4, 'Abs. Mutual Inductance: Spearman \rho', ...
    'Spearman Rank Correlation', 'Factor', FSIZE);
xlim(ax4, [-1 1]);

annotation('textbox', [0.01 0.01 0.98 0.04], ...
    'String', 'Source: Stage11A_Spearman_Results.csv | Stage 11A sensitivity analysis | Analytical dataset', ...
    'EdgeColor', 'none', 'FontSize', 8, 'Color', [0.4 0.4 0.4], ...
    'HorizontalAlignment', 'center');

saveas(figA2, fullfile(resultsRoot, 'A2_SpearmanCorrelation.png'));
fprintf('   Saved A2_SpearmanCorrelation.png\n');

% -- A3: Sensitivity sweep plots --------------------------------------------

figA3 = styled_figure([120 80 1100 680]);
t_A3 = tiledlayout(2, 4, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t_A3, 'A3. Individual Factor Sensitivity Sweeps (All Others Fixed)', ...
    'FontSize', FSIZE+3, 'FontWeight', 'bold');

sweepData  = {sepSens,   pitchSens,  overSens,    gndSens};
sweepFact  = {'Separation_mm', 'Pitch_mm', 'Overlap_mm', 'GroundClearance_mm'};
sweepXLab  = {'Separation (mm)', 'Pitch', 'Overlap (mm)', 'Ground Clearance (mm)'};
sweepTitle = {'Separation Sweep', 'Pitch Sweep', 'Overlap Sweep', 'Ground Clearance Sweep'};
sweepCols  = {COL_BLUE, COL_GREEN, COL_ORANGE, COL_PURPLE};

for k = 1:4
    sd  = sweepData{k};
    fx  = sweepFact{k};
    xl  = sweepXLab{k};

    % Capacitance row
    ax = nexttile;
    plot(ax, sd.(fx), sd.Capacitance_pF_per_m, '-o', ...
        'Color', sweepCols{k}, 'LineWidth', 2, 'MarkerSize', 7, ...
        'MarkerFaceColor', sweepCols{k});
    styled_axes(ax, sweepTitle{k}, xl, 'Capacitance (pF/m)', FSIZE);
    % Fixed factors annotation
    fixedStr = build_fixed_str(sd, fx);
    text(ax, 0.03, 0.95, fixedStr, 'Units', 'normalized', ...
        'FontSize', 8, 'Color', COL_GREY, 'VerticalAlignment', 'top');
end

for k = 1:4
    sd  = sweepData{k};
    fx  = sweepFact{k};
    xl  = sweepXLab{k};

    % Inductance row
    ax = nexttile;
    plot(ax, sd.(fx), sd.AbsMutualInductance_nH, '-s', ...
        'Color', sweepCols{k}, 'LineWidth', 2, 'MarkerSize', 7, ...
        'MarkerFaceColor', sweepCols{k});
    styled_axes(ax, sweepTitle{k}, xl, 'Abs. Mut. Inductance (nH)', FSIZE);
    fixedStr = build_fixed_str(sd, fx);
    text(ax, 0.03, 0.95, fixedStr, 'Units', 'normalized', ...
        'FontSize', 8, 'Color', COL_GREY, 'VerticalAlignment', 'top');
end

annotation('textbox', [0.01 0.01 0.98 0.025], ...
    'String', 'Source: Stage12B5 sensitivity CSVs | Frozen analytical model | Fixed factors shown per panel', ...
    'EdgeColor', 'none', 'FontSize', 8, 'Color', [0.4 0.4 0.4], ...
    'HorizontalAlignment', 'center');

saveas(figA3, fullfile(resultsRoot, 'A3_SensitivitySweeps.png'));
fprintf('   Saved A3_SensitivitySweeps.png\n');

%% =======================================================================
%  B. MULTI-OBJECTIVE PARETO OPTIMISATION
% =======================================================================

fprintf('\n[B] Generating Pareto optimisation plots...\n');

% -- B1: All 216 configs + Pareto front overlay ----------------------------

figB1 = styled_figure(FIG_SIZE);
ax = axes(figB1);

% Non-Pareto configurations
nonPareto = allCfg(allCfg.IsParetoOptimal == 0, :);
scatter(ax, nonPareto.Capacitance_pF_per_m, nonPareto.AbsMutualInductance_nH, ...
    40, COL_GREY, 'o', 'MarkerFaceAlpha', 0.5, 'DisplayName', ...
    sprintf('All configurations (n=%d)', height(nonPareto)));
hold(ax, 'on');

% Pareto configurations (corrected, 6)
scatter(ax, paretoCorr.Capacitance_pF_per_m, paretoCorr.AbsMutualInductance_nH, ...
    120, COL_ORANGE, '^', 'filled', 'LineWidth', 1.2, ...
    'MarkerEdgeColor', 'k', 'DisplayName', ...
    sprintf('Pareto-optimal (corrected, n=%d)', height(paretoCorr)));

% Sort Pareto by capacitance and connect with line
[~, iSort] = sort(paretoCorr.Capacitance_pF_per_m);
plot(ax, paretoCorr.Capacitance_pF_per_m(iSort), ...
    paretoCorr.AbsMutualInductance_nH(iSort), '--', ...
    'Color', COL_ORANGE, 'LineWidth', 1.5, 'HandleVisibility', 'off');

% Label Pareto points
for k = 1:height(paretoCorr)
    text(ax, paretoCorr.Capacitance_pF_per_m(k)+0.3, ...
        paretoCorr.AbsMutualInductance_nH(k), ...
        sprintf('Sep=%d, GC=%d', paretoCorr.Separation_mm(k), ...
        paretoCorr.GroundClearance_mm(k)), ...
        'FontSize', 8, 'Color', [0.55 0.27 0.07]);
end

% Best balanced design (top of balanced ranking)
bestBal = balRank(1,:);
scatter(ax, bestBal.Capacitance_pF_per_m, bestBal.AbsMutualInductance_nH, ...
    200, COL_RED, 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.2, ...
    'DisplayName', sprintf('Best balanced design (Sep=%d)', bestBal.Separation_mm));

styled_axes(ax, 'B1. Multi-Objective Pareto Front: Capacitance vs. Abs. Mutual Inductance', ...
    'Total Capacitance (pF/m)', 'Absolute Mutual Inductance (nH)', FSIZE);
legend(ax, 'Location', 'northeast', 'FontSize', 10);

annotation('textbox', [0.01 0.01 0.98 0.04], ...
    'String', ['Source: Stage12B7_AllConfigurations.csv, Stage12B7_Corrected_Pareto.csv | ' ...
    'Objective: minimise both capacitance AND inductance | No EMC compliance claim implied'], ...
    'EdgeColor', 'none', 'FontSize', 8, 'Color', [0.4 0.4 0.4], ...
    'HorizontalAlignment', 'center');

saveas(figB1, fullfile(resultsRoot, 'B1_Pareto_Front.png'));
fprintf('   Saved B1_Pareto_Front.png\n');

% -- B2: Pareto table coloured by separation -------------------------------

figB2 = styled_figure(FIG_SIZE);
ax = axes(figB2);

sepVals   = unique(allCfg.Separation_mm);
cmap      = lines(length(sepVals));

for k = 1:length(sepVals)
    s    = sepVals(k);
    grp  = allCfg(allCfg.Separation_mm == s, :);
    scatter(ax, grp.Capacitance_pF_per_m, grp.AbsMutualInductance_nH, ...
        35, cmap(k,:), 'o', 'filled', 'MarkerFaceAlpha', 0.55, ...
        'DisplayName', sprintf('Sep = %d mm', s));
    hold(ax, 'on');
end

% Pareto front overlay
scatter(ax, paretoCorr.Capacitance_pF_per_m, paretoCorr.AbsMutualInductance_nH, ...
    130, 'k', '^', 'filled', 'MarkerFaceAlpha', 0.85, ...
    'DisplayName', 'Pareto-optimal (corrected)');
[~, iSort] = sort(paretoCorr.Capacitance_pF_per_m);
plot(ax, paretoCorr.Capacitance_pF_per_m(iSort), ...
    paretoCorr.AbsMutualInductance_nH(iSort), 'k--', 'LineWidth', 1.5, ...
    'HandleVisibility', 'off');

styled_axes(ax, 'B2. Configuration Space Coloured by Separation', ...
    'Total Capacitance (pF/m)', 'Absolute Mutual Inductance (nH)', FSIZE);
legend(ax, 'Location', 'northeast', 'NumColumns', 2, 'FontSize', 9);

saveas(figB2, fullfile(resultsRoot, 'B2_Pareto_BySeparation.png'));
fprintf('   Saved B2_Pareto_BySeparation.png\n');

% Export Pareto table
writetable(paretoCorr, fullfile(resultsRoot, 'B_Pareto_Optimal_Configurations.csv'));
fprintf('   Exported B_Pareto_Optimal_Configurations.csv (%d rows)\n', height(paretoCorr));

%% =======================================================================
%  C. MODEL AGREEMENT ANALYSIS
% =======================================================================

fprintf('\n[C] Generating model agreement analysis...\n');

actCap  = modelPred.ActualCapacitance_pF_m;
predCap = modelPred.PredictedCapacitance_pF_m;
actMut  = modelPred.ActualAbsMutualInductance_nH;
predMut = modelPred.PredictedAbsMutualInductance_nH;

% Compute metrics
[mae_c, rmse_c, maxe_c, r2_c] = compute_metrics(actCap, predCap);
[mae_m, rmse_m, maxe_m, r2_m] = compute_metrics(actMut, predMut);

% -- C1: Predicted vs. Actual ----------------------------------------------

figC1 = styled_figure([120 80 1000 520]);
t_C1 = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t_C1, 'C1. Model Agreement: Predicted vs. Analytical Reference (Stage 12A)', ...
    'FontSize', FSIZE+3, 'FontWeight', 'bold');

ax1 = nexttile;
scatter(ax1, actCap, predCap, 45, COL_BLUE, 'filled', ...
    'MarkerFaceAlpha', 0.6, 'DisplayName', 'Configurations');
hold(ax1, 'on');
lims = [min([actCap; predCap])-1, max([actCap; predCap])+1];
plot(ax1, lims, lims, 'r--', 'LineWidth', 2, 'DisplayName', 'y = x (ideal)');
styled_axes(ax1, 'Capacitance: Predicted vs. Actual', ...
    'Actual (pF/m)', 'Predicted (pF/m)', FSIZE);
xlim(ax1, lims); ylim(ax1, lims);
legend(ax1, 'Location', 'northwest');
text(ax1, 0.05, 0.98, ...
    sprintf('MAE = %.3e pF/m\nRMSE = %.3e pF/m\nMax |e| = %.3e pF/m\nR^2 = %.6f', ...
    mae_c, rmse_c, maxe_c, r2_c), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontSize', 9, 'BackgroundColor', [0.95 0.95 0.95]);

ax2 = nexttile;
scatter(ax2, actMut, predMut, 45, COL_ORANGE, 'filled', ...
    'MarkerFaceAlpha', 0.6, 'DisplayName', 'Configurations');
hold(ax2, 'on');
lims2 = [min([actMut; predMut])-0.1, max([actMut; predMut])+0.1];
plot(ax2, lims2, lims2, 'r--', 'LineWidth', 2, 'DisplayName', 'y = x (ideal)');
styled_axes(ax2, 'Abs. Mutual Inductance: Predicted vs. Actual', ...
    'Actual (nH)', 'Predicted (nH)', FSIZE);
xlim(ax2, lims2); ylim(ax2, lims2);
legend(ax2, 'Location', 'northwest');
text(ax2, 0.05, 0.98, ...
    sprintf('MAE = %.3e nH\nRMSE = %.3e nH\nMax |e| = %.3e nH\nR^2 = %.6f', ...
    mae_m, rmse_m, maxe_m, r2_m), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontSize', 9, 'BackgroundColor', [0.95 0.95 0.95]);

annotation('textbox', [0.01 0.01 0.98 0.04], ...
    'String', ['NOTICE: R^2 ~ 1 and sub-femto residuals reflect EXACT least-squares reproduction '...
    'of the analytical dataset, not independent experimental validation. '...
    'Source: Stage12A_Selected_Model_Predictions.csv'], ...
    'EdgeColor', [0.8 0.5 0], 'BackgroundColor', [1 0.97 0.87], ...
    'FontSize', 8, 'Color', [0.5 0.3 0], ...
    'HorizontalAlignment', 'center');

saveas(figC1, fullfile(resultsRoot, 'C1_ModelAgreement_PredVsActual.png'));
fprintf('   Saved C1_ModelAgreement_PredVsActual.png\n');

% Print summary
fprintf('\n   --- Model Agreement Metrics (Stage 12A frozen model) ---\n');
fprintf('   Capacitance:         MAE=%.3e pF/m  RMSE=%.3e pF/m  R2=%.8f\n', ...
    mae_c, rmse_c, r2_c);
fprintf('   Abs. Mut. Inductance: MAE=%.3e nH   RMSE=%.3e nH   R2=%.8f\n', ...
    mae_m, rmse_m, r2_m);
fprintf('   (These metrics reflect dataset reproduction, not physical validation)\n\n');

%% =======================================================================
%  D. RESIDUAL ANALYSIS
% =======================================================================

fprintf('\n[D] Generating residual analysis...\n');

resC = modelPred.CapacitanceResidual_pF_m;
resM = modelPred.AbsMutualInductanceResidual_nH;

% -- D1: Residual histograms -----------------------------------------------

figD1 = styled_figure([120 80 1000 520]);
t_D1 = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t_D1, 'D1. Residual Distributions (Stage 12A Model)', ...
    'FontSize', FSIZE+3, 'FontWeight', 'bold');

ax1 = nexttile;
histogram(ax1, resC, 20, 'FaceColor', COL_BLUE, 'EdgeColor', 'w', ...
    'FaceAlpha', 0.8);
xline(ax1, 0, 'r--', 'LineWidth', 1.5, 'Label', 'Zero');
styled_axes(ax1, 'Capacitance Residuals', 'Residual (pF/m)', 'Count', FSIZE);
text(ax1, 0.05, 0.95, ...
    sprintf('Mean = %.2e\nStd  = %.2e\nMax|e| = %.2e', ...
    mean(resC), std(resC), max(abs(resC))), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 9, ...
    'BackgroundColor', [0.94 0.94 0.98]);

ax2 = nexttile;
histogram(ax2, resM, 20, 'FaceColor', COL_ORANGE, 'EdgeColor', 'w', ...
    'FaceAlpha', 0.8);
xline(ax2, 0, 'r--', 'LineWidth', 1.5, 'Label', 'Zero');
styled_axes(ax2, 'Abs. Mut. Inductance Residuals', 'Residual (nH)', 'Count', FSIZE);
text(ax2, 0.05, 0.95, ...
    sprintf('Mean = %.2e\nStd  = %.2e\nMax|e| = %.2e', ...
    mean(resM), std(resM), max(abs(resM))), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 9, ...
    'BackgroundColor', [1 0.96 0.87]);

annotation('textbox', [0.01 0.01 0.98 0.04], ...
    'String', ['Residuals are at floating-point machine precision (~ 10^-14 to 10^-13 level). '...
    'This reflects exact algebraic reconstruction, not measurement uncertainty. '...
    'Source: Stage12A_Selected_Model_Predictions.csv'], ...
    'EdgeColor', 'none', 'FontSize', 8, 'Color', [0.4 0.4 0.4], ...
    'HorizontalAlignment', 'center');

saveas(figD1, fullfile(resultsRoot, 'D1_ResidualHistograms.png'));
fprintf('   Saved D1_ResidualHistograms.png\n');

% -- D2: Residual vs. Predicted --------------------------------------------

figD2 = styled_figure([120 80 1000 520]);
t_D2 = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t_D2, 'D2. Residuals vs. Predicted Values', ...
    'FontSize', FSIZE+3, 'FontWeight', 'bold');

ax1 = nexttile;
scatter(ax1, predCap, resC, 45, COL_BLUE, 'filled', 'MarkerFaceAlpha', 0.55);
yline(ax1, 0, 'r--', 'LineWidth', 1.5);
styled_axes(ax1, 'Capacitance', 'Predicted (pF/m)', 'Residual (pF/m)', FSIZE);

ax2 = nexttile;
scatter(ax2, predMut, resM, 45, COL_ORANGE, 'filled', 'MarkerFaceAlpha', 0.55);
yline(ax2, 0, 'r--', 'LineWidth', 1.5);
styled_axes(ax2, 'Abs. Mutual Inductance', 'Predicted (nH)', 'Residual (nH)', FSIZE);

saveas(figD2, fullfile(resultsRoot, 'D2_ResidualVsPredicted.png'));
fprintf('   Saved D2_ResidualVsPredicted.png\n');

%% =======================================================================
%  E. ENGINEERING RESPONSE SURFACES
% =======================================================================

fprintf('\n[E] Generating engineering response surfaces...\n');

% Only discrete grid levels are available; no interpolated continuous surface
% is generated. Surfaces show trends across the actual design grid points.

sepLevels = unique(allCfg.Separation_mm);   % [8 12 16 20 24 28]
pitchLevels = unique(allCfg.Pitch_mm);       % [2 3 4]
ovLevels  = unique(allCfg.Overlap_mm);       % [150 300 450]
gcLevels  = unique(allCfg.GroundClearance_mm); % [10 25 50 100]

% -- E1: Capacitance surface: Separation x Ground Clearance (fixed Pitch=2, OL=300) -

figE1 = styled_figure([120 80 1100 500]);
t_E1 = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t_E1, 'E1. Response Surface: Separation x Ground Clearance (Pitch=2, Overlap=300 mm)', ...
    'FontSize', FSIZE+3, 'FontWeight', 'bold');

subE1 = allCfg(allCfg.Pitch_mm == 2 & allCfg.Overlap_mm == 300, :);

% Build matrices for surf
nSep = length(sepLevels);
nGC  = length(gcLevels);
CapMat = zeros(nGC, nSep);
MutMat = zeros(nGC, nSep);
for i = 1:nSep
    for j = 1:nGC
        row = subE1(subE1.Separation_mm == sepLevels(i) & ...
            subE1.GroundClearance_mm == gcLevels(j), :);
        if ~isempty(row)
            CapMat(j,i) = row.Capacitance_pF_per_m(1);
            MutMat(j,i) = row.AbsMutualInductance_nH(1);
        end
    end
end

[SEP, GC] = meshgrid(sepLevels, gcLevels);

ax1 = nexttile;
surf(ax1, SEP, GC, CapMat, 'EdgeColor', [0.3 0.3 0.3], 'EdgeAlpha', 0.4);
colormap(ax1, 'turbo');
colorbar(ax1);
styled_axes(ax1, 'Capacitance (pF/m)', 'Separation (mm)', 'Ground Clearance (mm)', FSIZE);
zlabel(ax1, 'Capacitance (pF/m)', 'FontSize', FSIZE);
view(ax1, -45, 30);

ax2 = nexttile;
surf(ax2, SEP, GC, MutMat, 'EdgeColor', [0.3 0.3 0.3], 'EdgeAlpha', 0.4);
colormap(ax2, 'turbo');
colorbar(ax2);
styled_axes(ax2, 'Abs. Mutual Inductance (nH)', 'Separation (mm)', 'Ground Clearance (mm)', FSIZE);
zlabel(ax2, 'Inductance (nH)', 'FontSize', FSIZE);
view(ax2, -45, 30);

annotation('textbox', [0.01 0.01 0.98 0.035], ...
    'String', ['Fixed: Pitch=2, Overlap=300 mm | Discrete grid values only - no interpolation | ' ...
    'Source: Stage12B7_AllConfigurations.csv'], ...
    'EdgeColor', 'none', 'FontSize', 8, 'Color', [0.4 0.4 0.4], ...
    'HorizontalAlignment', 'center');

saveas(figE1, fullfile(resultsRoot, 'E1_ResponseSurface_Sep_GC.png'));
fprintf('   Saved E1_ResponseSurface_Sep_GC.png\n');

% -- E2: Capacitance surface: Separation x Pitch (fixed OL=300, GC=50) ----

figE2 = styled_figure([120 80 1100 500]);
t_E2 = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
title(t_E2, 'E2. Response Surface: Separation x Pitch (Overlap=300 mm, GC=50 mm)', ...
    'FontSize', FSIZE+3, 'FontWeight', 'bold');

subE2 = allCfg(allCfg.Overlap_mm == 300 & allCfg.GroundClearance_mm == 50, :);

nP = length(pitchLevels);
CapMat2 = zeros(nP, nSep);
MutMat2 = zeros(nP, nSep);
for i = 1:nSep
    for j = 1:nP
        row = subE2(subE2.Separation_mm == sepLevels(i) & ...
            subE2.Pitch_mm == pitchLevels(j), :);
        if ~isempty(row)
            CapMat2(j,i) = row.Capacitance_pF_per_m(1);
            MutMat2(j,i) = row.AbsMutualInductance_nH(1);
        end
    end
end

[SEP2, PITCH] = meshgrid(sepLevels, pitchLevels);

ax1 = nexttile;
surf(ax1, SEP2, PITCH, CapMat2, 'EdgeColor', [0.3 0.3 0.3], 'EdgeAlpha', 0.4);
colormap(ax1, 'cool');
colorbar(ax1);
styled_axes(ax1, 'Capacitance (pF/m)', 'Separation (mm)', 'Pitch', FSIZE);
zlabel(ax1, 'Capacitance (pF/m)'); view(ax1, -50, 30);

ax2 = nexttile;
surf(ax2, SEP2, PITCH, MutMat2, 'EdgeColor', [0.3 0.3 0.3], 'EdgeAlpha', 0.4);
colormap(ax2, 'cool');
colorbar(ax2);
styled_axes(ax2, 'Abs. Mutual Inductance (nH)', 'Separation (mm)', 'Pitch', FSIZE);
zlabel(ax2, 'Inductance (nH)'); view(ax2, -50, 30);

annotation('textbox', [0.01 0.01 0.98 0.035], ...
    'String', 'Fixed: Overlap=300 mm, GC=50 mm | Discrete grid values only', ...
    'EdgeColor', 'none', 'FontSize', 8, 'Color', [0.4 0.4 0.4], ...
    'HorizontalAlignment', 'center');

saveas(figE2, fullfile(resultsRoot, 'E2_ResponseSurface_Sep_Pitch.png'));
fprintf('   Saved E2_ResponseSurface_Sep_Pitch.png\n');

%% =======================================================================
%  F. PROFESSIONAL 3D CABLE GEOMETRY VISUALISATION
% =======================================================================

fprintf('\n[F] Generating 3D cable geometry visualisation...\n');

% Representative configuration from best balanced design
cfgSep  = balRank.Separation_mm(1);
cfgPitch = balRank.Pitch_mm(1);
cfgOL   = balRank.Overlap_mm(1);
cfgGC   = balRank.GroundClearance_mm(1);

figF1 = styled_figure([50 50 1100 700]);
figF1.Color = [0.06 0.09 0.14];

ax3d = axes(figF1);
ax3d.Color = [0.07 0.11 0.17];
ax3d.GridColor = [0.3 0.35 0.42];
ax3d.XColor = [0.7 0.75 0.85];
ax3d.YColor = [0.7 0.75 0.85];
ax3d.ZColor = [0.7 0.75 0.85];
ax3d.LineWidth = 1;
grid(ax3d, 'on');
hold(ax3d, 'on');

% Geometry constants (illustrative representation; NOT a verified KONE cross-section)
cableR   = 3.0;    % representative cable radius, mm
n_theta  = 48;     % angular resolution for cylinders

% Coordinate convention:
%   X = horizontal (across cables)
%   Y = vertical (height above ground)
%   Z = along cable run direction

% Source cable: centre at (0, cfgGC)
% Victim cable: centre at (cfgSep, cfgGC + cfgPitch)
%   Note: cfgPitch is used as a small vertical offset (structural stagger)
%         consistent with the parameterised model geometry.

srcX = 0;
srcY = cfgGC;
vicX = cfgSep;
vicY = cfgGC + cfgPitch;

srcLen = cfgOL + 60;
vicLen = cfgOL;

% -- Draw source cable ----------------------------------------------------
[Xc, Yc, Zc] = build_cylinder(srcX, srcY, srcLen, cableR, n_theta);
surf(ax3d, Xc, Yc, Zc, 'FaceColor', [0.18 0.55 0.95], 'EdgeColor', 'none', ...
    'FaceAlpha', 0.88, 'SpecularStrength', 0.7, 'AmbientStrength', 0.35);

% Source cable end caps
[Xend, Yend] = get_circle(srcX, srcY, cableR, n_theta);
fill3(ax3d, Xend, Yend, repmat(-srcLen/2, 1, n_theta+1), [0.18 0.55 0.95], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.88);
fill3(ax3d, Xend, Yend, repmat(srcLen/2, 1, n_theta+1), [0.18 0.55 0.95], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.88);

% -- Draw victim cable -----------------------------------------------------
[Xv, Yv, Zv] = build_cylinder(vicX, vicY, vicLen, cableR, n_theta);
surf(ax3d, Xv, Yv, Zv, 'FaceColor', [1.00 0.55 0.10], 'EdgeColor', 'none', ...
    'FaceAlpha', 0.88, 'SpecularStrength', 0.7, 'AmbientStrength', 0.35);
[Xend2, Yend2] = get_circle(vicX, vicY, cableR, n_theta);
fill3(ax3d, Xend2, Yend2, repmat(-vicLen/2, 1, n_theta+1), [1.00 0.55 0.10], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.88);
fill3(ax3d, Xend2, Yend2, repmat(vicLen/2, 1, n_theta+1), [1.00 0.55 0.10], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.88);

% -- Ground reference plane ------------------------------------------------
gpXLim = [-cableR - 12, vicX + cableR + 12];
gpZLim = [-srcLen/2 - 30, srcLen/2 + 30];
[Xgp, Zgp] = meshgrid(linspace(gpXLim(1), gpXLim(2), 3), ...
    linspace(gpZLim(1), gpZLim(2), 3));
surf(ax3d, Xgp, zeros(size(Xgp)), Zgp, ...
    'FaceColor', [0.25 0.35 0.28], 'EdgeColor', [0.35 0.45 0.38], ...
    'EdgeAlpha', 0.4, 'FaceAlpha', 0.35, 'HandleVisibility', 'off');

% -- Overlap bracket markers -----------------------------------------------
bracketZ = [-vicLen/2, vicLen/2];
for z = bracketZ
    plot3(ax3d, [srcX - cableR - 4, vicX + cableR + 4], [srcY, vicY], [z z], ...
        '--', 'Color', [1 0.2 0.2], 'LineWidth', 1.5);
end

% -- Separation dimension line ---------------------------------------------
dimZ = srcLen/2 + 18;
plot3(ax3d, [srcX, vicX], [srcY, vicY], [dimZ, dimZ], ...
    '-', 'Color', [0.4 1 0.4], 'LineWidth', 2.5);
plot3(ax3d, [srcX, srcX], [srcY, srcY], [dimZ-5, dimZ+5], ...
    '-', 'Color', [0.4 1 0.4], 'LineWidth', 2);
plot3(ax3d, [vicX, vicX], [vicY, vicY], [dimZ-5, dimZ+5], ...
    '-', 'Color', [0.4 1 0.4], 'LineWidth', 2);
text(ax3d, (srcX+vicX)/2, (srcY+vicY)/2, dimZ + 7, ...
    sprintf('Sep = %d mm', cfgSep), ...
    'Color', [0.4 1 0.4], 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');

% -- Ground clearance dimension line ---------------------------------------
gcX = srcX - cableR - 10;
gcZ = -srcLen/2 + 20;
plot3(ax3d, [gcX, gcX], [0, srcY], [gcZ, gcZ], ...
    '-', 'Color', [0.8 0.8 0.8], 'LineWidth', 2);
plot3(ax3d, [gcX-4, gcX+4], [0, 0], [gcZ, gcZ], ...
    '-', 'Color', [0.8 0.8 0.8], 'LineWidth', 2);
plot3(ax3d, [gcX-4, gcX+4], [srcY, srcY], [gcZ, gcZ], ...
    '-', 'Color', [0.8 0.8 0.8], 'LineWidth', 2);
text(ax3d, gcX - 6, srcY/2, gcZ, ...
    sprintf('GC=%d mm', cfgGC), ...
    'Color', [0.8 0.8 0.8], 'FontSize', 9, 'Rotation', 0, ...
    'HorizontalAlignment', 'right');

% -- Overlap dimension line ------------------------------------------------
ovX = vicX + cableR + 12;
ovY = vicY;
plot3(ax3d, [ovX, ovX], [ovY, ovY], [-vicLen/2, vicLen/2], ...
    '-', 'Color', [1 0.8 0.2], 'LineWidth', 2);
text(ax3d, ovX + 6, ovY, 0, ...
    sprintf('OL=%d mm', cfgOL), ...
    'Color', [1 0.8 0.2], 'FontSize', 9, ...
    'HorizontalAlignment', 'left');

% -- Pitch annotation ------------------------------------------------------
ptZ = vicLen/2 + 15;
plot3(ax3d, [vicX, vicX], [srcY, vicY], [ptZ, ptZ], ...
    '-', 'Color', [0.85 0.5 1], 'LineWidth', 2);
text(ax3d, vicX + 3, (srcY+vicY)/2, ptZ, ...
    sprintf('Pitch=%d', cfgPitch), ...
    'Color', [0.85 0.5 1], 'FontSize', 9);

% -- Lighting --------------------------------------------------------------
camlight(ax3d, 'headlight');
material(ax3d, 'dull');
lighting(ax3d, 'gouraud');

% -- Labels & formatting --------------------------------------------------
xlabel(ax3d, 'X  -  Cross-cable direction (mm)', 'FontSize', FSIZE, 'Color', [0.7 0.75 0.85]);
ylabel(ax3d, 'Y  -  Height above ground (mm)',   'FontSize', FSIZE, 'Color', [0.7 0.75 0.85]);
zlabel(ax3d, 'Z  -  Cable run direction (mm)',    'FontSize', FSIZE, 'Color', [0.7 0.75 0.85]);

titleStr = sprintf(['F1. Cable Configuration 3D Geometry\n' ...
    'Sep=%d mm  |  Pitch=%d  |  Overlap=%d mm  |  GC=%d mm'], ...
    cfgSep, cfgPitch, cfgOL, cfgGC);
title(ax3d, titleStr, 'FontSize', FSIZE+2, 'FontWeight', 'bold', ...
    'Color', [0.9 0.92 0.98]);

legend(ax3d, {'Source cable (blue)', 'Victim cable (orange)', ...
    'Overlap boundary', 'Separation', 'Ground clearance', 'Overlap', 'Pitch'}, ...
    'Location', 'northeast', 'FontSize', 9, 'TextColor', [0.88 0.9 0.96], ...
    'Color', [0.09 0.13 0.2], 'EdgeColor', [0.3 0.35 0.42]);

view(ax3d, -38, 22);
axis(ax3d, 'equal');

% Annotation
annStr = ['ILLUSTRATIVE ENGINEERING REPRESENTATION: Cable geometry is parameterised ' ...
    'by separation, pitch, overlap and ground clearance. This is not a verified ' ...
    'KONE cable cross-section or a full electromagnetic field simulation.'];
annotation(figF1, 'textbox', [0.01 0.01 0.98 0.04], ...
    'String', annStr, ...
    'EdgeColor', [0.3 0.35 0.42], 'BackgroundColor', [0.09 0.13 0.2], ...
    'FontSize', 8, 'Color', [0.55 0.65 0.75], ...
    'HorizontalAlignment', 'center');

saveas(figF1, fullfile(resultsRoot, 'F1_3D_Cable_Geometry.png'));
fprintf('   Saved F1_3D_Cable_Geometry.png\n');

% -- F2: Static views of all supported geometric configurations ------------

figF2 = styled_figure([50 50 1100 700]);
figF2.Color = [0.06 0.09 0.14];
t_F2 = tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
t_F2.TileSpacing = 'compact';

titleObj = title(t_F2, 'F2. Selected Configuration Cross-Sections (Y-X plane at Z=0)', ...
    'Color', [0.88 0.91 0.97], 'FontSize', FSIZE+2, 'FontWeight', 'bold');

sampConfigs = [
    8,  2, 150, 10;
    12, 2, 150, 100;
    20, 3, 300, 50;
    28, 4, 450, 100;
    16, 3, 300, 25;
    24, 2, 150, 50;
];

for k = 1:6
    s  = sampConfigs(k,1);
    p  = sampConfigs(k,2);
    ov = sampConfigs(k,3);
    gc = sampConfigs(k,4);

    ax = nexttile;
    ax.Color = [0.07 0.11 0.17];
    ax.XColor = [0.6 0.65 0.75];
    ax.YColor = [0.6 0.65 0.75];
    hold(ax, 'on');

    % Ground line
    xr = [-cableR-8, s+cableR+8];
    plot(ax, xr, [0 0], '-', 'Color', [0.35 0.45 0.38], 'LineWidth', 2.5);

    % Source cable
    theta = linspace(0, 2*pi, 64);
    fill(ax, 0 + cableR*cos(theta), gc + cableR*sin(theta), ...
        [0.18 0.55 0.95], 'EdgeColor', [0.6 0.7 1], 'LineWidth', 1.2, ...
        'FaceAlpha', 0.85);
    text(ax, 0, gc, 'SRC', 'Color', 'w', 'FontSize', 7, ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');

    % Victim cable
    fill(ax, s + cableR*cos(theta), (gc+p) + cableR*sin(theta), ...
        [1 0.55 0.1], 'EdgeColor', [1 0.8 0.5], 'LineWidth', 1.2, ...
        'FaceAlpha', 0.85);
    text(ax, s, gc+p, 'VIC', 'Color', 'w', 'FontSize', 7, ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');

    % Separation arrow
    annotation_arrow(ax, [0, s], [gc, gc+p], s/2, gc-3, ...
        sprintf('Sep=%d', s), [0.4 1 0.4]);

    % Ground clearance
    text(ax, -6, gc/2, sprintf('GC=%d', gc), ...
        'Color', [0.75 0.75 0.75], 'FontSize', 7, 'HorizontalAlignment', 'right');

    title(ax, sprintf('Sep=%d  P=%d  OL=%d  GC=%d', s, p, ov, gc), ...
        'FontSize', 8.5, 'Color', [0.85 0.9 0.97], 'FontWeight', 'bold');
    axis(ax, 'equal');
    xlim(ax, [-10, s+10]);
    ylim(ax, [-4, max(gc+p+cableR+6, 25)]);
    xlabel(ax, 'X (mm)', 'FontSize', 8, 'Color', [0.6 0.65 0.75]);
    ylabel(ax, 'Y (mm)', 'FontSize', 8, 'Color', [0.6 0.65 0.75]);
    grid(ax, 'on');
    ax.GridAlpha = 0.2;
    ax.GridColor = [0.3 0.35 0.42];
end

saveas(figF2, fullfile(resultsRoot, 'F2_CrossSection_MultipleConfigs.png'));
fprintf('   Saved F2_CrossSection_MultipleConfigs.png\n');

%% =======================================================================
%  EXPORT COMPLETE RESULTS TABLE
% =======================================================================

fprintf('\n[7] Exporting results tables...\n');

% Full configuration table with Pareto flag and balanced score (where available)
allCfgFull = allCfg;
writetable(allCfgFull, fullfile(resultsRoot, 'All_216_Configurations_With_Pareto.csv'));
writetable(balRank,    fullfile(resultsRoot, 'Balanced_Design_Ranking.csv'));
writetable(modelSumm,  fullfile(resultsRoot, 'Stage12A_Model_Summary.csv'));
writetable(globalSens, fullfile(resultsRoot, 'GlobalSensitivity_Summary.csv'));

fprintf('   Exported 4 result tables.\n');

%% =======================================================================
%  FINAL SUMMARY
% =======================================================================

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('  KONE Final Engineering Demo COMPLETE\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('  Output folder: %s\n', resultsRoot);
fprintf('\n  Figures generated:\n');
fprintf('    A1_GlobalSensitivity_ConditionalRange.png\n');
fprintf('    A2_SpearmanCorrelation.png\n');
fprintf('    A3_SensitivitySweeps.png\n');
fprintf('    B1_Pareto_Front.png\n');
fprintf('    B2_Pareto_BySeparation.png\n');
fprintf('    C1_ModelAgreement_PredVsActual.png\n');
fprintf('    D1_ResidualHistograms.png\n');
fprintf('    D2_ResidualVsPredicted.png\n');
fprintf('    E1_ResponseSurface_Sep_GC.png\n');
fprintf('    E2_ResponseSurface_Sep_Pitch.png\n');
fprintf('    F1_3D_Cable_Geometry.png\n');
fprintf('    F2_CrossSection_MultipleConfigs.png\n');
fprintf('\n  CSV exports:\n');
fprintf('    B_Pareto_Optimal_Configurations.csv\n');
fprintf('    All_216_Configurations_With_Pareto.csv\n');
fprintf('    Balanced_Design_Ranking.csv\n');
fprintf('    Stage12A_Model_Summary.csv\n');
fprintf('    GlobalSensitivity_Summary.csv\n');
fprintf('\n  IMPORTANT NOTICES:\n');
fprintf('    * All results are from the deterministic analytical MATLAB dataset.\n');
fprintf('    * R^2~=1 reflects dataset reproduction, NOT independent validation.\n');
fprintf('    * No physical EMC measurements or compliance determination is shown.\n');
fprintf('    * 3D geometry is an illustrative parameterised representation.\n');
fprintf('%s\n\n', repmat('=', 1, 70));

%% =======================================================================
%  LOCAL HELPER FUNCTIONS
% =======================================================================

function [Xs, Ys, Zs] = build_cylinder(cx, cy, len, r, n)
%BUILD_CYLINDER  Generate surface matrices for a cylinder along Z.
    theta = linspace(0, 2*pi, n+1);
    z     = [-len/2, len/2];
    [TH, Z] = meshgrid(theta, z);
    Xs = cx + r * cos(TH);
    Ys = cy + r * sin(TH);
    Zs = Z;
end

function [Xc, Yc] = get_circle(cx, cy, r, n)
%GET_CIRCLE  Circle coordinates for end-cap fill.
    theta = linspace(0, 2*pi, n+1);
    Xc = cx + r * cos(theta);
    Yc = cy + r * sin(theta);
end

function [mae, rmse, maxe, r2] = compute_metrics(actual, predicted)
%COMPUTE_METRICS  MAE, RMSE, max absolute error and R^2.
    err  = predicted - actual;
    mae  = mean(abs(err));
    rmse = sqrt(mean(err.^2));
    maxe = max(abs(err));
    ss_res = sum(err.^2);
    ss_tot = sum((actual - mean(actual)).^2);
    if ss_tot < eps
        r2 = 1.0;   % Degenerate case: constant response
    else
        r2 = 1 - ss_res / ss_tot;
    end
end

function annotation_arrow(ax, xs, ys, tx, ty, label, col)
%ANNOTATION_ARROW  Draw a simple dimensioning line on axes ax.
    plot(ax, xs, ys, '-', 'Color', col, 'LineWidth', 1.5);
    text(ax, tx, ty, label, 'Color', col, 'FontSize', 7, ...
        'HorizontalAlignment', 'center');
end

function s = build_fixed_str(tbl, varying_field)
%BUILD_FIXED_STR  Build annotation string of fixed factor values.
    allFields = {'Separation_mm', 'Pitch_mm', 'Overlap_mm', 'GroundClearance_mm'};
    labels    = {'Sep', 'Pitch', 'OL', 'GC'};
    units     = {' mm', '', ' mm', ' mm'};
    parts = {};
    for k = 1:length(allFields)
        if ~strcmp(allFields{k}, varying_field)
            val = tbl.(allFields{k})(1);
            parts{end+1} = sprintf('%s=%g%s', labels{k}, val, units{k});
        end
    end
    s = strjoin(parts, ', ');
end


function h = styled_figure(fig_size)
    h = figure('Color', [1 1 1], 'Position', fig_size);
end

function styled_axes(ax, title_str, xlab, ylab, fsize)
    ax.FontSize   = fsize;
    ax.Box        = 'on';
    ax.GridAlpha  = 0.25;
    ax.MinorGridAlpha = 0.15;
    grid(ax, 'on');
    title(ax, title_str, 'FontSize', fsize+2, 'FontWeight', 'bold');
    xlabel(ax, xlab,     'FontSize', fsize+1);
    ylabel(ax, ylab,     'FontSize', fsize+1);
end
