
%% EMC-GUARD STAGE 11C
% Model Adequacy, Residual Diagnostics and LOOCV Robustness
% Rebuilds the Stage 11B two-factor interaction model.
% No additional toolboxes are required.

clc;
clear;
close all;

fprintf('\n===== EMC-GUARD STAGE 11C =====\n');
fprintf('MODEL ADEQUACY AND ROBUSTNESS ANALYSIS\n\n');

%% 1. PATHS

baseFolder = ...
    'D:\Downloads\kone\01_MATLAB_Models';

stage10Folder = fullfile(baseFolder, '05_Stage10C_Results');
resultFolder = fullfile(baseFolder, '06_Stage11A_Results');

dataFile = fullfile(stage10Folder, ...
    'Stage10C_All_Results.csv');

stage11BFile = fullfile(resultFolder, ...
    'Stage11B_Complete.mat');

if ~isfile(dataFile)
    error('Stage10C CSV file not found: %s', dataFile);
end

if ~isfile(stage11BFile)
    error('Stage11B MAT file not found: %s', stage11BFile);
end

if ~isfolder(resultFolder)
    mkdir(resultFolder);
end

%% 2. LOAD ORIGINAL DATA

T = readtable(dataFile);

requiredVariables = {
    'PairSeparation_mm'
    'PairPitch_mm'
    'Overlap_mm'
    'GroundClearance_mm'
    'TotalCapacitance_pF_per_m'
    'AbsMutualInductance_nH'
};

for k = 1:numel(requiredVariables)
    if ~ismember(requiredVariables{k}, T.Properties.VariableNames)
        error('Required column missing: %s', requiredVariables{k});
    end
end

factorNames = {
    'PairSeparation'
    'PairPitch'
    'Overlap'
    'GroundClearance'
};

factorColumns = {
    'PairSeparation_mm'
    'PairPitch_mm'
    'Overlap_mm'
    'GroundClearance_mm'
};

Y = [
    T.TotalCapacitance_pF_per_m, ...
    T.AbsMutualInductance_nH
];

outputNames = {
    'Capacitance'
    'AbsMutualInductance'
};

n = height(T);

if n ~= 216
    warning('Expected 216 configurations, but found %d.', n);
end

if any(~isfinite(Y), 'all')
    error('Output data contains NaN or Inf values.');
end

fprintf('Configurations loaded: %d\n', n);

%% 3. BUILD FACTOR MATRICES

F = cell(1,4);
factorLevels = cell(1,4);

for i = 1:4
    F{i} = T.(factorColumns{i});
    factorLevels{i} = unique(F{i});
end

fprintf('\nFactor levels:\n');

for i = 1:4
    fprintf('%s: %d levels\n', ...
        factorNames{i}, numel(factorLevels{i}));
end

%% 4. BUILD FULL TWO-FACTOR INTERACTION MODEL

% Reference-level dummy coding:
% Intercept + main effects + all six two-factor interactions.

X = ones(n,1);
designNames = {'Intercept'};

D = cell(1,4);

for i = 1:4

    levels = factorLevels{i};
    D{i} = zeros(n, numel(levels)-1);

    for j = 2:numel(levels)
        D{i}(:,j-1) = double(F{i} == levels(j));

        designNames{end+1} = sprintf( ...
            '%s_Level_%g', factorNames{i}, levels(j));
    end

    X = [X, D{i}];

end

interactionPairs = nchoosek(1:4,2);

interactionNames = {
    'Separation_x_Pitch'
    'Separation_x_Overlap'
    'Separation_x_GroundClearance'
    'Pitch_x_Overlap'
    'Pitch_x_GroundClearance'
    'Overlap_x_GroundClearance'
};

interactionColumns = cell(6,1);

for k = 1:6

    a = interactionPairs(k,1);
    b = interactionPairs(k,2);

    A = D{a};
    B = D{b};

    block = [];

    for ia = 1:size(A,2)
        for ib = 1:size(B,2)

            block = [block, A(:,ia).*B(:,ib)];

            designNames{end+1} = sprintf( ...
                '%s_x_%s', ...
                factorNames{a}, factorNames{b});

        end
    end

    interactionColumns{k} = size(X,2) + (1:size(block,2));

    X = [X, block];

end

%% 5. MODEL STRUCTURE AUDIT

modelRank = rank(X);
numberParameters = size(X,2);
residualDF = n - modelRank;

fprintf('\n===== MODEL STRUCTURE =====\n');
fprintf('Design matrix size: %d x %d\n', size(X,1), size(X,2));
fprintf('Model rank: %d\n', modelRank);
fprintf('Number of parameters: %d\n', numberParameters);
fprintf('Residual degrees of freedom: %d\n', residualDF);

if modelRank < numberParameters
    error('Design matrix is rank deficient. Stop and investigate.');
end

if residualDF <= 0
    error('Insufficient residual degrees of freedom.');
end

conditionNumber = cond(X);

fprintf('Design matrix condition number: %.6g\n', conditionNumber);

%% 6. FIT BOTH OUTPUT MODELS

Summary = table();
PredictionResults = table();

residualsAll = zeros(n,2);
predictedAll = zeros(n,2);
betaAll = cell(1,2);
leverageAll = cell(1,2);

for j = 1:2

    y = Y(:,j);

    beta = X \ y;
    predicted = X * beta;
    residuals = y - predicted;

    betaAll{j} = beta;
    predictedAll(:,j) = predicted;
    residualsAll(:,j) = residuals;

    SSE = sum(residuals.^2);
    SST = sum((y - mean(y)).^2);

    R2 = 1 - SSE/SST;

    adjustedR2 = 1 - ...
        (SSE/residualDF)/(SST/(n-1));

    RMSE = sqrt(mean(residuals.^2));
    MAE = mean(abs(residuals));
    maxError = max(abs(residuals));

    %% 7. LEAVE-ONE-OUT CROSS-VALIDATION

    [Q,~] = qr(X,0);

    leverage = sum(Q(:,1:modelRank).^2,2);
    leverageAll{j} = leverage;

    denominator = 1 - leverage;

    if any(denominator < 1e-10)
        warning('Very high leverage detected for %s.', outputNames{j});
        PRESS = NaN;
        CV_RMSE = NaN;
        Q2 = NaN;
    else
        looResiduals = residuals ./ denominator;

        PRESS = sum(looResiduals.^2);
        CV_RMSE = sqrt(PRESS/n);
        Q2 = 1 - PRESS/SST;
    end

    fprintf('\n===== %s =====\n', outputNames{j});
    fprintf('R-squared: %.10f\n', R2);
    fprintf('Adjusted R-squared: %.10f\n', adjustedR2);
    fprintf('Training RMSE: %.8g\n', RMSE);
    fprintf('Training MAE: %.8g\n', MAE);
    fprintf('Maximum absolute error: %.8g\n', maxError);
    fprintf('LOOCV RMSE: %.8g\n', CV_RMSE);
    fprintf('LOOCV Q2: %.10f\n', Q2);

    %% SUMMARY TABLE

    newRow = table( ...
        string(outputNames{j}), ...
        n, modelRank, residualDF, ...
        conditionNumber, R2, adjustedR2, ...
        RMSE, MAE, maxError, ...
        PRESS, CV_RMSE, Q2, ...
        'VariableNames', {
        'Output','Configurations','ModelRank','ResidualDF', ...
        'ConditionNumber','R2','AdjustedR2', ...
        'Training_RMSE','Training_MAE','Maximum_Absolute_Error', ...
        'PRESS','LOOCV_RMSE','LOOCV_Q2'});

    Summary = [Summary; newRow];

    %% PREDICTION TABLE

    predictionTable = table( ...
        (1:n)', y, predicted, residuals, ...
        abs(residuals), leverage, ...
        'VariableNames', {
        'Configuration','Actual','Predicted', ...
        'Residual','AbsoluteResidual','Leverage'});

    predictionTable.Output = repmat( ...
        string(outputNames{j}),n,1);

    PredictionResults = [PredictionResults; predictionTable];

    %% 8. PREDICTED VS ACTUAL PLOT

    figure('Name', [outputNames{j} ' Prediction']);

    scatter(y,predicted,35,'filled');
    hold on;

    lower = min([y;predicted]);
    upper = max([y;predicted]);

    plot([lower upper],[lower upper],'k--','LineWidth',1.5);

    xlabel('Calculated simulation output');
    ylabel('Model-predicted output');
    title([outputNames{j} ' - Predicted vs Actual']);
    grid on;
    axis tight;

    saveas(gcf,fullfile(resultFolder, ...
        ['Stage11C_' outputNames{j} '_Predicted_vs_Actual.png']));

    %% 9. RESIDUAL HISTOGRAM

    figure('Name', [outputNames{j} ' Residual Histogram']);

    histogram(residuals,20);
    xlabel('Residual');
    ylabel('Frequency');
    title([outputNames{j} ' - Residual Distribution']);
    grid on;

    saveas(gcf,fullfile(resultFolder, ...
        ['Stage11C_' outputNames{j} '_Residual_Histogram.png']));

    %% 10. RESIDUALS AGAINST FOUR FACTORS

    figure('Name', [outputNames{j} ' Residual Diagnostics']);

    tiledlayout(2,2);

    for i = 1:4
        nexttile;

        scatter(F{i},residuals,28,'filled');
        hold on;
        yline(0,'k--');

        xlabel(factorNames{i});
        ylabel('Residual');

        title([factorNames{i} ' vs Residual']);
        grid on;
    end

    sgtitle([outputNames{j} ' - Residual vs Geometry Factors']);

    saveas(gcf,fullfile(resultFolder, ...
        ['Stage11C_' outputNames{j} '_Residual_vs_Factors.png']));

end

%% 11. RESIDUAL SUMMARY BY FACTOR LEVEL

for j = 1:2

    residuals = residualsAll(:,j);

    Factor = strings(0,1);
    Level = [];
    Count = [];
    MeanResidual = [];
    ResidualRMSE = [];

    for i = 1:4

        levels = factorLevels{i};

        for k = 1:numel(levels)

            mask = F{i} == levels(k);
            r = residuals(mask);

            Factor(end+1,1) = string(factorNames{i});
            Level(end+1,1) = levels(k);
            Count(end+1,1) = sum(mask);
            MeanResidual(end+1,1) = mean(r);
            ResidualRMSE(end+1,1) = sqrt(mean(r.^2));

        end
    end

    FactorResidualTable = table( ...
        Factor,Level,Count,MeanResidual,ResidualRMSE);

    writetable(FactorResidualTable,fullfile(resultFolder, ...
        ['Stage11C_' outputNames{j} '_Factor_Residual_Summary.csv']));

end

%% 12. SAVE ALL RESULTS

writetable(Summary,fullfile(resultFolder, ...
    'Stage11C_Model_Adequacy_Summary.csv'));

writetable(PredictionResults,fullfile(resultFolder, ...
    'Stage11C_Prediction_Residual_Results.csv'));

save(fullfile(resultFolder,'Stage11C_Complete.mat'), ...
    'T','X','Y','betaAll','predictedAll', ...
    'residualsAll','Summary','PredictionResults', ...
    'factorNames','factorLevels','interactionNames', ...
    'interactionColumns','modelRank','residualDF', ...
    'conditionNumber','leverageAll');

%% 13. FINAL STATUS

fprintf('\n===== STAGE 11C COMPLETED =====\n');

disp(Summary);

fprintf('\nSaved results to:\n%s\n',resultFolder);

fprintf('\nReview the LOOCV RMSE and residual plots before interpreting model robustness.\n');

fprintf('\n===== END OF STAGE 11C =====\n');
