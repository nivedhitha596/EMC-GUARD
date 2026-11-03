%% EMC-GUARD STAGE 12A - SELECTED SURROGATE MODEL FREEZE
% Freezes the response-specific models selected by corrected Stage 11E:
%   Capacitance            -> B2 (Separation x Pitch x Ground)
%   Abs mutual inductance  -> B1 (Separation x Pitch x Overlap)
%
% IMPORTANT:
% This script fits coefficients to the deterministic Stage 11D simulation
% dataset. It does not establish experimental accuracy or EMC compliance.
%
% Before running:
% 1) Put Stage11D_Complete.mat in the Stage11A results folder below.
% 2) Edit projectRoot if your project is stored elsewhere.
% 3) Run this script in MATLAB.

clear; clc; close all;

%% PATHS
projectRoot = 'D:\Downloads\kone\01_MATLAB_Models';
resultsRoot = fullfile(projectRoot, '06_Stage11A_Results');
inputFile = fullfile(resultsRoot, 'Stage11D_Complete.mat');
outputFolder = fullfile(resultsRoot, '07_Stage12A_ModelFreeze');

if ~isfile(inputFile)
    error(['Cannot find Stage11D_Complete.mat: ' inputFile newline ...
        'Copy the uploaded Stage11D_Complete.mat into this folder and rerun.']);
end
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

fprintf('\n===== EMC-GUARD STAGE 12A MODEL FREEZE =====\n');
fprintf('Input: %s\n', inputFile);

%% LOAD AND CHECK SOURCE DATA
S = load(inputFile);

requiredVars = {'modelMatrices','modelNames','Y'};
for k = 1:numel(requiredVars)
    if ~isfield(S, requiredVars{k})
        error('Missing required variable "%s" in Stage11D MAT file.', requiredVars{k});
    end
end

modelNames = string(S.modelNames(:));
Y = double(S.Y);

if size(Y,1) ~= 216 || size(Y,2) < 2
    error('Unexpected response matrix dimensions. Expected 216 x at least 2.');
end
if numel(S.modelMatrices) < 5
    error('Expected at least five model matrices in Stage11D file.');
end

% Match by model name rather than assuming a particular cell-array order.
idxB1 = find(contains(modelNames, "B1"), 1, 'first');
idxB2 = find(contains(modelNames, "B2"), 1, 'first');
if isempty(idxB1) || isempty(idxB2)
    error('Could not identify B1 and B2 in modelNames.');
end

XB1 = double(S.modelMatrices{idxB1});
XB2 = double(S.modelMatrices{idxB2});

if size(XB1,1) ~= size(Y,1) || size(XB2,1) ~= size(Y,1)
    error('Model matrix row count does not match response data.');
end

%% SELECT RESPONSE-SPECIFIC MODELS
% Y(:,1) = Total capacitance (pF/m)
% Y(:,2) = Absolute mutual inductance (nH)
Xcap = XB2;
Xmut = XB1;
yCap = Y(:,1);
yMut = Y(:,2);

%% FIT LEAST-SQUARES COEFFICIENTS
if rank(Xcap) < size(Xcap,2)
    error('B2 capacitance matrix is rank deficient.');
end
if rank(Xmut) < size(Xmut,2)
    error('B1 mutual-inductance matrix is rank deficient.');
end

betaCap = Xcap \ yCap;
betaMut = Xmut \ yMut;

predCap = Xcap * betaCap;
predMut = Xmut * betaMut;

resCap = yCap - predCap;
resMut = yMut - predMut;

rmseCap = sqrt(mean(resCap.^2));
rmseMut = sqrt(mean(resMut.^2));
maeCap = mean(abs(resCap));
maeMut = mean(abs(resMut));
maxAbsCap = max(abs(resCap));
maxAbsMut = max(abs(resMut));

R2Cap = 1 - sum(resCap.^2) / sum((yCap - mean(yCap)).^2);
R2Mut = 1 - sum(resMut.^2) / sum((yMut - mean(yMut)).^2);

%% RECORD CROSS-VALIDATION METRICS FROM THE SUCCESSFUL STAGE 11E CONSOLE LOG
% These are copied from the user's successful 20 x 5-fold run.
% They are reported as CV metrics, not recalculated from the stale MAT file.
cvCapRMSE = 5.4606e-14;  % B2, pooled CV RMSE, pF/m
cvMutRMSE = 1.0659e-14; % B1, pooled CV RMSE, nH
cvFoldsPerModel = 100;
cvRepeats = 20;
cvFoldsPerRepeat = 5;

%% BUILD SUMMARY TABLE
Response = ["Total capacitance"; "Absolute mutual inductance"];
SelectedModel = ["B2_SP_Ground"; "B1_SP_Overlap"];
Units = ["pF/m"; "nH"];
NumTerms = [size(Xcap,2); size(Xmut,2)];
MatrixRank = [rank(Xcap); rank(Xmut)];
ConditionNumber = [cond(Xcap); cond(Xmut)];
TrainingRMSE = [rmseCap; rmseMut];
TrainingMAE = [maeCap; maeMut];
MaximumAbsoluteTrainingError = [maxAbsCap; maxAbsMut];
TrainingR2 = [R2Cap; R2Mut];
PooledCV_RMSE_from_Stage11E = [cvCapRMSE; cvMutRMSE];
ValidCVFolds = [cvFoldsPerModel; cvFoldsPerModel];

summaryTable = table(Response, SelectedModel, Units, NumTerms, MatrixRank, ...
    ConditionNumber, TrainingRMSE, TrainingMAE, ...
    MaximumAbsoluteTrainingError, TrainingR2, ...
    PooledCV_RMSE_from_Stage11E, ValidCVFolds);

writetable(summaryTable, fullfile(outputFolder, ...
    'Stage12A_Selected_Model_Summary.csv'));

%% SAVE COEFFICIENTS AND REPRODUCIBILITY DATA
modelFreeze = struct();
modelFreeze.project = 'EMC-GUARD';
modelFreeze.stage = '12A - Selected Surrogate Model Freeze';
modelFreeze.sourceFile = inputFile;
modelFreeze.selectedModels.capacitance.name = 'B2_SP_Ground';
modelFreeze.selectedModels.capacitance.description = ...
    'Base two-factor model plus Separation x Pitch x Ground interaction';
modelFreeze.selectedModels.capacitance.units = 'pF/m';
modelFreeze.selectedModels.capacitance.X = Xcap;
modelFreeze.selectedModels.capacitance.beta = betaCap;
modelFreeze.selectedModels.capacitance.trainingPrediction = predCap;
modelFreeze.selectedModels.capacitance.trainingResidual = resCap;
modelFreeze.selectedModels.mutualInductance.name = 'B1_SP_Overlap';
modelFreeze.selectedModels.mutualInductance.description = ...
    'Base two-factor model plus Separation x Pitch x Overlap interaction';
modelFreeze.selectedModels.mutualInductance.units = 'nH';
modelFreeze.selectedModels.mutualInductance.X = Xmut;
modelFreeze.selectedModels.mutualInductance.beta = betaMut;
modelFreeze.selectedModels.mutualInductance.trainingPrediction = predMut;
modelFreeze.selectedModels.mutualInductance.trainingResidual = resMut;
modelFreeze.responses = Y(:,1:2);
modelFreeze.cvEvidence = struct( ...
    'repeats', cvRepeats, ...
    'foldsPerRepeat', cvFoldsPerRepeat, ...
    'validFoldsPerModel', cvFoldsPerModel, ...
    'commonFoldAssignments', true, ...
    'allTrainingMatricesFullRank', true, ...
    'capacitanceB2PooledRMSE_pF_per_m', cvCapRMSE, ...
    'mutualInductanceB1PooledRMSE_nH', cvMutRMSE, ...
    'source', 'Successful corrected Stage 11E console output supplied by user');
modelFreeze.caveat = ['Cross-validation is internal to the deterministic simulated ' ...
    '216-configuration design. No physical cable measurement or EMC compliance ' ...
    'claim is established by this model freeze.'];

save(fullfile(outputFolder, 'Stage12A_Selected_Models_FROZEN.mat'), ...
    'modelFreeze', 'betaCap', 'betaMut', 'Xcap', 'Xmut', ...
    'yCap', 'yMut', 'summaryTable', '-v7.3');

%% EXPORT PREDICTIONS AND RESIDUALS
Configuration = (1:size(Y,1))';
predictionTable = table(Configuration, yCap, predCap, resCap, ...
    yMut, predMut, resMut, ...
    'VariableNames', {'Configuration','ActualCapacitance_pF_m', ...
    'PredictedCapacitance_pF_m','CapacitanceResidual_pF_m', ...
    'ActualAbsMutualInductance_nH','PredictedAbsMutualInductance_nH', ...
    'AbsMutualInductanceResidual_nH'});
writetable(predictionTable, fullfile(outputFolder, ...
    'Stage12A_Selected_Model_Predictions.csv'));

%% PLOTS
fig = figure('Color','w','Name','EMC-GUARD Stage 12A Model Freeze');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

nexttile;
scatter(yCap, predCap, 28, 'filled'); hold on;
lims = [min(yCap), max(yCap)];
plot(lims, lims, 'k--','LineWidth',1.2);
xlabel('Actual capacitance (pF/m)');
ylabel('Predicted capacitance (pF/m)');
title('B2 capacitance model'); grid on; axis tight;

nexttile;
scatter(yMut, predMut, 28, 'filled'); hold on;
lims = [min(yMut), max(yMut)];
plot(lims, lims, 'k--','LineWidth',1.2);
xlabel('Actual |M| (nH)');
ylabel('Predicted |M| (nH)');
title('B1 mutual-inductance model'); grid on; axis tight;

nexttile;
plot(Configuration, resCap, '.-');
yline(0,'k--'); xlabel('Configuration');
ylabel('Residual (pF/m)');
title('B2 capacitance residual'); grid on;

nexttile;
plot(Configuration, resMut, '.-');
yline(0,'k--'); xlabel('Configuration');
ylabel('Residual (nH)');
title('B1 mutual-inductance residual'); grid on;

exportgraphics(fig, fullfile(outputFolder, ...
    'Stage12A_Selected_Model_Diagnostics.png'), 'Resolution', 200);
savefig(fig, fullfile(outputFolder, ...
    'Stage12A_Selected_Model_Diagnostics.fig'));

%% FINAL AUDIT
fprintf('\n===== STAGE 12A SUMMARY =====\n');
disp(summaryTable);
fprintf('B2 capacitance coefficient count: %d\n', numel(betaCap));
fprintf('B1 mutual-inductance coefficient count: %d\n', numel(betaMut));
fprintf('Corrected Stage 11E CV folds/model: %d\n', cvFoldsPerModel);
fprintf('Outputs saved to:\n%s\n', outputFolder);
fprintf(['NOTE: Stage 11E MAT file previously uploaded did not match the ' ...
    'successful five-model console run. CV metrics above are therefore ' ...
    'recorded from the console log, not extracted from that MAT file.\n']);
fprintf('===== STAGE 12A MODEL FREEZE COMPLETED =====\n');
