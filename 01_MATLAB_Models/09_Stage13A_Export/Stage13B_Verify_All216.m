
clc;
clear;
close all;

%% STAGE 13B - ALL 216 PREDICTION VERIFICATION

basePath = 'D:\Downloads\kone\01_MATLAB_Models';

sourceFile = fullfile(basePath, ...
    '06_Stage11A_Results', 'Stage11D_Complete.mat');

outputFolder = fullfile(basePath, ...
    '09_Stage13A_Export');

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

%% LOAD REFERENCE DATA

S = load(sourceFile);

assert(isfield(S,'T'), ...
    'Reference table T not found.');

T = S.T;

n = height(T);

assert(n == 216, ...
    'Expected 216 reference configurations.');

%% REQUIRED REFERENCE COLUMNS

requiredColumns = {
    'PairSeparation_mm'
    'PairPitch_mm'
    'Overlap_mm'
    'GroundClearance_mm'
    'TotalCapacitance_pF_per_m'
    'AbsMutualInductance_nH'
};

for k = 1:numel(requiredColumns)
    assert(ismember(requiredColumns{k}, ...
        T.Properties.VariableNames), ...
        ['Missing column: ', requiredColumns{k}]);
end

%% INITIALIZE ARRAYS

predCap = zeros(n,1);
predMut = zeros(n,1);

designIndex = zeros(n,1);

capModel = cell(n,1);
mutModel = cell(n,1);

%% CLEAR PERSISTENT FUNCTION STATE

clear EMC_GUARD_Predict;

fprintf('\n===== STAGE 13B VERIFICATION STARTED =====\n');

%% RUN ALL 216 CONFIGURATIONS

for i = 1:n

    sep = T.PairSeparation_mm(i);
    pitch = T.PairPitch_mm(i);
    overlap = T.Overlap_mm(i);
    ground = T.GroundClearance_mm(i);

    result = EMC_GUARD_Predict( ...
        sep, pitch, overlap, ground);

    predCap(i) = result.Capacitance_pF_per_m;

    predMut(i) = result.AbsMutualInductance_nH;

    designIndex(i) = result.DesignIndex;

    capModel{i} = result.CapacitanceModel;

    mutModel{i} = result.MutualInductanceModel;

    if mod(i,25) == 0 || i == n
        fprintf('Verified %d / %d configurations\n', i, n);
    end

end

%% REFERENCE VALUES

refCap = T.TotalCapacitance_pF_per_m;

refMut = T.AbsMutualInductance_nH;

%% CALCULATE ERRORS

capError = predCap - refCap;

mutError = predMut - refMut;

absCapError = abs(capError);

absMutError = abs(mutError);

%% ERROR METRICS

capRMSE = sqrt(mean(capError.^2));

capMAE = mean(absCapError);

capMaxError = max(absCapError);

mutRMSE = sqrt(mean(mutError.^2));

mutMAE = mean(absMutError);

mutMaxError = max(absMutError);

%% DESIGN INDEX CHECK

indexMatches = (designIndex == (1:n)');

indexMismatchCount = sum(~indexMatches);

%% VERIFICATION TOLERANCES

% Numerical reconstruction tolerances.
capTolerance = 1e-9;
mutTolerance = 1e-10;

capPass = absCapError <= capTolerance;

mutPass = absMutError <= mutTolerance;

overallPass = capPass & mutPass & indexMatches;

passCount = sum(overallPass);

failCount = sum(~overallPass);

%% BUILD VERIFICATION TABLE

if ismember('Config_ID', T.Properties.VariableNames)
    configID = string(T.Config_ID);
else
    configID = compose('EMC-%03d',(1:n)');
end

VerificationTable = table( ...
    configID, ...
    T.PairSeparation_mm, ...
    T.PairPitch_mm, ...
    T.Overlap_mm, ...
    T.GroundClearance_mm, ...
    refCap, ...
    predCap, ...
    absCapError, ...
    refMut, ...
    predMut, ...
    absMutError, ...
    designIndex, ...
    indexMatches, ...
    overallPass, ...
    'VariableNames', { ...
    'Config_ID', ...
    'Separation_mm', ...
    'Pitch_mm', ...
    'Overlap_mm', ...
    'GroundClearance_mm', ...
    'Reference_C_pF_per_m', ...
    'Predicted_C_pF_per_m', ...
    'Absolute_C_Error', ...
    'Reference_AbsM_nH', ...
    'Predicted_AbsM_nH', ...
    'Absolute_M_Error', ...
    'DesignIndex', ...
    'DesignIndex_Match', ...
    'Overall_Pass'});

%% EXPORT VERIFICATION RESULTS

verificationFile = fullfile(outputFolder, ...
    'MATLAB_216_Verification.csv');

writetable(VerificationTable, verificationFile);

%% EXPORT SUMMARY

summaryTable = table( ...
    ["Capacitance"; "Absolute Mutual Inductance"], ...
    [capRMSE; mutRMSE], ...
    [capMAE; mutMAE], ...
    [capMaxError; mutMaxError], ...
    [capTolerance; mutTolerance], ...
    'VariableNames', { ...
    'Response', ...
    'RMSE', ...
    'MAE', ...
    'MaximumAbsoluteError', ...
    'Tolerance'});

summaryFile = fullfile(outputFolder, ...
    'MATLAB_216_Verification_Summary.csv');

writetable(summaryTable, summaryFile);

%% DISPLAY RESULTS

fprintf('\n===== STAGE 13B SUMMARY =====\n');

fprintf('Total configurations: %d\n', n);

fprintf('Passed configurations: %d\n', passCount);

fprintf('Failed configurations: %d\n', failCount);

fprintf('Design index mismatches: %d\n', ...
    indexMismatchCount);

fprintf('\nCAPACITANCE\n');
fprintf('RMSE: %.12e pF/m\n', capRMSE);
fprintf('MAE: %.12e pF/m\n', capMAE);
fprintf('Maximum error: %.12e pF/m\n', capMaxError);

fprintf('\nABSOLUTE MUTUAL INDUCTANCE\n');
fprintf('RMSE: %.12e nH\n', mutRMSE);
fprintf('MAE: %.12e nH\n', mutMAE);
fprintf('Maximum error: %.12e nH\n', mutMaxError);

fprintf('\nVerification CSV: %s\n', verificationFile);
fprintf('Summary CSV: %s\n', summaryFile);

%% SHOW FAILED CONFIGURATIONS

if failCount > 0

    fprintf('\n===== FAILED CONFIGURATIONS =====\n');

    disp(VerificationTable(~overallPass,:));

else

    fprintf('\n===== ALL 216 CONFIGURATIONS PASSED =====\n');

end
%% ADD MATLAB PROJECT FOLDER TO PATH

basePath = 'D:\Downloads\kone\01_MATLAB_Models';

addpath(basePath);

assert(exist('EMC_GUARD_Predict','file') == 2, ...
    'EMC_GUARD_Predict.m not found. Check project folder.');
fprintf('\n===== STAGE 13B COMPLETE =====\n');
