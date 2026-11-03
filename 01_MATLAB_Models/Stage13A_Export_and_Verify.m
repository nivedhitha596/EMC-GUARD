
clc;
clear;
close all;

%% PATHS
basePath = 'D:\Downloads\kone\01_MATLAB_Models';

sourceFile = fullfile(basePath, ...
    '06_Stage11A_Results', 'Stage11D_Complete.mat');

outputFolder = fullfile(basePath, ...
    '09_Stage13A_Export');

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

%% LOAD ORIGINAL VERIFIED DATA
S = load(sourceFile);

assert(isfield(S, 'T'), ...
    'Source MAT file does not contain table T.');

T = S.T;

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

%% CREATE CONFIGURATION IDs
n = height(T);

T.Config_ID = compose('EMC-%03d', (1:n)');

%% SELECT AND ORDER EXPORT COLUMNS
exportColumns = [
    {'Config_ID'}
    requiredColumns
];

ExportTable = T(:, exportColumns);

%% BASIC DATA VALIDATIOn
%% BASIC DATA VALIDATION

assert(height(ExportTable) == 216, ...
    'Expected 216 configurations.');

assert(numel(unique(ExportTable.Config_ID)) == 216, ...
    'Duplicate configuration IDs found.');

% Type-safe missing-value check
missingCount = 0;

for k = 1:width(ExportTable)
    col = ExportTable.(ExportTable.Properties.VariableNames{k});
    missingCount = missingCount + sum(ismissing(col));
end

assert(missingCount == 0, ...
    'Missing values found in export.');

% Numeric target validation
assert(all(isfinite(ExportTable.TotalCapacitance_pF_per_m)), ...
    'Invalid capacitance values.');

assert(all(isfinite(ExportTable.AbsMutualInductance_nH)), ...
    'Invalid mutual inductance values.');

fprintf('\n===== DATA VALIDATION PASSED =====\n');
fprintf('Total configurations: %d\n', height(ExportTable));
fprintf('Missing values: %d\n', missingCount);
fprintf('Unique Config IDs: %d\n', ...
    numel(unique(ExportTable.Config_ID)));

%% CHECK MISSING VALUES
%% CHECK MISSING VALUES — TYPE-SAFE

missingCount = 0;

for k = 1:width(ExportTable)
    col = ExportTable.(ExportTable.Properties.VariableNames{k});

    missingCount = missingCount + sum(ismissing(col));
end

assert(missingCount == 0, ...
    'Missing values found in export.');

fprintf('Missing values: %d\n', missingCount);

%% EXPORT
csvFile = fullfile(outputFolder, ...
    'EMC_GUARD_Physics_Dataset.csv');

writetable(ExportTable, csvFile);

%% SAVE DATASET METADATA
metadata.Project = 'EMC-GUARD';
metadata.DatasetSize = height(ExportTable);
metadata.InputFeatures = {
    'PairSeparation_mm'
    'PairPitch_mm'
    'Overlap_mm'
    'GroundClearance_mm'
};
metadata.Targets = {
    'TotalCapacitance_pF_per_m'
    'AbsMutualInductance_nH'
};
metadata.DatasetSource = 'Stage11D_Complete.mat';
metadata.DatasetType = ...
    'Deterministic MATLAB-generated computational dataset';
metadata.ValidationDomain = ...
    'Original 216-point factorial design grid only';

save(fullfile(outputFolder, ...
    'MATLAB_Model_Metadata.mat'), 'metadata');

fprintf('\n===== STAGE 13A EXPORT COMPLETE =====\n');
fprintf('Configurations: %d\n', height(ExportTable));
fprintf('Dataset: %s\n', csvFile);

disp(ExportTable(1:5,:));
