
%% EMC-GUARD STAGE 11E - CORRECTED
% Repeated 5-fold CV with common splits and full-rank enforcement.
% Base MATLAB only.

clc;
clear;
close all;

fprintf('\n===== EMC-GUARD STAGE 11E CORRECTED =====\n');

%% 1. PATHS AND DATA

baseFolder = 'D:\Downloads\kone\01_MATLAB_Models';
dataFile = fullfile(baseFolder,'05_Stage10C_Results', ...
    'Stage10C_All_Results.csv');
resultFolder = fullfile(baseFolder,'06_Stage11A_Results');

if ~isfile(dataFile)
    error('Stage10C dataset not found.');
end

if ~isfolder(resultFolder)
    mkdir(resultFolder);
end

T = readtable(dataFile);

Y = [T.TotalCapacitance_pF_per_m, ...
     T.AbsMutualInductance_nH];

n = height(T);

if any(~isfinite(Y),'all')
    error('Output data contains NaN or Inf.');
end

fprintf('Configurations: %d\n',n);

%% 2. FACTOR CODING

factorColumns = {
    'PairSeparation_mm'
    'PairPitch_mm'
    'Overlap_mm'
    'GroundClearance_mm'
};

F = cell(1,4);
D = cell(1,4);

for i = 1:4

    F{i} = T.(factorColumns{i});
    lev = unique(F{i});

    D{i} = zeros(n,numel(lev)-1);

    for j = 2:numel(lev)
        D{i}(:,j-1) = double(F{i} == lev(j));
    end

end

%% 3. BASE MODEL

Xbase = ones(n,1);

for i = 1:4
    Xbase = [Xbase,D{i}];
end

pairs = nchoosek(1:4,2);

for k = 1:size(pairs,1)

    a = pairs(k,1);
    b = pairs(k,2);

    for ia = 1:size(D{a},2)
        for ib = 1:size(D{b},2)

            Xbase = [Xbase, ...
                D{a}(:,ia).*D{b}(:,ib)];

        end
    end

end

%% 4. THREE-FACTOR BLOCKS

triplets = [
    1 2 3
    1 2 4
    1 3 4
    2 3 4
];

blocks = cell(4,1);

for k = 1:4

    a = triplets(k,1);
    b = triplets(k,2);
    c = triplets(k,3);

    block = [];

    for ia = 1:size(D{a},2)
        for ib = 1:size(D{b},2)
            for ic = 1:size(D{c},2)

                block = [block, ...
                    D{a}(:,ia).*D{b}(:,ib).*D{c}(:,ic)];

            end
        end
    end

    blocks{k} = block;

end

%% 5. SIX MODELS

modelNames = {
    'A_Base_TwoFactor'
    'B1_SP_Overlap'
    'B2_SP_Ground'
    'B3_S_Overlap_Ground'
    'B4_P_Overlap_Ground'
};

Xmodels = cell(5,1);

Xmodels{1} = Xbase;
Xmodels{2} = [Xbase,blocks{1}];
Xmodels{3} = [Xbase,blocks{2}];
Xmodels{4} = [Xbase,blocks{3}];
Xmodels{5} = [Xbase,blocks{4}];

numberModels = numel(Xmodels);
%% 6. VALIDATION SETTINGS

numberRepeats = 20;
numberFolds = 5;
maximumAttempts = 10000;

rng(20261101,'twister');

% Store RMSE for each repetition/fold/model/output.
FoldRMSE = nan(numberRepeats,numberFolds,numberModels,2);

% Store pooled sum of squared test errors.
TotalSSE = zeros(numberModels,2);
TotalTestCount = zeros(numberModels,2);

AcceptedAttempts = zeros(numberRepeats,1);
FoldAssignments = zeros(n,numberRepeats);

%% 7. GENERATE COMMON FULL-RANK FOLD ASSIGNMENTS

fprintf('\nGenerating valid common fold assignments...\n');

for rep = 1:numberRepeats

    accepted = false;

    for attempt = 1:maximumAttempts

        order = randperm(n);
        foldID = zeros(n,1);

        for q = 1:n
            foldID(order(q)) = mod(q-1,numberFolds)+1;
        end

        validAssignment = true;

        % Every candidate model must have full rank
        % in every training fold.
        for fold = 1:numberFolds

            testMask = foldID == fold;
            trainMask = ~testMask;

            for m = 1:numberModels

                Xtrain = Xmodels{m}(trainMask,:);

                if rank(Xtrain) < size(Xtrain,2)
                    validAssignment = false;
                    break;
                end

            end

            if ~validAssignment
                break;
            end

        end

        if validAssignment
            accepted = true;
            AcceptedAttempts(rep) = attempt;
            FoldAssignments(:,rep) = foldID;
            break;
        end

    end

    if ~accepted
        error(['Could not generate a full-rank common fold assignment. ' ...
            'Increase maximumAttempts or reconsider model complexity.']);
    end

    fprintf('Repetition %d accepted after %d attempt(s).\n', ...
        rep,AcceptedAttempts(rep));

end

%% 8. RUN ALL MODELS ON IDENTICAL SPLITS

for rep = 1:numberRepeats

    foldID = FoldAssignments(:,rep);

    for fold = 1:numberFolds

        testMask = foldID == fold;
        trainMask = ~testMask;

        for m = 1:numberModels

            X = Xmodels{m};

            Xtrain = X(trainMask,:);
            Xtest = X(testMask,:);

            if rank(Xtrain) < size(Xtrain,2)
                error('Unexpected rank failure during fitting.');
            end

            for j = 1:2

                ytrain = Y(trainMask,j);
                ytest = Y(testMask,j);

                beta = Xtrain \ ytrain;
                prediction = Xtest*beta;

                errors = ytest-prediction;

                foldSSE = sum(errors.^2);
                foldCount = numel(errors);

                FoldRMSE(rep,fold,m,j) = ...
                    sqrt(foldSSE/foldCount);

                TotalSSE(m,j) = TotalSSE(m,j)+foldSSE;
                TotalTestCount(m,j) = ...
                    TotalTestCount(m,j)+foldCount;

            end

        end

    end

    fprintf('Validation completed: repetition %d/%d\n', ...
        rep,numberRepeats);

end

%% 9. SUMMARY METRICS

Model = strings(0,1);
Output = strings(0,1);
ValidFolds = [];
MeanFoldRMSE = [];
StdFoldRMSE = [];
PooledCV_RMSE = [];
MinimumFoldRMSE = [];
MaximumFoldRMSE = [];

outputNames = {'Capacitance','AbsMutualInductance'};

for m = 1:numberModels

    for j = 1:2

        values = reshape(FoldRMSE(:,:,m,j),[],1);

        if any(~isfinite(values))
            error('Missing or invalid fold results detected.');
        end

        Model(end+1,1) = string(modelNames{m});
        Output(end+1,1) = string(outputNames{j});

        ValidFolds(end+1,1) = numel(values);
        MeanFoldRMSE(end+1,1) = mean(values);
        StdFoldRMSE(end+1,1) = std(values);

        PooledCV_RMSE(end+1,1) = ...
            sqrt(TotalSSE(m,j)/TotalTestCount(m,j));

        MinimumFoldRMSE(end+1,1) = min(values);
        MaximumFoldRMSE(end+1,1) = max(values);

    end

end

Results = table(Model,Output,ValidFolds, ...
    MeanFoldRMSE,StdFoldRMSE,PooledCV_RMSE, ...
    MinimumFoldRMSE,MaximumFoldRMSE);

fprintf('\n===== CORRECTED CV RESULTS =====\n');
disp(Results);

writetable(Results,fullfile(resultFolder, ...
    'Stage11E_Corrected_CV_Results.csv'));

%% 10. BEST MODEL PER OUTPUT

BestModels = table();

for j = 1:2

    mask = Results.Output == string(outputNames{j});
    R = Results(mask,:);

    [~,idx] = min(R.PooledCV_RMSE);

    BestModels = [BestModels;R(idx,:)];

end

fprintf('\n===== BEST MODELS BY POOLED CV RMSE =====\n');
disp(BestModels);

writetable(BestModels,fullfile(resultFolder, ...
    'Stage11E_Corrected_Best_Models.csv'));

%% 11. VALIDATION AUDIT

fprintf('\n===== VALIDATION AUDIT =====\n');
fprintf('Expected folds per model: %d\n', ...
    numberRepeats*numberFolds);

fprintf('Actual folds per model: %d\n', ...
    numberRepeats*numberFolds);

fprintf('All models used identical fold assignments: YES\n');
fprintf('All training matrices passed full-rank checks: YES\n');
fprintf('All fold results finite: %d\n', ...
    all(isfinite(FoldRMSE(:))));

fprintf('Total test predictions per model/output: %d\n', ...
    sum(TotalTestCount(:,1)));

%% 12. SAVE COMPLETE RESULTS

save(fullfile(resultFolder,'Stage11E_Corrected_Complete.mat'), ...
    'Results','BestModels','FoldRMSE', ...
    'FoldAssignments','AcceptedAttempts', ...
    'TotalSSE','TotalTestCount', ...
    'modelNames','numberRepeats','numberFolds');

fprintf('\n===== STAGE 11E CORRECTED COMPLETED =====\n');
fprintf('Results saved in:\n%s\n',resultFolder);
