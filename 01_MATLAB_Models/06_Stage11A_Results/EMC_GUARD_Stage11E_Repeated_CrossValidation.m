
%% EMC-GUARD STAGE 11E
% Repeated 5-Fold Cross-Validation
% Compares the six Stage 11D candidate models.

clc;
clear;
close all;

fprintf('\n===== EMC-GUARD STAGE 11E =====\n');
fprintf('REPEATED 5-FOLD CROSS-VALIDATION\n\n');

%% 1. PATHS

baseFolder = 'D:\Downloads\kone\01_MATLAB_Models';
dataFolder = fullfile(baseFolder,'05_Stage10C_Results');
resultFolder = fullfile(baseFolder,'06_Stage11A_Results');

dataFile = fullfile(dataFolder,'Stage10C_All_Results.csv');

if ~isfile(dataFile)
    error('Stage10C dataset not found.');
end

%% 2. LOAD DATA

T = readtable(dataFile);

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

outputNames = {'Capacitance','AbsMutualInductance'};

n = height(T);

if n ~= 216
    warning('Expected 216 rows; found %d.',n);
end

%% 3. FACTOR CODING

F = cell(1,4);
levels = cell(1,4);
D = cell(1,4);

for i = 1:4

    F{i} = T.(factorColumns{i});
    levels{i} = unique(F{i});

    D{i} = zeros(n,numel(levels{i})-1);

    for j = 2:numel(levels{i})
        D{i}(:,j-1) = double(F{i} == levels{i}(j));
    end

end

%% 4. BUILD BASE MODEL

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

%% 5. BUILD THREE-FACTOR BLOCKS

triplets = [
    1 2 3
    1 2 4
    1 3 4
    2 3 4
];

threeWayBlocks = cell(4,1);

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

    threeWayBlocks{k} = block;

end

%% 6. CANDIDATE MODELS

modelNames = {
    'A_Base_TwoFactor'
    'B1_SP_Overlap'
    'B2_SP_Ground'
    'B3_S_Overlap_Ground'
    'B4_P_Overlap_Ground'
    'C_All_ThreeFactor'
};

Xmodels = cell(6,1);

Xmodels{1} = Xbase;
Xmodels{2} = [Xbase,threeWayBlocks{1}];
Xmodels{3} = [Xbase,threeWayBlocks{2}];
Xmodels{4} = [Xbase,threeWayBlocks{3}];
Xmodels{5} = [Xbase,threeWayBlocks{4}];

Xmodels{6} = [
    Xbase,threeWayBlocks{1}, ...
    threeWayBlocks{2},threeWayBlocks{3}, ...
    threeWayBlocks{4}
];

%% 7. CROSS-VALIDATION SETTINGS

numberRepeats = 20;
numberFolds = 5;

rng(20261101,'twister');

% Store one RMSE per repetition, fold, model and output.
CV_RMSE = nan(numberRepeats,numberFolds,6,2);

%% 8. REPEATED STRATIFIED RANDOM FOLD ASSIGNMENT

fprintf('Repeats: %d\n',numberRepeats);
fprintf('Folds per repeat: %d\n',numberFolds);

for rep = 1:numberRepeats

    % Randomly distribute configurations among five folds.
    order = randperm(n);

    foldID = zeros(n,1);

    for k = 1:n
        foldID(order(k)) = mod(k-1,numberFolds)+1;
    end

    for fold = 1:numberFolds

        testMask = foldID == fold;
        trainMask = ~testMask;

        for m = 1:6

            X = Xmodels{m};

            Xtrain = X(trainMask,:);
            Xtest = X(testMask,:);

            if rank(Xtrain) < size(Xtrain,2)
                warning('Rank deficiency: repeat %d fold %d model %s', ...
                    rep,fold,modelNames{m});
                continue;
            end

            for j = 1:2

                ytrain = Y(trainMask,j);
                ytest = Y(testMask,j);

                beta = Xtrain \ ytrain;
                prediction = Xtest*beta;

                errorVector = ytest-prediction;

                CV_RMSE(rep,fold,m,j) = ...
                    sqrt(mean(errorVector.^2));

            end

        end

    end

    fprintf('Completed repetition %d/%d\n',rep,numberRepeats);

end

%% 9. SUMMARISE RESULTS

Model = strings(0,1);
Output = strings(0,1);
Mean_RMSE = [];
Std_RMSE = [];
Minimum_RMSE = [];
Maximum_RMSE = [];

for m = 1:6
    for j = 1:2

        values = reshape(CV_RMSE(:,:,m,j),[],1);
        values = values(isfinite(values));

        Model(end+1,1) = string(modelNames{m});
        Output(end+1,1) = string(outputNames{j});

        Mean_RMSE(end+1,1) = mean(values);
        Std_RMSE(end+1,1) = std(values);
        Minimum_RMSE(end+1,1) = min(values);
        Maximum_RMSE(end+1,1) = max(values);

    end
end

Results = table( ...
    Model,Output,Mean_RMSE,Std_RMSE, ...
    Minimum_RMSE,Maximum_RMSE);

disp(Results);

writetable(Results,fullfile(resultFolder, ...
    'Stage11E_Repeated_CV_Results.csv'));

%% 10. SELECT BEST MODEL PER OUTPUT

BestModels = table();

for j = 1:2

    mask = Results.Output == string(outputNames{j});
    R = Results(mask,:);

    [~,idx] = min(R.Mean_RMSE);

    BestModels = [BestModels;R(idx,:)];

end

fprintf('\n===== BEST REPEATED-CV MODELS =====\n');
disp(BestModels);

writetable(BestModels,fullfile(resultFolder, ...
    'Stage11E_Best_Models.csv'));

%% 11. SAVE COMPLETE DATA

save(fullfile(resultFolder,'Stage11E_Complete.mat'), ...
    'CV_RMSE','Results','BestModels', ...
    'modelNames','numberRepeats','numberFolds');

fprintf('\n===== STAGE 11E COMPLETED =====\n');
fprintf('Results saved to:\n%s\n',resultFolder);
