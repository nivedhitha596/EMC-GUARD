
%% EMC-GUARD STAGE 11D
% Three-factor interaction refinement and model comparison
% Base MATLAB only. No additional toolboxes required.

clc;
clear;
close all;

fprintf('\n===== EMC-GUARD STAGE 11D =====\n');
fprintf('THREE-FACTOR INTERACTION REFINEMENT\n\n');

%% 1. PATHS

baseFolder = 'D:\Downloads\kone\01_MATLAB_Models';
dataFolder = fullfile(baseFolder,'05_Stage10C_Results');
resultFolder = fullfile(baseFolder,'06_Stage11A_Results');

dataFile = fullfile(dataFolder,'Stage10C_All_Results.csv');

if ~isfile(dataFile)
    error('Stage10C data file not found.');
end

if ~isfolder(resultFolder)
    mkdir(resultFolder);
end

%% 2. LOAD DATA

T = readtable(dataFile);

factorColumns = {
    'PairSeparation_mm'
    'PairPitch_mm'
    'Overlap_mm'
    'GroundClearance_mm'
};

factorNames = {
    'Separation'
    'Pitch'
    'Overlap'
    'GroundClearance'
};

outputNames = {
    'Capacitance'
    'AbsMutualInductance'
};

Y = [
    T.TotalCapacitance_pF_per_m, ...
    T.AbsMutualInductance_nH
];

n = height(T);

if n ~= 216
    warning('Expected 216 configurations; found %d.',n);
end

if any(~isfinite(Y),'all')
    error('Non-finite output values detected.');
end

fprintf('Configurations: %d\n',n);

%% 3. FACTOR DUMMY MATRICES

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

    fprintf('%s: %d levels\n', ...
        factorNames{i},numel(levels{i}));

end

%% 4. BUILD BASE MODEL: MAIN + TWO-FACTOR INTERACTIONS

Xbase = ones(n,1);

for i = 1:4
    Xbase = [Xbase,D{i}];
end

pairs = nchoosek(1:4,2);

for k = 1:size(pairs,1)

    a = pairs(k,1);
    b = pairs(k,2);

    A = D{a};
    B = D{b};

    for ia = 1:size(A,2)
        for ib = 1:size(B,2)
            Xbase = [Xbase,A(:,ia).*B(:,ib)];
        end
    end

end

fprintf('\nBase model columns: %d\n',size(Xbase,2));

%% 5. BUILD FOUR THREE-FACTOR BLOCKS

triplets = [
    1 2 3
    1 2 4
    1 3 4
    2 3 4
];

threeWayNames = {
    'Separation_x_Pitch_x_Overlap'
    'Separation_x_Pitch_x_Ground'
    'Separation_x_Overlap_x_Ground'
    'Pitch_x_Overlap_x_Ground'
};

threeWayBlocks = cell(4,1);

for k = 1:4

    a = triplets(k,1);
    b = triplets(k,2);
    c = triplets(k,3);

    A = D{a};
    B = D{b};
    C = D{c};

    block = [];

    for ia = 1:size(A,2)
        for ib = 1:size(B,2)
            for ic = 1:size(C,2)

                term = A(:,ia).*B(:,ib).*C(:,ic);
                block = [block,term];

            end
        end
    end

    threeWayBlocks{k} = block;

    fprintf('%s: %d columns\n', ...
        threeWayNames{k},size(block,2));

end

%% 6. DEFINE SIX CANDIDATE MODELS

modelNames = {
    'A_Base_TwoFactor'
    'B1_Add_SP_Overlap'
    'B2_Add_SP_Ground'
    'B3_Add_S_Overlap_Ground'
    'B4_Add_P_Overlap_Ground'
    'C_All_ThreeFactor'
};

modelMatrices = cell(6,1);

modelMatrices{1} = Xbase;

for k = 1:4
    modelMatrices{k+1} = [Xbase,threeWayBlocks{k}];
end

modelMatrices{6} = [
    Xbase, ...
    threeWayBlocks{1}, ...
    threeWayBlocks{2}, ...
    threeWayBlocks{3}, ...
    threeWayBlocks{4}
];

%% 7. FIT AND VALIDATE EVERY MODEL

rows = {};

for m = 1:6

    X = modelMatrices{m};

    p = rank(X);
    df = n-p;

    if p < size(X,2)
        error('Rank-deficient model: %s',modelNames{m});
    end

    if df <= 0
        warning('No residual degrees of freedom: %s',modelNames{m});
        continue;
    end

    fprintf('\n===== MODEL %s =====\n',modelNames{m});
    fprintf('Parameters: %d\n',p);
    fprintf('Residual DF: %d\n',df);

    % QR decomposition for leverage
    [Q,~] = qr(X,0);
    h = sum(Q(:,1:p).^2,2);

    for j = 1:2

        y = Y(:,j);

        beta = X\y;
        yhat = X*beta;
        r = y-yhat;

        SSE = sum(r.^2);
        SST = sum((y-mean(y)).^2);

        R2 = 1-SSE/SST;

        adjustedR2 = 1- ...
            (SSE/df)/(SST/(n-1));

        trainingRMSE = sqrt(mean(r.^2));
        trainingMAE = mean(abs(r));
        maximumError = max(abs(r));

        % PRESS-based leave-one-out validation
        denominator = 1-h;

        if any(denominator < 1e-10)
            press = NaN;
            looRMSE = NaN;
            Q2 = NaN;
        else
            looResidual = r./denominator;

            press = sum(looResidual.^2);
            looRMSE = sqrt(press/n);
            Q2 = 1-press/SST;
        end

        rows(end+1,:) = {
            modelNames{m}, ...
            outputNames{j}, ...
            p,df,R2,adjustedR2, ...
            trainingRMSE,trainingMAE, ...
            maximumError,press,looRMSE,Q2
            };

        fprintf('\nOutput: %s\n',outputNames{j});
        fprintf('R2: %.8f\n',R2);
        fprintf('Adjusted R2: %.8f\n',adjustedR2);
        fprintf('Training RMSE: %.8g\n',trainingRMSE);
        fprintf('Training MAE: %.8g\n',trainingMAE);
        fprintf('Maximum error: %.8g\n',maximumError);
        fprintf('LOOCV RMSE: %.8g\n',looRMSE);
        fprintf('LOOCV Q2: %.8f\n',Q2);

    end

end

%% 8. CREATE MODEL COMPARISON TABLE

Results = cell2table(rows, ...
    'VariableNames',{
    'Model','Output','Parameters','ResidualDF', ...
    'R2','AdjustedR2','Training_RMSE','Training_MAE', ...
    'Maximum_Error','PRESS','LOOCV_RMSE','LOOCV_Q2'});

writetable(Results,fullfile(resultFolder, ...
    'Stage11D_Model_Comparison.csv'));

disp(Results);

%% 9. CALCULATE LOOCV IMPROVEMENT OVER BASE MODEL

Improvement = table();

for j = 1:2

    mask = Results.Output == string(outputNames{j});
    R = Results(mask,:);

    baselineRMSE = R.LOOCV_RMSE(1);

    improvementPercent = 100 .* ...
        (baselineRMSE-R.LOOCV_RMSE)./baselineRMSE;

    temp = table( ...
        R.Model, ...
        R.Output, ...
        R.LOOCV_RMSE, ...
        improvementPercent, ...
        'VariableNames',{
        'Model','Output','LOOCV_RMSE','Improvement_Percent'});

    Improvement = [Improvement;temp];

end

writetable(Improvement,fullfile(resultFolder, ...
    'Stage11D_LOOCV_Improvement.csv'));

%% 10. IDENTIFY BEST MODEL BY LOOCV RMSE

BestModels = table();

for j = 1:2

    mask = Results.Output == string(outputNames{j});
    R = Results(mask,:);

    valid = isfinite(R.LOOCV_RMSE);

    R = R(valid,:);

    [~,idx] = min(R.LOOCV_RMSE);

    BestModels = [BestModels;R(idx,:)];

end

fprintf('\n===== BEST MODEL BY LOOCV RMSE =====\n');
disp(BestModels);

writetable(BestModels,fullfile(resultFolder, ...
    'Stage11D_Best_Models.csv'));

%% 11. SAVE COMPLETE RESULTS

save(fullfile(resultFolder,'Stage11D_Complete.mat'), ...
    'T','Y','Xbase','threeWayBlocks', ...
    'modelMatrices','modelNames','threeWayNames', ...
    'Results','Improvement','BestModels');

fprintf('\n===== STAGE 11D COMPLETED =====\n');
fprintf('Results saved in:\n%s\n',resultFolder);
