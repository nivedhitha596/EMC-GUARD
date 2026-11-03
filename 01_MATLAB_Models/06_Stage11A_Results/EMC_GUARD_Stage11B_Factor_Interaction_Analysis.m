
%% EMC-GUARD STAGE 11B
% Factor Interaction and Design Robustness Analysis
clc;
clear;
close all;

fprintf('\n===== EMC-GUARD STAGE 11B =====\n');

%% 1. Load dataset
basePath = 'D:\Downloads\kone\01_MATLAB_Models';

inputPath = fullfile(basePath, ...
    '05_Stage10C_Results','Stage10C_All_Results.csv');

outputPath = fullfile(basePath,'06_Stage11A_Results');

if ~isfolder(outputPath)
    mkdir(outputPath);
end

T = readtable(inputPath);

assert(height(T)==216,'Expected 216 configurations.');

%% 2. Define factorial parameters
factorNames = {
    'PairSeparation_mm'
    'PairPitch_mm'
    'Overlap_mm'
    'GroundClearance_mm'
};

outputNames = {
    'TotalCapacitance_pF_per_m'
    'AbsMutualInductance_nH'
};

Xraw = T{:,factorNames};
Y = T{:,outputNames};

valid = all(isfinite(Xraw),2) & ...
        all(isfinite(Y),2) & ...
        T.GeometryValid==1 & ...
        T.C_PositiveDefinite==1 & ...
        T.L_PositiveDefinite==1;

Xraw = Xraw(valid,:);
Y = Y(valid,:);

n = size(Xraw,1);

fprintf('\nValid configurations: %d\n',n);

%% 3. Generate categorical dummy matrices
D = cell(4,1);
levels = cell(4,1);

for i = 1:4

    levels{i} = unique(Xraw(:,i));
    D{i} = zeros(n,numel(levels{i})-1);

    for j = 2:numel(levels{i})
        D{i}(:,j-1) = Xraw(:,i)==levels{i}(j);
    end

end

%% 4. Construct main-effect and interaction blocks
blocks = {};
termNames = {};
termType = {};

% Intercept
blocks{end+1} = ones(n,1);
termNames{end+1} = 'Intercept';
termType{end+1} = 'Intercept';

% Main effects
for i = 1:4

    blocks{end+1} = D{i};
    termNames{end+1} = factorNames{i};
    termType{end+1} = 'Main effect';

end

% All six two-factor interactions
interactionPairs = [
    1 2
    1 3
    1 4
    2 3
    2 4
    3 4
];

interactionNames = {
    'Separation_x_Pitch'
    'Separation_x_Overlap'
    'Separation_x_GroundClearance'
    'Pitch_x_Overlap'
    'Pitch_x_GroundClearance'
    'Overlap_x_GroundClearance'
};

for k = 1:6

    a = interactionPairs(k,1);
    b = interactionPairs(k,2);

    A = D{a};
    B = D{b};

    interactionBlock = [];

    for i = 1:size(A,2)
        for j = 1:size(B,2)

            interactionBlock = [interactionBlock, ...
                A(:,i).*B(:,j)];

        end
    end

    blocks{end+1} = interactionBlock;
    termNames{end+1} = interactionNames{k};
    termType{end+1} = 'Two-factor interaction';

end

%% 5. Build complete regression model
Xfull = [];
termColumns = cell(numel(blocks),1);

columnStart = 1;

for k = 1:numel(blocks)

    B = blocks{k};

    Xfull = [Xfull,B];

    q = size(B,2);

    termColumns{k} = columnStart:(columnStart+q-1);

    columnStart = columnStart+q;

end

modelRank = rank(Xfull);
residualDF = n-modelRank;

fprintf('\n===== MODEL STRUCTURE =====\n');
fprintf('Total model parameters: %d\n',modelRank);
fprintf('Residual degrees of freedom: %d\n',residualDF);

assert(modelRank==size(Xfull,2), ...
    'Design matrix is not full rank.');

%% 6. Calculate full-model fit
nOutputs = size(Y,2);

R2 = zeros(nOutputs,1);
AdjustedR2 = zeros(nOutputs,1);
FullSSE = zeros(nOutputs,1);

for j = 1:nOutputs

    beta = Xfull\Y(:,j);
    prediction = Xfull*beta;

    residual = Y(:,j)-prediction;

    FullSSE(j) = sum(residual.^2);

    SST = sum((Y(:,j)-mean(Y(:,j))).^2);

    R2(j) = 1-FullSSE(j)/SST;

    AdjustedR2(j) = 1- ...
        (1-R2(j))*(n-1)/residualDF;

end

%% 7. Calculate conditional interaction effects
nInteractions = 6;

PartialR2 = zeros(nInteractions,nOutputs);
Fstat = zeros(nInteractions,nOutputs);
Pvalue = nan(nInteractions,nOutputs);

for k = 1:nInteractions

    termIndex = k+4;
    removedColumns = termColumns{termIndex};

    keep = true(1,size(Xfull,2));
    keep(removedColumns) = false;

    Xreduced = Xfull(:,keep);

    q = numel(removedColumns);

    for j = 1:nOutputs

        betaReduced = Xreduced\Y(:,j);

        residualReduced = ...
            Y(:,j)-Xreduced*betaReduced;

        SSEreduced = sum(residualReduced.^2);

        deltaSSE = max(0,SSEreduced-FullSSE(j));

        PartialR2(k,j) = ...
            deltaSSE/max(SSEreduced,eps);

        if FullSSE(j)>eps && residualDF>0

            Fstat(k,j) = ...
                (deltaSSE/q)/(FullSSE(j)/residualDF);

            if exist('fcdf','file')==2
                Pvalue(k,j) = ...
                    1-fcdf(Fstat(k,j),q,residualDF);
            end

        end

    end

end

%% 8. Display model performance
fprintf('\n===== FULL MODEL PERFORMANCE =====\n');

for j = 1:nOutputs

    fprintf('\nOutput: %s\n',outputNames{j});
    fprintf('R-squared: %.8f\n',R2(j));
    fprintf('Adjusted R-squared: %.8f\n',AdjustedR2(j));
    fprintf('Residual SSE: %.12g\n',FullSSE(j));

end

%% 9. Interaction results table
Results = table( ...
    string(interactionNames(:)), ...
    PartialR2(:,1),Fstat(:,1),Pvalue(:,1), ...
    PartialR2(:,2),Fstat(:,2),Pvalue(:,2), ...
    'VariableNames',{ ...
    'Interaction', ...
    'Capacitance_PartialR2', ...
    'Capacitance_F', ...
    'Capacitance_p', ...
    'MutualInductance_PartialR2', ...
    'MutualInductance_F', ...
    'MutualInductance_p'});

fprintf('\n===== TWO-FACTOR INTERACTION RESULTS =====\n');
disp(Results);

%% 10. Rank interactions
fprintf('\n===== INTERACTION RANKING =====\n');

for j = 1:nOutputs

    [values,order] = sort(PartialR2(:,j),'descend');

    fprintf('\nOutput: %s\n',outputNames{j});

    for k = 1:6

        fprintf('%d. %s : %.8f\n', ...
            k,interactionNames{order(k)},values(k));

    end

end

%% 11. Generate interaction heatmap
figure('Name','EMC-GUARD Interaction Analysis');

imagesc(PartialR2);

colorbar;

set(gca,'XTick',1:2, ...
    'XTickLabel',{'Capacitance','Abs Mutual Inductance'});

set(gca,'YTick',1:6, ...
    'YTickLabel',interactionNames);

xtickangle(15);

ylabel('Factor interaction');
title('Conditional Two-Factor Interaction Strength');

grid on;

saveas(gcf,fullfile(outputPath, ...
    'Stage11B_Interaction_Heatmap.png'));

%% 12. Save results
writetable(Results,fullfile(outputPath, ...
    'Stage11B_Interaction_Results.csv'));

ModelSummary = table( ...
    string(outputNames(:)),R2,AdjustedR2,FullSSE, ...
    repmat(modelRank,nOutputs,1), ...
    repmat(residualDF,nOutputs,1), ...
    'VariableNames',{'Output','R2','AdjustedR2', ...
    'ResidualSSE','ModelRank','ResidualDF'});

writetable(ModelSummary,fullfile(outputPath, ...
    'Stage11B_Model_Summary.csv'));

save(fullfile(outputPath,'Stage11B_Complete.mat'), ...
    'Results','ModelSummary','PartialR2','Fstat', ...
    'Pvalue','Xfull','Y','R2','AdjustedR2', ...
    'FullSSE','factorNames','outputNames');

fprintf('\n===== STAGE 11B COMPLETED =====\n');
fprintf('216 configurations analysed.\n');
fprintf('Six two-factor interactions evaluated.\n');
fprintf('Results and model summary exported.\n');
fprintf('Interaction heatmap generated.\n');
