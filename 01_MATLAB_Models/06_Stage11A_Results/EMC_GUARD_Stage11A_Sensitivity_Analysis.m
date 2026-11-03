
%% EMC-GUARD STAGE 11A — SENSITIVITY ANALYSIS
clc;
close all;

fprintf('\n===== EMC-GUARD STAGE 11A =====\n');

%% 1. Load recovered dataset
projectPath = 'D:\Downloads\kone\01_MATLAB_Models';
resultPath = fullfile(projectPath,'05_Stage10C_Results');
outputPath = fullfile(projectPath,'06_Stage11A_Results');

if ~isfolder(outputPath)
    mkdir(outputPath);
end

T = readtable(fullfile(resultPath,'Stage10C_All_Results.csv'));

assert(height(T)==216,'Unexpected configuration count.');

%% 2. Define input and output parameters
inputNames = {
    'PairSeparation_mm'
    'PairPitch_mm'
    'Overlap_mm'
    'GroundClearance_mm'
};

outputNames = {
    'TotalCapacitance_pF_per_m'
    'AbsMutualInductance_nH'
    'MutualRatio'
};

X = T{:,inputNames};
Y = T{:,outputNames};

%% 3. Data validity checks
valid = all(isfinite(X),2) & all(isfinite(Y),2) & ...
    T.GeometryValid==1 & ...
    T.C_PositiveDefinite==1 & ...
    T.L_PositiveDefinite==1;

fprintf('\nValid configurations: %d / %d\n',sum(valid),height(T));

X = X(valid,:);
Y = Y(valid,:);
Tvalid = T(valid,:);

%% 4. Calculate parameter sensitivity
nInputs = numel(inputNames);
nOutputs = numel(outputNames);

EtaSquared = zeros(nInputs,nOutputs);
SpearmanR = zeros(nInputs,nOutputs);

for i = 1:nInputs

    x = X(:,i);
    levels = unique(x);

    % Rank correlation
    rx = rankTies(x);

    for j = 1:nOutputs

        y = Y(:,j);
        ry = rankTies(y);

        % Spearman correlation
        rx0 = rx-mean(rx);
        ry0 = ry-mean(ry);

        SpearmanR(i,j) = ...
            sum(rx0.*ry0)/sqrt(sum(rx0.^2)*sum(ry0.^2));

        % Grouped variance sensitivity index
        grandMean = mean(y);
        totalSS = sum((y-grandMean).^2);

        betweenSS = 0;

        for k = 1:numel(levels)

            idx = (x==levels(k));
            groupMean = mean(y(idx));

            betweenSS = betweenSS + ...
                sum(idx)*(groupMean-grandMean)^2;
        end

        if totalSS > 0
            EtaSquared(i,j) = betweenSS/totalSS;
        end

    end
end

%% 5. Display sensitivity results
fprintf('\n===== ETA-SQUARED SENSITIVITY =====\n');

SensitivityTable = array2table(EtaSquared, ...
    'RowNames',inputNames, ...
    'VariableNames',{'Capacitance','AbsMutualInductance','MutualRatio'});

disp(SensitivityTable);

fprintf('\n===== SPEARMAN RANK CORRELATION =====\n');

CorrelationTable = array2table(SpearmanR, ...
    'RowNames',inputNames, ...
    'VariableNames',{'Capacitance','AbsMutualInductance','MutualRatio'});

disp(CorrelationTable);

%% 6. Identify strongest associated parameter
fprintf('\n===== SENSITIVITY RANKING =====\n');

for j = 1:nOutputs

    [sortedValues,order] = sort(EtaSquared(:,j),'descend');

    fprintf('\nOutput: %s\n',outputNames{j});

    for k = 1:nInputs
        fprintf('%d. %s : %.5f\n', ...
            k,inputNames{order(k)},sortedValues(k));
    end
end

%% 7. Identify worst-case configurations
[~,idxC] = max(Y(:,1));
[~,idxM] = max(Y(:,2));
[~,idxR] = max(Y(:,3));

fprintf('\n===== WORST-CASE CONFIGURATIONS =====\n');

fprintf('\nMaximum capacitance:\n');
disp(Tvalid(idxC,:));

fprintf('\nMaximum absolute mutual inductance:\n');
disp(Tvalid(idxM,:));

fprintf('\nMaximum mutual ratio:\n');
disp(Tvalid(idxR,:));

WorstCaseTable = Tvalid([idxC;idxM;idxR],:);
WorstCaseTable.Case = {
    'Maximum capacitance'
    'Maximum absolute mutual inductance'
    'Maximum mutual ratio'
};

WorstCaseTable = movevars(WorstCaseTable,'Case','Before',1);

%% 8. Generate sensitivity comparison graph
figure('Name','EMC-GUARD Sensitivity Analysis');

bar(EtaSquared);

set(gca,'XTick',1:nInputs, ...
    'XTickLabel',inputNames);

xtickangle(25);

ylabel('Eta-squared sensitivity index');
title('Geometry Parameter Sensitivity');
legend(outputNames,'Location','best');
grid on;

saveas(gcf,fullfile(outputPath, ...
    'Stage11A_Sensitivity_Comparison.png'));

%% 9. Generate correlation graph
figure('Name','EMC-GUARD Spearman Correlation');

imagesc(SpearmanR);
colorbar;
clim([-1 1]);

set(gca,'XTick',1:nOutputs, ...
    'XTickLabel',outputNames);

set(gca,'YTick',1:nInputs, ...
    'YTickLabel',inputNames);

xtickangle(20);

title('Spearman Rank Correlation');
grid on;

saveas(gcf,fullfile(outputPath, ...
    'Stage11A_Spearman_Correlation.png'));

%% 10. Save numerical results
writetable(SensitivityTable,fullfile(outputPath, ...
    'Stage11A_Sensitivity_Indices.csv'), ...
    'WriteRowNames',true);

writetable(CorrelationTable,fullfile(outputPath, ...
    'Stage11A_Spearman_Results.csv'), ...
    'WriteRowNames',true);

writetable(WorstCaseTable,fullfile(outputPath, ...
    'Stage11A_Worst_Case_Configurations.csv'));

save(fullfile(outputPath,'Stage11A_Complete.mat'), ...
    'Tvalid','EtaSquared','SpearmanR', ...
    'SensitivityTable','CorrelationTable', ...
    'WorstCaseTable','inputNames','outputNames');

fprintf('\n===== STAGE 11A COMPLETED =====\n');
fprintf('Configurations analysed: %d\n',height(Tvalid));
fprintf('Sensitivity indices saved.\n');
fprintf('Correlation results saved.\n');
fprintf('Worst-case configurations saved.\n');
fprintf('Graphs generated successfully.\n');

%% Local function: average ranks for tied values
function r = rankTies(x)

    [sortedX,order] = sort(x);
    r = zeros(size(x));

    i = 1;
    while i <= numel(x)

        j = i;

        while j < numel(x) && sortedX(j+1)==sortedX(i)
            j = j+1;
        end

        averageRank = (i+j)/2;
        r(order(i:j)) = averageRank;

        i = j+1;
    end
end
