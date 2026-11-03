
%% STAGE 11B — P-VALUE CORRECTION
clc;
clear;

fprintf('\n===== STAGE 11B P-VALUE CORRECTION =====\n');

basePath = 'D:\Downloads\kone\01_MATLAB_Models';

outputPath = fullfile(basePath,'06_Stage11A_Results');

load(fullfile(outputPath,'Stage11B_Complete.mat'));

% Numerator degrees of freedom for six interactions
df1 = [10; 10; 15; 4; 6; 6];

% Residual degrees of freedom
df2 = 152;

PvalueCorrected = zeros(6,2);

for i = 1:6
    for j = 1:2

        F = Fstat(i,j);

        % Upper-tail F-distribution probability
        x = df2/(df2 + df1(i)*F);

        PvalueCorrected(i,j) = ...
            betainc(x,df2/2,df1(i)/2);

    end
end

Results.Capacitance_p = PvalueCorrected(:,1);
Results.MutualInductance_p = PvalueCorrected(:,2);

fprintf('\n===== CORRECTED P-VALUES =====\n');
disp(Results);

writetable(Results,fullfile(outputPath, ...
    'Stage11B_Interaction_Results_Corrected.csv'));

save(fullfile(outputPath,'Stage11B_Complete.mat'), ...
    'Results','PvalueCorrected','-append');

fprintf('\nP-VALUE CORRECTION COMPLETED\n');
