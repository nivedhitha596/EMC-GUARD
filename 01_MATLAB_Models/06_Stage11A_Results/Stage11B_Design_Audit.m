
%% EMC-GUARD STAGE 11B — DESIGN AUDIT
clc;
clear;
close all;

fprintf('\n===== STAGE 11B DESIGN AUDIT =====\n');

basePath = 'D:\Downloads\kone\01_MATLAB_Models';

T = readtable(fullfile(basePath, ...
    '05_Stage10C_Results', ...
    'Stage10C_All_Results.csv'));

outputPath = fullfile(basePath,'06_Stage11A_Results');

%% Unique parameter levels
names = {
    'PairSeparation_mm'
    'PairPitch_mm'
    'Overlap_mm'
    'GroundClearance_mm'
};

fprintf('\n===== FACTOR LEVELS =====\n');

for i = 1:numel(names)

    levels = unique(T.(names{i}));

    fprintf('\n%s: %d levels\n',names{i},numel(levels));
    disp(levels');

end

%% Check unique geometry combinations
G = T(:,names);
[uniqueGeometry,~,groupID] = unique(G,'rows');

fprintf('\nTotal rows: %d\n',height(T));
fprintf('Unique geometries: %d\n',height(uniqueGeometry));

counts = accumarray(groupID,1);

fprintf('Minimum repetitions: %d\n',min(counts));
fprintf('Maximum repetitions: %d\n',max(counts));

%% Check mutual ratio mathematical relationship
M = T.AbsMutualInductance_nH;
R = T.MutualRatio;

p = polyfit(M,R,1);
Rpred = polyval(p,M);

residual = R-Rpred;

fprintf('\n===== MUTUAL RATIO RELATIONSHIP =====\n');
fprintf('Slope: %.12g\n',p(1));
fprintf('Intercept: %.12g\n',p(2));
fprintf('Maximum absolute residual: %.12g\n',max(abs(residual)));

%% Save audit
AuditTable = table( ...
    p(1),p(2),max(abs(residual)), ...
    'VariableNames',{'Slope','Intercept','MaximumResidual'});

writetable(AuditTable,fullfile(outputPath, ...
    'Stage11B_Design_Audit.csv'));

fprintf('\n===== STAGE 11B AUDIT COMPLETE =====\n');
