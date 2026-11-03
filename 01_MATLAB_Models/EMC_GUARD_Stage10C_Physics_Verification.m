
%% EMC-GUARD STAGE 10C
% Physics model behaviour verification
% Checks:
% 1. Separation vs mutual inductance
% 2. Overlap vs mutual inductance
% 3. Ground clearance vs total capacitance
%
% This is numerical verification, not experimental validation.

clear; clc; close all;

%% 1. MODEL PARAMETERS

CableWidth_mm = 35.45;
CableHeight_mm = 5.80;

a_mm = sqrt(0.50/pi);
a = a_mm*1e-3;

D_values = [8 12 16 20 24 28];
p_values = [2 3 4];
overlap_values = [150 300 450];
h_values = [10 25 50 100];

er = 2.3;
eps0 = 8.854187817e-12;
mu0 = 4*pi*1e-7;

Ls = 2e-6;
Lv = 10e-6;

currentSigns = [1 -1];

outDir = '05_Stage10C_Results';

if ~exist(outDir,'dir')
    mkdir(outDir);
end

%% 2. CALCULATE ALL 216 CONFIGURATIONS

rows = [];

for D = D_values
    for p = p_values

        x_mm = [-D/2-p/2, -D/2+p/2, ...
                 D/2-p/2,  D/2+p/2];

        y_mm = zeros(1,4);

        % Copper-core geometry checks
        widthOK = all(abs(x_mm)+a_mm <= CableWidth_mm/2);
        heightOK = all(abs(y_mm)+a_mm <= CableHeight_mm/2);

        minSpacing = inf;

        for i = 1:4
            for j = i+1:4
                minSpacing = min(minSpacing, ...
                    hypot(x_mm(i)-x_mm(j), ...
                          y_mm(i)-y_mm(j)));
            end
        end

        nonOverlapOK = minSpacing > 2*a_mm;
        geometryOK = widthOK && heightOK && nonOverlapOK;

        if ~geometryOK
            continue;
        end

        x = x_mm*1e-3;

        for hClear = h_values

            z = ones(1,4)*(hClear+CableHeight_mm/2)*1e-3;

            %% 3. CAPACITANCE MATRIX

            epsilon = eps0*er;
            P = zeros(4,4);

            for i = 1:4
                for j = 1:4

                    if i == j

                        P(i,j) = log(2*z(i)/a) / ...
                                 (2*pi*epsilon);

                    else

                        directDistance = hypot( ...
                            x(i)-x(j),z(i)-z(j));

                        imageDistance = hypot( ...
                            x(i)-x(j),z(i)+z(j));

                        P(i,j) = log(imageDistance/directDistance) / ...
                                 (2*pi*epsilon);
                    end
                end
            end

            P = 0.5*(P+P');

            Cprime = P\eye(4);
            Cprime = 0.5*(Cprime+Cprime');

            eigC = eig(Cprime);

            Cpositive = all(eigC > 0);

            % Total capacitance per unit length to reference plane
            Ctotal_pF_per_m = sum(Cprime(:))*1e12;

            %% 4. MUTUAL INDUCTANCE FOR EACH OVERLAP

            for overlap = overlap_values

                Lseg = overlap*1e-3;
                M = 0;

                for i = 1:2
                    for j = 3:4

                        distance = hypot( ...
                            x(i)-x(j),z(i)-z(j));

                        Mpartial = mu0/(2*pi) * ...
                            (Lseg*asinh(Lseg/distance) ...
                            -sqrt(Lseg^2+distance^2)+distance);

                        M = M + currentSigns(i)* ...
                                currentSigns(j-2)*Mpartial;
                    end
                end

                Lmatrix = [Ls M; M Lv];
                eigL = eig(Lmatrix);

                Lpositive = all(eigL > 0);

                mutualRatio = abs(M)/sqrt(Ls*Lv);

                rows = [rows; ...
                    D,p,overlap,hClear, ...
                    minSpacing,Ctotal_pF_per_m, ...
                    M*1e9,abs(M)*1e9,mutualRatio, ...
                    min(eigC)*1e12,min(eigL)*1e9, ...
                    geometryOK,Cpositive,Lpositive];
            end
        end
    end
end

%% 5. CREATE RESULTS TABLE

Results = array2table(rows, ...
    'VariableNames',{ ...
    'PairSeparation_mm', ...
    'PairPitch_mm', ...
    'Overlap_mm', ...
    'GroundClearance_mm', ...
    'MinimumCopperSpacing_mm', ...
    'TotalCapacitance_pF_per_m', ...
    'MutualInductance_nH', ...
    'AbsMutualInductance_nH', ...
    'MutualRatio', ...
    'MinEigen_C_pF_per_m', ...
    'MinEigen_L_nH', ...
    'GeometryValid', ...
    'C_PositiveDefinite', ...
    'L_PositiveDefinite'});

writetable(Results, ...
    fullfile(outDir,'Stage10C_All_Results.csv'));

%% 6. TREND CHECK 1: SEPARATION

separationChecks = 0;
separationPass = 0;

for p = p_values
    for h = h_values
        for overlap = overlap_values

            subset = Results( ...
                Results.PairPitch_mm == p & ...
                Results.GroundClearance_mm == h & ...
                Results.Overlap_mm == overlap,:);

            subset = sortrows(subset,'PairSeparation_mm');

            values = subset.AbsMutualInductance_nH;

            separationChecks = separationChecks+1;

            if all(diff(values) <= 1e-10)
                separationPass = separationPass+1;
            end
        end
    end
end

%% 7. TREND CHECK 2: OVERLAP

overlapChecks = 0;
overlapPass = 0;

for D = D_values
    for p = p_values
        for h = h_values

            subset = Results( ...
                Results.PairSeparation_mm == D & ...
                Results.PairPitch_mm == p & ...
                Results.GroundClearance_mm == h,:);

            subset = sortrows(subset,'Overlap_mm');

            values = subset.AbsMutualInductance_nH;

            overlapChecks = overlapChecks+1;

            if all(diff(values) >= -1e-10)
                overlapPass = overlapPass+1;
            end
        end
    end
end

%% 8. TREND CHECK 3: GROUND CLEARANCE

clearanceChecks = 0;
clearancePass = 0;

for D = D_values
    for p = p_values

        subset = Results( ...
            Results.PairSeparation_mm == D & ...
            Results.PairPitch_mm == p & ...
            Results.Overlap_mm == 150,:);

        % One capacitance value per clearance.
        subset = unique(subset(:, ...
            {'GroundClearance_mm','TotalCapacitance_pF_per_m'}));

        subset = sortrows(subset,'GroundClearance_mm');

        values = subset.TotalCapacitance_pF_per_m;

        clearanceChecks = clearanceChecks+1;

        if all(diff(values) <= 1e-10)
            clearancePass = clearancePass+1;
        end
    end
end

%% 9. MATRIX AND GEOMETRY CHECKS

allGeometry = all(Results.GeometryValid);
allCpositive = all(Results.C_PositiveDefinite);
allLpositive = all(Results.L_PositiveDefinite);

%% 10. VERIFICATION SUMMARY

fprintf('\n===== EMC-GUARD STAGE 10C =====\n');

fprintf('Configurations calculated: %d\n',height(Results));

fprintf('\n--- Separation trend ---\n');
fprintf('Checks passed: %d / %d\n', ...
    separationPass,separationChecks);

fprintf('\n--- Overlap trend ---\n');
fprintf('Checks passed: %d / %d\n', ...
    overlapPass,overlapChecks);

fprintf('\n--- Ground-clearance capacitance trend ---\n');
fprintf('Checks passed: %d / %d\n', ...
    clearancePass,clearanceChecks);

fprintf('\n--- Matrix and geometry checks ---\n');
fprintf('All geometries valid: %d\n',allGeometry);
fprintf('All C matrices positive definite: %d\n',allCpositive);
fprintf('All L matrices positive definite: %d\n',allLpositive);

fprintf('\nMaximum |M|: %.6f nH\n', ...
    max(Results.AbsMutualInductance_nH));

fprintf('Minimum total capacitance: %.6f pF/m\n', ...
    min(Results.TotalCapacitance_pF_per_m));

fprintf('Maximum total capacitance: %.6f pF/m\n', ...
    max(Results.TotalCapacitance_pF_per_m));

%% 11. SAVE VERIFICATION SUMMARY

CheckName = { ...
    'Separation reduces absolute mutual inductance'; ...
    'Overlap increases absolute mutual inductance'; ...
    'Ground clearance reduces total capacitance'; ...
    'All geometries pass'; ...
    'All C matrices positive definite'; ...
    'All L matrices positive definite'};

Passed = [ ...
    separationPass == separationChecks; ...
    overlapPass == overlapChecks; ...
    clearancePass == clearanceChecks; ...
    allGeometry; ...
    allCpositive; ...
    allLpositive];

VerificationSummary = table(CheckName,Passed);

writetable(VerificationSummary, ...
    fullfile(outDir,'Stage10C_Verification_Summary.csv'));

save(fullfile(outDir,'Stage10C_Complete.mat'), ...
    'Results','VerificationSummary');

%% 12. PLOT A: SEPARATION TREND

figure('Name','Separation Trend');

subset = Results( ...
    Results.PairPitch_mm == 4 & ...
    Results.Overlap_mm == 450 & ...
    Results.GroundClearance_mm == 50,:);

subset = sortrows(subset,'PairSeparation_mm');

plot(subset.PairSeparation_mm, ...
    subset.AbsMutualInductance_nH,'-o','LineWidth',1.5);

grid on;
xlabel('Pair-centre separation (mm)');
ylabel('|Mutual inductance| (nH)');
title('Effect of Pair Separation on Mutual Inductance');

exportgraphics(gcf, ...
    fullfile(outDir,'Stage10C_Separation_Trend.png'), ...
    'Resolution',300);

%% 13. PLOT B: OVERLAP TREND

figure('Name','Overlap Trend');

subset = Results( ...
    Results.PairSeparation_mm == 8 & ...
    Results.PairPitch_mm == 4 & ...
    Results.GroundClearance_mm == 50,:);

subset = sortrows(subset,'Overlap_mm');

plot(subset.Overlap_mm, ...
    subset.AbsMutualInductance_nH,'-o','LineWidth',1.5);

grid on;
xlabel('Conductor overlap (mm)');
ylabel('|Mutual inductance| (nH)');
title('Effect of Overlap on Mutual Inductance');

exportgraphics(gcf, ...
    fullfile(outDir,'Stage10C_Overlap_Trend.png'), ...
    'Resolution',300);

%% 14. PLOT C: GROUND CLEARANCE TREND

figure('Name','Ground Clearance Trend');

subset = Results( ...
    Results.PairSeparation_mm == 16 & ...
    Results.PairPitch_mm == 3 & ...
    Results.Overlap_mm == 150,:);

subset = unique(subset(:, ...
    {'GroundClearance_mm','TotalCapacitance_pF_per_m'}));

subset = sortrows(subset,'GroundClearance_mm');

plot(subset.GroundClearance_mm, ...
    subset.TotalCapacitance_pF_per_m,'-o','LineWidth',1.5);

grid on;
xlabel('Ground clearance (mm)');
ylabel('Total capacitance (pF/m)');
title('Effect of Ground Clearance on Capacitance');

exportgraphics(gcf, ...
    fullfile(outDir,'Stage10C_GroundClearance_Trend.png'), ...
    'Resolution',300);

fprintf('\nStage 10C completed.\n');
fprintf('Results saved in %s\n',outDir);
