
%% EMC-GUARD STAGE 10B-2A
% 2D cross-section validation and ground-clearance sensitivity
% Simplified four-conductor / two-pair model.
% NOT a verified manufacturer cable cross-section.

clear; clc; close all;

%% 1. CABLE REFERENCE DIMENSIONS

CableWidth_mm  = 35.45;
CableHeight_mm = 5.80;

% Equivalent copper radius for 0.50 mm^2 conductor
CopperArea_mm2 = 0.50;
a_mm = sqrt(CopperArea_mm2/pi);

% Ground clearance measured from cable bottom face to reference plane
GroundClearance_mm = [10 25 50 100];

% Hypothetical conductor arrangements
PairSeparation_mm = [8 12 16 20 24 28];
PairPitch_mm = [2 3 4];
Overlap_mm = [150 300 450];

% Electrical assumptions retained from previous model
er = 2.3;
eps0 = 8.854187817e-12;
mu0 = 4*pi*1e-7;

Ltotal_m = 0.450;
Ls = 2e-6;
Lv = 10e-6;

currentSigns = [1 -1];

%% 2. OUTPUT DIRECTORY

outDir = '04_Stage10B_2A_Results';

if ~exist(outDir,'dir')
    mkdir(outDir);
end

%% 3. CREATE CROSS-SECTION GEOMETRY

geometryRows = [];
resultRows = [];

geometryCount = 0;
validGeometryCount = 0;
invalidGeometryCount = 0;

for D = PairSeparation_mm
    for p = PairPitch_mm

        % Four conductor x-coordinates
        x_mm = [-D/2-p/2, -D/2+p/2, ...
                 D/2-p/2,  D/2+p/2];

        % First-pass single-row arrangement
        y_mm = zeros(1,4);

        % Check copper-core clearance inside cable envelope
        widthOK = all(abs(x_mm) + a_mm <= CableWidth_mm/2);
        heightOK = all(abs(y_mm) + a_mm <= CableHeight_mm/2);

        % Pairwise copper-core non-overlap
        minSpacing_mm = inf;

        for i = 1:4
            for j = i+1:4
                spacing = hypot(x_mm(i)-x_mm(j), ...
                                y_mm(i)-y_mm(j));
                minSpacing_mm = min(minSpacing_mm,spacing);
            end
        end

        nonOverlapOK = minSpacing_mm > 2*a_mm;

        geometryValid = widthOK && heightOK && nonOverlapOK;

        geometryCount = geometryCount + 1;

        if geometryValid
            validGeometryCount = validGeometryCount + 1;
        else
            invalidGeometryCount = invalidGeometryCount + 1;
        end

        geometryRows = [geometryRows; ...
            D, p, minSpacing_mm, widthOK, heightOK, ...
            nonOverlapOK, geometryValid];

        if ~geometryValid
            continue;
        end

        %% 4. GROUND-CLEARANCE SENSITIVITY

        for hClear = GroundClearance_mm

            % Conductor centre height above reference plane
            z_mm = hClear + CableHeight_mm/2 + y_mm;

            x_m = x_mm*1e-3;
            z_m = z_mm*1e-3;
            a_m = a_mm*1e-3;

            epsilon = eps0*er;

            %% 5. POTENTIAL COEFFICIENT MATRIX

            P = zeros(4,4);

            for i = 1:4
                for j = 1:4

                    if i == j
                        P(i,j) = log(2*z_m(i)/a_m) / ...
                                 (2*pi*epsilon);
                    else
                        directDistance = hypot( ...
                            x_m(i)-x_m(j), ...
                            z_m(i)-z_m(j));

                        imageDistance = hypot( ...
                            x_m(i)-x_m(j), ...
                            z_m(i)+z_m(j));

                        P(i,j) = log(imageDistance/directDistance) / ...
                                 (2*pi*epsilon);
                    end
                end
            end

            % Numerical symmetry
            P = 0.5*(P+P');

            % Capacitance per unit length
            Cprime = P\eye(4);
            Cprime = 0.5*(Cprime+Cprime');

            eigC = eig(Cprime);

            Cpositive = all(eigC > 0);
            symmetryC = norm(Cprime-Cprime','fro');

            %% 6. MUTUAL INDUCTANCE

            overlapValues = Overlap_mm;

            for overlap = overlapValues

                Lseg = overlap*1e-3;

                M = 0;

                % Pair 1 conductors: 1,2
                % Pair 2 conductors: 3,4
                for i = 1:2
                    for j = 3:4

                        distance = hypot( ...
                            x_m(i)-x_m(j), ...
                            z_m(i)-z_m(j));

                        Mpartial = mu0/(2*pi) * ...
                            (Lseg*asinh(Lseg/distance) ...
                            - sqrt(Lseg^2+distance^2) ...
                            + distance);

                        M = M + currentSigns(i)* ...
                                currentSigns(j-2)*Mpartial;
                    end
                end

                % Two-loop inductance matrix
                Lmatrix = [Ls M; M Lv];

                eigL = eig(Lmatrix);
                Lpositive = all(eigL > 0);

                symmetryL = norm(Lmatrix-Lmatrix','fro');

                mutualRatio = abs(M)/sqrt(Ls*Lv);

                %% 7. STORE RESULTS

                resultRows = [resultRows; ...
                    D, p, overlap, hClear, ...
                    minSpacing_mm, ...
                    min(eigC)*1e12, max(eigC)*1e12, ...
                    M*1e9, mutualRatio, ...
                    min(eigL)*1e9, ...
                    Cpositive, Lpositive, ...
                    symmetryC, symmetryL];
            end
        end
    end
end

%% 8. EXPORT RESULTS

GeometryTable = array2table(geometryRows, ...
    'VariableNames', { ...
    'PairSeparation_mm', ...
    'PairPitch_mm', ...
    'MinimumCopperSpacing_mm', ...
    'WidthCheck', ...
    'HeightCheck', ...
    'CopperNonOverlapCheck', ...
    'CopperGeometryValid'});

ResultsTable = array2table(resultRows, ...
    'VariableNames', { ...
    'PairSeparation_mm', ...
    'PairPitch_mm', ...
    'Overlap_mm', ...
    'GroundClearance_mm', ...
    'MinimumCopperSpacing_mm', ...
    'MinEigen_C_pF_per_m', ...
    'MaxEigen_C_pF_per_m', ...
    'MutualInductance_nH', ...
    'MutualRatio', ...
    'MinEigen_L_nH', ...
    'C_PositiveDefinite', ...
    'L_PositiveDefinite', ...
    'SymmetryError_C', ...
    'SymmetryError_L'});

writetable(GeometryTable, ...
    fullfile(outDir,'Stage10B_2A_Geometry_Check.csv'));

writetable(ResultsTable, ...
    fullfile(outDir,'Stage10B_2A_Sensitivity_Results.csv'));

save(fullfile(outDir,'Stage10B_2A_Results.mat'), ...
    'GeometryTable','ResultsTable');

%% 9. SUMMARY

fprintf('\n===== EMC-GUARD STAGE 10B-2A =====\n');

fprintf('Cross-section: %.2f x %.2f mm\n', ...
    CableWidth_mm,CableHeight_mm);

fprintf('Equivalent copper radius: %.4f mm\n',a_mm);

fprintf('Geometry arrangements checked: %d\n',geometryCount);
fprintf('Copper-core feasible arrangements: %d\n',validGeometryCount);
fprintf('Invalid arrangements: %d\n',invalidGeometryCount);

fprintf('Sensitivity cases calculated: %d\n',height(ResultsTable));

fprintf('Minimum C eigenvalue: %.6f pF/m\n', ...
    min(ResultsTable.MinEigen_C_pF_per_m));

fprintf('Maximum C eigenvalue: %.6f pF/m\n', ...
    max(ResultsTable.MaxEigen_C_pF_per_m));

fprintf('Maximum absolute mutual inductance: %.6f nH\n', ...
    max(abs(ResultsTable.MutualInductance_nH)));

fprintf('Maximum mutual ratio: %.8f\n', ...
    max(ResultsTable.MutualRatio));

fprintf('All C matrices positive definite: %d\n', ...
    all(ResultsTable.C_PositiveDefinite));

fprintf('All L matrices positive definite: %d\n', ...
    all(ResultsTable.L_PositiveDefinite));

%% 10. CROSS-SECTION PLOT

figure('Name','EMC-GUARD Cable Cross Section');

rectangle('Position', ...
    [-CableWidth_mm/2, -CableHeight_mm/2, ...
    CableWidth_mm, CableHeight_mm], ...
    'EdgeColor','k','LineWidth',2);

hold on;

D = 16;
p = 3;

x_mm = [-D/2-p/2, -D/2+p/2, ...
         D/2-p/2,  D/2+p/2];

y_mm = zeros(1,4);

theta = linspace(0,2*pi,100);

for i = 1:4
    plot(x_mm(i)+a_mm*cos(theta), ...
         y_mm(i)+a_mm*sin(theta),'b','LineWidth',1.5);

    text(x_mm(i),y_mm(i)+0.7,sprintf('C%d',i), ...
        'HorizontalAlignment','center');
end

axis equal;
grid on;
xlabel('Width direction (mm)');
ylabel('Height direction (mm)');
title('Hypothetical Copper-Core Arrangement');
xlim([-20 20]);
ylim([-5 5]);

exportgraphics(gcf, ...
    fullfile(outDir,'Cable_CrossSection.png'), ...
    'Resolution',300);

%% 11. GROUND-CLEARANCE SENSITIVITY PLOT

figure('Name','Ground Clearance Sensitivity');

hold on;
grid on;

for hClear = GroundClearance_mm

    subset = ResultsTable( ...
        ResultsTable.GroundClearance_mm == hClear & ...
        ResultsTable.Overlap_mm == 450, :);

    plot(subset.PairSeparation_mm, ...
         abs(subset.MutualInductance_nH), ...
         '-o','DisplayName',sprintf('%d mm',hClear));
end

xlabel('Pair-centre separation (mm)');
ylabel('|Mutual inductance| (nH)');
title('Ground-Clearance Sensitivity at 450 mm Overlap');
legend('Location','best');

exportgraphics(gcf, ...
    fullfile(outDir,'GroundClearance_Sensitivity.png'), ...
    'Resolution',300);

fprintf('\nStage 10B-2A completed. Results saved in %s\n',outDir);
