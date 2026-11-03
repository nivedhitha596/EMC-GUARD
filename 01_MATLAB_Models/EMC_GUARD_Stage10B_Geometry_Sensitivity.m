
%% EMC-GUARD STAGE 10B-2
% Geometry Sensitivity Analysis
% Simplified four-conductor / two-pair model
%
% IMPORTANT:
% This is a parameterised research model.
% It is NOT a verified manufacturer cable cross-section.
% Shield and twisting effects are not included yet.

clear;
clc;

%% 1. Physical constants

mu0 = 4*pi*1e-7;
eps0 = 8.854187817e-12;

%% 2. Cable reference dimensions

CableWidth = 35.45;        % mm
CableHeight = 5.80;        % mm

% Equivalent radius of 0.50 mm2 solid copper
a = sqrt(0.50/pi);         % mm

% Simplified modelling assumptions
h = 50;                    % mm above reference ground plane
er = 2.3;                  % assumed effective relative permittivity
eps = eps0*er;

Ltotal = 0.450;             % m

% Assumed self-inductances from Stage 10A
Ls = 2e-6;
Lv = 10e-6;

%% 3. Geometry sweep

% Separation between centres of the two pairs
PairSeparation = [8 12 16 20 24 28];  % mm

% Spacing between conductors inside each pair
PairPitch = [2 3 4];                  % mm

% Parallel overlap length
Overlap = [150 300 450];              % mm

%% 4. Initialise results

Results = [];
row = 0;

fprintf('===== EMC-GUARD STAGE 10B-2 =====\n');

%% 5. Geometry sensitivity calculation

for Dmm = PairSeparation

    for Pmm = PairPitch

        for Lpmm = Overlap

            % Convert dimensions into metres
            D = Dmm*1e-3;
            pitch = Pmm*1e-3;
            Lp = Lpmm*1e-3;

            %% 5A. Conductor positions

            xpos_mm = [ ...
                -Dmm/2-Pmm/2, ...
                -Dmm/2+Pmm/2, ...
                 Dmm/2-Pmm/2, ...
                 Dmm/2+Pmm/2];

            % Check cable width and conductor clearance
            withinWidth = ...
                max(abs(xpos_mm))+a <= CableWidth/2;

            clearances = diff(xpos_mm);

            noOverlap = all(clearances > 2*a);

            geometryValid = withinWidth && noOverlap;

            if ~geometryValid
                continue;
            end

            xpos = xpos_mm*1e-3;
            ar = a*1e-3;
            hr = h*1e-3;

            %% 5B. Potential coefficient matrix

            P = zeros(4,4);

            for i = 1:4

                for j = 1:4

                    if i == j

                        P(i,i) = ...
                            log(2*hr/ar)/(2*pi*eps);

                    else

                        r = abs(xpos(i)-xpos(j));

                        ri = sqrt(r^2+(2*hr)^2);

                        P(i,j) = ...
                            log(ri/r)/(2*pi*eps);

                    end
                end
            end

            %% 5C. Capacitance matrix per unit length

            Cprime = P\eye(4);

            symmetryC = norm(Cprime-Cprime','fro');

            eigC = eig(0.5*(Cprime+Cprime'));

            passC = all(eigC > 0);

            %% 5D. Mutual inductance

            % Aggressor pair and victim pair
            sx = [-D/2-pitch/2, -D/2+pitch/2];

            vx = [D/2-pitch/2, D/2+pitch/2];

            currentSign = [1 -1];

            M = 0;

            for i = 1:2

                for j = 1:2

                    r = abs(sx(i)-vx(j));

                    Mp = mu0/(2*pi)*( ...
                        Lp*asinh(Lp/r) ...
                        -sqrt(Lp^2+r^2)+r);

                    M = M + ...
                        currentSign(i)*currentSign(j)*Mp;

                end
            end

            %% 5E. Inductance matrix

            Lmatrix = [Ls M; M Lv];

            eigL = eig(0.5*(Lmatrix+Lmatrix'));

            symmetryL = norm(Lmatrix-Lmatrix','fro');

            passL = all(eigL > 0);

            mutualRatio = abs(M)/sqrt(Ls*Lv);

            %% 5F. Store results

            row = row+1;

            Results(row,:) = [ ...
                Dmm, ...
                Pmm, ...
                Lpmm, ...
                xpos_mm, ...
                M*1e9, ...
                min(eigL)*1e9, ...
                min(eigC)*1e12, ...
                symmetryL, ...
                symmetryC, ...
                mutualRatio, ...
                passL, ...
                passC, ...
                geometryValid];

        end
    end
end

%% 6. Create results table

T = array2table(Results, ...
    'VariableNames',{ ...
    'PairSeparation_mm', ...
    'PairPitch_mm', ...
    'Overlap_mm', ...
    'Conductor1_X_mm', ...
    'Conductor2_X_mm', ...
    'Conductor3_X_mm', ...
    'Conductor4_X_mm', ...
    'MutualInductance_nH', ...
    'MinEigen_L_nH', ...
    'MinEigen_C_pF_per_m', ...
    'SymmetryError_L', ...
    'SymmetryError_C', ...
    'MutualRatio', ...
    'L_PositiveDefinite', ...
    'C_PositiveDefinite', ...
    'GeometryValid'});

disp(T);

%% 7. Overall verification

fprintf('\n===== STAGE 10B-2 SUMMARY =====\n');

fprintf('Total valid configurations: %d\n',height(T));

fprintf('All geometries within cable envelope: %d\n', ...
    all(T.GeometryValid));

fprintf('All L matrices positive definite: %d\n', ...
    all(T.L_PositiveDefinite));

fprintf('All C matrices positive definite: %d\n', ...
    all(T.C_PositiveDefinite));

fprintf('Maximum mutual ratio: %.8f\n', ...
    max(T.MutualRatio));

%% 8. Export results

outFolder = fullfile(pwd,'03_Stage10B_Results');

if ~exist(outFolder,'dir')
    mkdir(outFolder);
end

writetable(T,fullfile(outFolder, ...
    'EMC_GUARD_Stage10B_Geometry_Sensitivity.csv'));

save(fullfile(outFolder, ...
    'EMC_GUARD_Stage10B_Geometry_Sensitivity.mat'), ...
    'T','CableWidth','CableHeight','a','h','er');

fprintf('\nStage 10B-2 completed.\n');
fprintf('Results exported successfully.\n');
