
%% EMC-GUARD | STAGE 8B
% Four-conductor Maxwell capacitance coefficient matrix
%
% Analytical approximation:
% - Four parallel cylindrical conductors
% - Homogeneous dielectric
% - Infinite grounded reference plane
% - Method-of-images potential coefficient formulation
%
% NOT a validated model of a real insulated elevator cable.

clear; clc; close all;

%% Constants
eps0 = 8.854187817e-12;
mu0 = 4*pi*1e-7; %#ok<NASGU>

%% Geometry assumptions
L = 0.450;                 % Parallel length (m)
a = 0.0005;                % Conductor radius (m)
pitch = 0.006;             % Core pitch (m)
h = 0.050;                 % Height above grounded plane (m)
er = 2.3;                  % Assumed homogeneous dielectric
eps = eps0*er;

separationList = [0.020 0.040 0.080 0.100];

%% Storage
nCases = length(separationList);
Cmatrices = zeros(4,4,nCases);
results = zeros(nCases,17);

for q = 1:nCases

    D = separationList(q);

    % Conductor centre positions along x-axis
    x = [-D/2-pitch/2, ...
         -D/2+pitch/2, ...
          D/2-pitch/2, ...
          D/2+pitch/2];

    % Potential coefficient matrix per unit length
    P = zeros(4,4);

    for i = 1:4
        for j = 1:4

            if i == j

                % Self potential coefficient
                P(i,i) = log(2*h/a)/(2*pi*eps);

            else

                rij = abs(x(i)-x(j));

                % Distance to image conductor
                rImage = sqrt(rij^2+(2*h)^2);

                P(i,j) = log(rImage/rij)/(2*pi*eps);

            end
        end
    end

    % Maxwell capacitance matrix per unit length
    Cprime = inv(P);

    % Convert per-unit-length matrix to total capacitance
    Ctotal = Cprime*L;

    % Numerical checks
    symmetryError = max(max(abs(Ctotal-Ctotal')));
    eigenvalues = eig(Ctotal);

    if symmetryError > 1e-12
        error('Capacitance matrix is not symmetric.');
    end

    if min(eigenvalues) <= 0
        error('Capacitance matrix is not positive definite.');
    end

    % Store
    Cmatrices(:,:,q) = Ctotal;

    results(q,1) = D*1000;

    results(q,2:17) = reshape(Ctotal*1e12,1,16);

    fprintf('\n====================================\n');
    fprintf('SEPARATION = %.1f mm\n',D*1000);
    fprintf('MAXWELL CAPACITANCE MATRIX (pF)\n');
    disp(Ctotal*1e12);

    fprintf('Symmetry error: %.3e\n',symmetryError);
    fprintf('Minimum eigenvalue: %.6e\n',min(eigenvalues));
end

%% Create output table
names = {'Separation_mm', ...
    'C11_pF','C12_pF','C13_pF','C14_pF', ...
    'C21_pF','C22_pF','C23_pF','C24_pF', ...
    'C31_pF','C32_pF','C33_pF','C34_pF', ...
    'C41_pF','C42_pF','C43_pF','C44_pF'};

ResultsTable = array2table(results,'VariableNames',names);

disp('STAGE 8B - MAXWELL MATRIX RESULTS');
disp(ResultsTable);

%% Plot cross-coupling coefficient magnitudes
figure;

coupling = zeros(nCases,4);

for q = 1:nCases
    C = Cmatrices(:,:,q)*1e12;

    coupling(q,:) = [-C(1,3), -C(1,4), ...
                     -C(2,3), -C(2,4)];
end

plot(separationList*1000,coupling,'-o','LineWidth',1.5);
grid on;

xlabel('Cable centre separation (mm)');
ylabel('Absolute cross-coupling coefficient (pF)');

legend('C13','C14','C23','C24','Location','best');
title('Maxwell matrix cross-coupling coefficients');

%% Export
outFolder = fullfile(pwd,'02_Simulation_Results');

if ~exist(outFolder,'dir')
    mkdir(outFolder);
end

writetable(ResultsTable,fullfile(outFolder, ...
    'EMC_GUARD_Stage8B_MaxwellMatrix.csv'));

fprintf('\nStage 8B completed successfully.\n');
