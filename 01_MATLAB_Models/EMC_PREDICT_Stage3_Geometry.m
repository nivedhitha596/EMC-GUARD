
%% EMC-PREDICT | Stage 3
% Geometry-based preliminary coupling estimation
% Two-core source cable beside two-core victim cable.
% Unshielded, parallel, straight conductors.

clear; clc; close all;

%% Physical constants
mu0 = 4*pi*1e-7;
eps0 = 8.854187817e-12;

%% Cable geometry
cableLength = 0.450;       % metres
conductorRadius = 0.0005;  % metres
pairPitch = 0.006;         % metres

% Cable centre-to-centre separation
spacing = [0.020 0.040 0.080 0.100];

%% Conductor positions
% Source pair: outgoing and return
sourceX = [-pairPitch/2, pairPitch/2];
sourceSign = [1, -1];

%% Results
M_values = zeros(length(spacing),1);
Cm_values = zeros(length(spacing),1);

for k = 1:length(spacing)

    D = spacing(k);

    % Victim pair positions
    victimX = [D-pairPitch/2, D+pairPitch/2];
    victimSign = [1, -1];

    %% Mutual inductance of two complete loops
    Mloop = 0;

    for i = 1:2
        for j = 1:2

            r = abs(sourceX(i)-victimX(j));

            Mpartial = mu0/(2*pi) * ...
                (cableLength*asinh(cableLength/r) ...
                - sqrt(cableLength^2+r^2)+r);

            Mloop = Mloop + ...
                sourceSign(i)*victimSign(j)*Mpartial;

        end
    end

    M_values(k) = abs(Mloop);

    %% Preliminary capacitive coupling estimate
    % Approximation for two equal parallel cylindrical conductors
    % in a homogeneous air environment.

    d = D;

    CperLength = pi*eps0 / acosh(d/(2*conductorRadius));

    Cm_values(k) = CperLength*cableLength;

end

%% Display results
Results = table( ...
    spacing'*1000, ...
    M_values*1e6, ...
    Cm_values*1e12, ...
    'VariableNames', ...
    {'Cable_Separation_mm', ...
     'Mutual_Inductance_uH', ...
     'Estimated_Coupling_Capacitance_pF'});

disp('EMC-PREDICT STAGE 3 - GEOMETRY RESULTS');
disp(Results);

%% Save dataset
writetable(Results,'EMC_PREDICT_Stage3_Geometry.csv');

%% Plot mutual inductance
figure;
plot(spacing*1000,M_values*1e6,'-o','LineWidth',1.8);
xlabel('Cable centre separation (mm)');
ylabel('Mutual inductance magnitude (\muH)');
title('Estimated Loop Mutual Inductance vs Separation');
grid on;

%% Plot capacitance
figure;
plot(spacing*1000,Cm_values*1e12,'-s','LineWidth',1.8);
xlabel('Cable centre separation (mm)');
ylabel('Estimated coupling capacitance (pF)');
title('Approximate Capacitive Coupling vs Separation');
grid on;

fprintf('\nStage 3 geometry estimation completed.\n');
fprintf('Results saved to EMC_PREDICT_Stage3_Geometry.csv\n');
