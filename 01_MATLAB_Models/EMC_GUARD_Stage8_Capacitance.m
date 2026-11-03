
%% EMC-GUARD | STAGE 8
% Preliminary multi-conductor capacitance geometry study
% Assumption: two unshielded, two-core insulated cables
% This is NOT a complete capacitance matrix or compliance model.

clear; clc; close all;

%% Constants
eps0 = 8.854187817e-12;

%% Cable geometry - preliminary assumptions
L = 0.450;                 % Parallel cable length (m)
a = 0.0005;                % Conductor radius (m)
corePitch = 0.006;         % Core centre spacing (m)
separationList = [0.020 0.040 0.080 0.100];

% Relative permittivity placeholder.
% Replace with actual cable insulation datasheet value.
er = 2.3;
eps = eps0*er;

%% Store results
Results = [];

for q = 1:length(separationList)

    D = separationList(q);

    % Four conductor centre positions:
    % Source cable: x = -D/2
    % Victim cable: x = +D/2

    x = [-D/2-corePitch/2, ...
         -D/2+corePitch/2, ...
          D/2-corePitch/2, ...
          D/2+corePitch/2];

    % Pairwise centre distances
    pairDistance = zeros(4,4);
    pairCapacitance = zeros(4,4);

    for i = 1:4
        for j = 1:4

            if i ~= j

                r = abs(x(i)-x(j));

                pairDistance(i,j) = r;

                % Simplified two-cylinder capacitance estimate.
                % Assumes homogeneous dielectric and isolated pair.
                if r > 2*a
                    CperLength = pi*eps / acosh(r/(2*a));
                    pairCapacitance(i,j) = CperLength*L;
                else
                    pairCapacitance(i,j) = NaN;
                end

            end
        end
    end

    % Cross-cable pair estimates
    C13 = pairCapacitance(1,3);
    C14 = pairCapacitance(1,4);
    C23 = pairCapacitance(2,3);
    C24 = pairCapacitance(2,4);

    Results = [Results; D*1000 C13*1e12 ...
        C14*1e12 C23*1e12 C24*1e12];

end

%% Display
ResultsTable = array2table(Results, ...
    'VariableNames', ...
    {'Separation_mm','C13_pF','C14_pF','C23_pF','C24_pF'});

disp('STAGE 8 - PRELIMINARY CROSS-CABLE CAPACITANCE');
disp(ResultsTable);

%% Plot
figure;
plot(Results(:,1),Results(:,2:5),'-o','LineWidth',1.5);
grid on;
xlabel('Cable centre separation (mm)');
ylabel('Pairwise capacitance estimate (pF)');
legend('C13','C14','C23','C24','Location','best');
title('Preliminary cross-cable capacitance estimates');

%% Export
outFolder = fullfile(pwd,'02_Simulation_Results');

if ~exist(outFolder,'dir')
    mkdir(outFolder);
end

writetable(ResultsTable, ...
    fullfile(outFolder,'EMC_GUARD_Stage8_Capacitance.csv'));

fprintf('\nStage 8 preliminary geometry study completed.\n');
