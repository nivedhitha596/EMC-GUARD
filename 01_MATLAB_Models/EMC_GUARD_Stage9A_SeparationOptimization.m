
%% EMC-GUARD | STAGE 9A
% Separation-based cable routing optimisation
% Uses corrected Stage 8C circuit topology.

clear; clc; close all;

%% 1. Constants
eps0 = 8.854187817e-12;
mu0 = 4*pi*1e-7;

%% 2. Cable geometry
L = 0.450;
a = 0.0005;
pitch = 0.006;
h = 0.050;
er = 2.3;
eps = eps0*er;

%% 3. Electrical parameters
Vdc = 5;
fpwm = 5000;
duty = 0.50;

Rs = 1;
Rload = 47;
Ls = 2e-6;

Rv = 10;
Rt = 10e3;
Lv = 10e-6;

%% 4. Circuit matrices
B = [1 0;
    -1 0;
     0 1;
     0 -1];

Rloop = [Rs; Rv];

Gmatrix = B*diag([1/Rload,1/Rt])*B';

%% 5. Optimisation search range
separationList = (20:10:100)*1e-3;

nCases = length(separationList);

%% 6. Simulation settings
T = 2e-3;
dtOutput = 10e-9;
t = (0:dtOutput:T)';

x0 = zeros(6,1);

options = odeset( ...
    'RelTol',1e-6, ...
    'AbsTol',[1e-10 1e-10 1e-7 1e-7 1e-7 1e-7], ...
    'MaxStep',5e-9);

%% 7. Result storage
Vpp = zeros(nCases,1);
Vrms = zeros(nCases,1);
Vmax = zeros(nCases,1);

Mvalues = zeros(nCases,1);
C13values = zeros(nCases,1);

%% 8. Evaluate every separation
for q = 1:nCases

    D = separationList(q);

    %% 8A. Four conductor positions
    x = [-D/2-pitch/2, ...
         -D/2+pitch/2, ...
          D/2-pitch/2, ...
          D/2+pitch/2];

    %% 8B. Potential coefficient matrix
    P = zeros(4,4);

    for i = 1:4
        for j = 1:4

            if i == j

                P(i,i) = log(2*h/a)/(2*pi*eps);

            else

                rij = abs(x(i)-x(j));
                rImage = sqrt(rij^2+(2*h)^2);

                P(i,j) = log(rImage/rij)/(2*pi*eps);

            end
        end
    end

    %% 8C. Maxwell capacitance matrix
    Cmatrix = (P \ eye(4))*L;
    Cmatrix = 0.5*(Cmatrix+Cmatrix');

    if min(eig(Cmatrix)) <= 0
        error('Invalid capacitance matrix.');
    end

    C13values(q) = Cmatrix(1,3);

    %% 8D. Signed mutual inductance
    sourceX = [-pitch/2 pitch/2];
    victimX = [D-pitch/2 D+pitch/2];

    sourceSign = [1 -1];
    victimSign = [1 -1];

    M = 0;

    for i = 1:2
        for j = 1:2

            r = abs(sourceX(i)-victimX(j));

            Mp = mu0/(2*pi)*( ...
                L*asinh(L/r)-sqrt(L^2+r^2)+r);

            M = M + sourceSign(i)*victimSign(j)*Mp;

        end
    end

    Mvalues(q) = M;

    %% 8E. Coupled inductance matrix
    Lmatrix = [Ls M; M Lv];

    if min(eig(Lmatrix)) <= 0
        error('Invalid inductance matrix.');
    end

    %% 8F. Solve circuit
    [tSol,xSol] = ode15s( ...
        @(tt,xx) rhs9a(tt,xx,Lmatrix,Cmatrix, ...
        B,Rloop,Gmatrix,Vdc,fpwm,duty), ...
        t,x0,options);

    Vvictim = xSol(:,5)-xSol(:,6);

    if any(~isfinite(Vvictim))
        error('Non-finite result at %.1f mm.',D*1000);
    end

    %% 8G. Calculate steady-state performance
    idx = tSol >= (T-0.5e-3);
    steady = Vvictim(idx);

    Vpp(q) = max(steady)-min(steady);
    Vrms(q) = sqrt(mean(steady.^2));
    Vmax(q) = max(abs(steady));

    fprintf('Completed %.1f mm | Vpp = %.6f mV | RMS = %.6f mV\n', ...
        D*1000,Vpp(q)*1000,Vrms(q)*1000);

end

%% 9. Identify minimum predicted RMS
[bestRMS,bestIndex] = min(Vrms);

bestSeparation = separationList(bestIndex);

fprintf('\n====================================\n');
fprintf('STAGE 9A OPTIMISATION RESULT\n');
fprintf('====================================\n');

fprintf('Best tested separation: %.1f mm\n', ...
    bestSeparation*1000);

fprintf('Minimum predicted RMS: %.6f mV\n', ...
    bestRMS*1000);

fprintf('Corresponding Vpp: %.6f mV\n', ...
    Vpp(bestIndex)*1000);

%% 10. Results table
Results = table( ...
    separationList'*1000, ...
    Mvalues*1e9, ...
    C13values*1e12, ...
    Vpp*1000, ...
    Vrms*1000, ...
    Vmax*1000, ...
    'VariableNames', ...
    {'Separation_mm','M_nH','C13_pF', ...
    'Victim_Vpp_mV','Victim_RMS_mV','Victim_MaxAbs_mV'});

disp(Results);

%% 11. Plot optimisation results
figure;

plot(separationList*1000,Vrms*1000,'-o', ...
    'LineWidth',1.5);

hold on;

plot(bestSeparation*1000,bestRMS*1000,'p', ...
    'MarkerSize',12);

grid on;

xlabel('Cable separation (mm)');
ylabel('Predicted victim RMS (mV)');

title('EMC-GUARD Stage 9A - Separation optimisation');

legend('Predicted RMS','Best tested configuration', ...
    'Location','best');

%% 12. Export
outFolder = fullfile(pwd,'02_Simulation_Results');

if ~exist(outFolder,'dir')
    mkdir(outFolder);
end

writetable(Results,fullfile(outFolder, ...
    'EMC_GUARD_Stage9A_Optimization.csv'));

fprintf('\nSTAGE 9A COMPLETED.\n');

%% 13. Circuit differential equations
function dx = rhs9a(t,x,Lmatrix,Cmatrix, ...
    B,Rloop,Gmatrix,Vdc,fpwm,duty)

    i = x(1:2);
    v = x(3:6);

    period = 1/fpwm;
    tr = 100e-9;
    tf = 100e-9;

    phase = mod(t,period);

    rise = min(phase/tr,1);
    fall = min(max((duty*period-phase)/tf,0),1);

    pwm = min(rise,fall);
    Vs = Vdc*pwm;

    e = [Vs;0];

    di = Lmatrix \ (e-Rloop.*i-B'*v);
    dv = Cmatrix \ (B*i-Gmatrix*v);

    dx = [di;dv];

end
