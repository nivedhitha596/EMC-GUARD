
%% EMC-GUARD | STAGE 8C
% Integrated four-conductor capacitance + coupled-loop model
% Preliminary analytical model, not experimentally validated.

clear; clc; close all;

%% 1. Constants and assumptions
eps0 = 8.854187817e-12;
mu0 = 4*pi*1e-7;

epsr = 2.3;
eps = eps0*epsr;

L = 0.450;
a = 0.0005;
pitch = 0.006;
h = 0.050;

separationList = [0.020 0.040 0.080 0.100];

%% 2. Electrical parameters
Vdc = 5;
fpwm = 5000;
duty = 0.50;

Rs = 1;
Rload = 47;
Ls = 2e-6;

Rv = 10;
Rt = 10e3;
Lv = 10e-6;

LmatrixBase = [Ls 0; 0 Lv];

%% 3. Loop-to-conductor connection matrix
B = [ 1  0;
     -1  0;
      0  1;
      0 -1];

Rloop = [Rs; Rv];

Gmatrix = B*diag([1/Rload, 1/Rt])*B';
%% 4. Simulation settings
dt = 10e-9;
T = 2e-3;

t = (0:dt:T)';
N = length(t);

period = 1/fpwm;
tr = 100e-9;
tf = 100e-9;

phase = mod(t,period);
highEnd = duty*period;

rise = min(phase/tr,1);
fall = min(max((highEnd-phase)/tf,0),1);

PWM = min(rise,fall);
Vs = Vdc*PWM;

%% 5. Result storage
nCases = length(separationList);

Vpp = zeros(nCases,1);
Vrms = zeros(nCases,1);
Vmax = zeros(nCases,1);

Mvalues = zeros(nCases,1);
C13values = zeros(nCases,1);
C24values = zeros(nCases,1);

allCapacitanceMatrices = zeros(4,4,nCases);

%% 6. Separation sweep
for q = 1:nCases

    D = separationList(q);

    %% 6A. Four-conductor geometry
    x = [-D/2-pitch/2, ...
         -D/2+pitch/2, ...
          D/2-pitch/2, ...
          D/2+pitch/2];

    %% 6B. Potential coefficient matrix
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

    %% 6C. Maxwell capacitance matrix
    Cmatrix = (P \ eye(4))*L;

    Cmatrix = 0.5*(Cmatrix+Cmatrix');

    if min(eig(Cmatrix)) <= 0
        error('Capacitance matrix is not positive definite.');
    end

    allCapacitanceMatrices(:,:,q) = Cmatrix;

    C13values(q) = Cmatrix(1,3);
    C24values(q) = Cmatrix(2,4);

    %% 6D. Signed mutual inductance
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

    %% 6E. Coupled inductance matrix
    Lmatrix = [Ls M; M Lv];

    if min(eig(Lmatrix)) <= 0
        error('Inductance matrix is not positive definite.');
    end

    %% 6F. Stiff-system integration using ode15s

x0 = zeros(6,1);

options = odeset( ...
    'RelTol',1e-6, ...
    'AbsTol',[1e-10 1e-10 1e-7 1e-7 1e-7 1e-7], ...
    'MaxStep',tr/5);

[tSol,xSol] = ode15s( ...
    @(tt,xx) rhs8c_stiff(tt,xx,Lmatrix,Cmatrix, ...
    B,Rloop,Gmatrix,Vdc,period,duty,tr,tf), ...
    t,x0,options);


Vvictim = xSol(:,5)-xSol(:,6);
    %% 6G. Numerical checks
    if any(~isfinite(Vvictim))
        error('Non-finite result at separation %.1f mm',D*1000);
    end

    idx = t >= (T-0.5e-3);
    steady = Vvictim(idx);

    Vpp(q) = max(steady)-min(steady);
    Vrms(q) = sqrt(mean(steady.^2));
    Vmax(q) = max(abs(steady));

    fprintf('\nCompleted separation = %.1f mm\n',D*1000);
    fprintf('M = %.5f nH\n',M*1e9);
    fprintf('C13 = %.5f pF\n',Cmatrix(1,3)*1e12);
    fprintf('Victim Vpp = %.8f V\n',Vpp(q));
    fprintf('Victim RMS = %.8f V\n',Vrms(q));

end

%% 7. Results table
Results = table( ...
    separationList'*1000, ...
    Mvalues*1e9, ...
    C13values*1e12, ...
    C24values*1e12, ...
    Vpp, Vrms, Vmax, ...
    'VariableNames', ...
    {'Separation_mm','M_nH','C13_pF','C24_pF', ...
     'Victim_Vpp_V','Victim_RMS_V','Victim_MaxAbs_V'});

disp('STAGE 8C - INTEGRATED MODEL RESULTS');
disp(Results);

%% 8. Plot victim response versus separation
figure;

plot(separationList*1000,Vpp,'-o','LineWidth',1.5);

grid on;
xlabel('Cable separation (mm)');
ylabel('Victim peak-to-peak voltage (V)');
title('Stage 8C - Integrated coupling prediction');

%% 9. Export
outFolder = fullfile(pwd,'02_Simulation_Results');

if ~exist(outFolder,'dir')
    mkdir(outFolder);
end

writetable(Results,fullfile(outFolder, ...
    'EMC_GUARD_Stage8C_IntegratedResults.csv'));

save(fullfile(outFolder, ...
    'EMC_GUARD_Stage8C_CapacitanceMatrices.mat'), ...
    'allCapacitanceMatrices','separationList');

fprintf('\nSTAGE 8C COMPLETED.\n');

function dx = rhs8c_stiff(t,x,Lmatrix,Cmatrix, ...
    B,Rloop,Gmatrix,Vdc,period,duty,tr,tf)

    i = x(1:2);
    v = x(3:6);

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