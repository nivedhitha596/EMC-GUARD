
%% EMC-GUARD | STAGE 8D
% Solver maximum-step convergence test
% Tests the 20 mm cable separation.
% Uses the corrected Stage 8C circuit topology.

clear; clc; close all;

%% 1. Constants
eps0 = 8.854187817e-12;
mu0 = 4*pi*1e-7;

%% 2. Geometry
L = 0.450;
a = 0.0005;
pitch = 0.006;
h = 0.050;
er = 2.3;
eps = eps0*er;

D = 0.020;  % 20 mm test case

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

%% 4. Build four-conductor capacitance matrix
x = [-D/2-pitch/2, ...
     -D/2+pitch/2, ...
      D/2-pitch/2, ...
      D/2+pitch/2];

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

Cmatrix = (P \ eye(4))*L;
Cmatrix = 0.5*(Cmatrix+Cmatrix');

%% 5. Build signed mutual inductance
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

Lmatrix = [Ls M; M Lv];

if min(eig(Cmatrix)) <= 0 || min(eig(Lmatrix)) <= 0
    error('Invalid capacitance or inductance matrix.');
end

%% 6. Circuit matrices
B = [1 0;
    -1 0;
     0 1;
     0 -1];

Rloop = [Rs; Rv];

Gmatrix = B*diag([1/Rload,1/Rt])*B';

%% 7. Simulation settings
T = 2e-3;

% Common output interval for all three simulations
dtOutput = 10e-9;
t = (0:dtOutput:T)';

x0 = zeros(6,1);

% Internal maximum solver steps
maxStepList = [20e-9 10e-9 5e-9];

nCases = length(maxStepList);

Vpp = zeros(nCases,1);
Vrms = zeros(nCases,1);
Vmax = zeros(nCases,1);

%% 8. Run convergence tests
for q = 1:nCases

    maxStep = maxStepList(q);

    options = odeset( ...
        'RelTol',1e-6, ...
        'AbsTol',[1e-10 1e-10 1e-7 1e-7 1e-7 1e-7], ...
        'MaxStep',maxStep);

    [tSol,xSol] = ode15s( ...
        @(tt,xx) rhs8d(tt,xx,Lmatrix,Cmatrix, ...
        B,Rloop,Gmatrix,Vdc,fpwm,duty), ...
        t,x0,options);

    Vvictim = xSol(:,5)-xSol(:,6);

    if any(~isfinite(Vvictim))
        error('Non-finite output at maximum step %.2f ns', ...
            maxStep*1e9);
    end

    idx = tSol >= (T-0.5e-3);
    steady = Vvictim(idx);

    Vpp(q) = max(steady)-min(steady);
    Vrms(q) = sqrt(mean(steady.^2));
    Vmax(q) = max(abs(steady));

    fprintf('\nCompleted maximum step = %.2f ns\n', ...
        maxStep*1e9);

end

%% 9. Calculate relative changes
ChangeVpp = NaN(nCases,1);
ChangeRMS = NaN(nCases,1);

for q = 2:nCases

    ChangeVpp(q) = abs(Vpp(q)-Vpp(q-1))/Vpp(q)*100;

    ChangeRMS(q) = abs(Vrms(q)-Vrms(q-1))/Vrms(q)*100;

end

%% 10. Display results
Results = table( ...
    maxStepList'*1e9, ...
    Vpp, Vrms, Vmax, ...
    ChangeVpp, ChangeRMS, ...
    'VariableNames', ...
    {'MaximumStep_ns','Vpp_V','RMS_V','MaxAbs_V', ...
    'ChangeVpp_percent','ChangeRMS_percent'});

disp('STAGE 8D - CONVERGENCE RESULTS');
disp(Results);

%% 11. Export
outFolder = fullfile(pwd,'02_Simulation_Results');

if ~exist(outFolder,'dir')
    mkdir(outFolder);
end

writetable(Results,fullfile(outFolder, ...
    'EMC_GUARD_Stage8D_Convergence.csv'));

fprintf('\nSTAGE 8D COMPLETED.\n');

%% 12. Differential equations
function dx = rhs8d(t,x,Lmatrix,Cmatrix, ...
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
