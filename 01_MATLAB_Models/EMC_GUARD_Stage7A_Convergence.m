
%% EMC-GUARD | STAGE 7A
% Time-step convergence test for the coupled lumped model

clear; clc; close all;

%% Fixed parameters
Vdc = 5;
fpwm = 5000;
duty = 0.50;

Rs = 1;
Rload = 47;
Ls = 2e-6;

Rv = 10;
Lv = 10e-6;
Cg = 100e-12;
Rt = 10e3;

mu0 = 4*pi*1e-7;
eps0 = 8.854187817e-12;

cableLength = 0.450;
radius = 0.0005;
pitch = 0.006;
D = 0.020;

%% Calculate signed mutual inductance
sourceX = [-pitch/2 pitch/2];
victimX = [D-pitch/2 D+pitch/2];

sourceSign = [1 -1];
victimSign = [1 -1];

M = 0;

for i = 1:2
    for j = 1:2
        r = abs(sourceX(i)-victimX(j));

        Mp = mu0/(2*pi)*( ...
            cableLength*asinh(cableLength/r) ...
            -sqrt(cableLength^2+r^2)+r);

        M = M + sourceSign(i)*victimSign(j)*Mp;
    end
end

Cm = pi*eps0*cableLength/acosh(D/(2*radius));

Lmatrix = [Ls M; M Lv];

if min(eig(Lmatrix)) <= 0
    error('Inductance matrix is not positive definite.');
end

%% Time-step sweep
dtList = [10e-9 5e-9 2.5e-9];
T = 2e-3;

Vpp = zeros(3,1);
Vrms = zeros(3,1);

for q = 1:length(dtList)

    dt = dtList(q);
    t = (0:dt:T)';
    N = length(t);

    period = 1/fpwm;
    tr = 100e-9;
    tf = 100e-9;

    phase = mod(t,period);
    highEnd = duty*period;

    rise = min(phase/tr,1);
    fall = min(max((highEnd-phase)/tf,0),1);

    pwm = min(rise,fall);
    Vs = Vdc*pwm;

    x = zeros(3,1);
    V = zeros(N,1);

    for k = 1:N-1

        u1 = Vs(k);
        um = 0.5*(Vs(k)+Vs(k+1));
        u4 = Vs(k+1);

        k1 = rhs7a(x,u1,Lmatrix,Rs,Rload,Rv,Cg,Cm,Rt);
        k2 = rhs7a(x+0.5*dt*k1,um,Lmatrix,Rs,Rload,Rv,Cg,Cm,Rt);
        k3 = rhs7a(x+0.5*dt*k2,um,Lmatrix,Rs,Rload,Rv,Cg,Cm,Rt);
        k4 = rhs7a(x+dt*k3,u4,Lmatrix,Rs,Rload,Rv,Cg,Cm,Rt);

        x = x+(dt/6)*(k1+2*k2+2*k3+k4);
        V(k+1) = x(3);
    end

    if any(~isfinite(V))
        error('Non-finite result at dt = %.3g',dt);
    end

    idx = t >= (T-0.5e-3);
    steady = V(idx);

    Vpp(q) = max(steady)-min(steady);
    Vrms(q) = sqrt(mean(steady.^2));

    fprintf('Completed dt = %.2f ns\n',dt*1e9);
end

%% Convergence comparison
ChangeVpp = [NaN; abs(diff(Vpp))./Vpp(2:end)*100];
ChangeRMS = [NaN; abs(diff(Vrms))./Vrms(2:end)*100];

Results = table(dtList'*1e9,Vpp,Vrms,...
    ChangeVpp,ChangeRMS,...
    'VariableNames',{'TimeStep_ns','Vpp_V','RMS_V',...
    'ChangeVpp_percent','ChangeRMS_percent'});

disp('STAGE 7A CONVERGENCE RESULTS');
disp(Results);

%% Export
outFolder = fullfile(pwd,'02_Simulation_Results');

if ~exist(outFolder,'dir')
    mkdir(outFolder);
end

writetable(Results,fullfile(outFolder,...
    'EMC_GUARD_Stage7A_Convergence.csv'));

fprintf('\nStage 7A completed.\n');

%% Model differential equations
function dx = rhs7a(x,Vs,Lmatrix,Rs,Rload,Rv,Cg,Cm,Rt)

    is = x(1);
    iv = x(2);
    v = x(3);

    b = [Vs-(Rs+Rload)*is;
         -Rv*iv-v];

    di = Lmatrix\b;

    dis = di(1);
    div = di(2);

    dv = (iv+Cm*Rload*dis-v/Rt)/(Cg+Cm);

    dx = [dis;div;dv];
end
