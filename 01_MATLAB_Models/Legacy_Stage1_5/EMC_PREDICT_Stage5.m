
%% EMC-PREDICT | STAGE 5
% Separate inductive and capacitive coupling contributions
% Preliminary model only - not experimentally validated.

clear; clc; close all;

%% Source parameters
Vdc = 5;
fpwm = 5000;
duty = 0.50;
Rload = 47;

%% Simulation
dt = 2.5e-9;
T = 5e-3;
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
Is = Vs/Rload;

dVs = gradient(Vs,dt);
dIs = gradient(Is,dt);

%% Geometry
mu0 = 4*pi*1e-7;
eps0 = 8.854187817e-12;

cableLength = 0.450;
conductorRadius = 0.0005;
pairPitch = 0.006;

spacing = [0.020 0.040 0.080 0.100];

%% Victim circuit
Rv = 10;
Lv = 10e-6;
Cv = 100e-12;
Rt = 10e3;

%% Results
nCases = length(spacing);

M_nH = zeros(nCases,1);
Cm_pF = zeros(nCases,1);

Vpp_ind = zeros(nCases,1);
Vpp_cap = zeros(nCases,1);
Vpp_combined = zeros(nCases,1);

RMS_ind = zeros(nCases,1);
RMS_cap = zeros(nCases,1);
RMS_combined = zeros(nCases,1);

for n = 1:nCases

    D = spacing(n);

    sourceX = [-pairPitch/2, pairPitch/2];
    victimX = [D-pairPitch/2, D+pairPitch/2];

    sourceSign = [1,-1];
    victimSign = [1,-1];

    %% Mutual inductance
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

    M = abs(Mloop);

    %% Preliminary capacitance estimate
    Cm = pi*eps0*cableLength / ...
        acosh(D/(2*conductorRadius));

    M_nH(n) = M*1e9;
    Cm_pF(n) = Cm*1e12;

    %% Coupling sources
    Vind = M*dIs;
    Ic = Cm*dVs;

    %% Run three separate cases
    [v1, r1] = simulate_case(Vind, zeros(N,1), ...
        t, dt, Rv, Lv, Cv, Rt);

    [v2, r2] = simulate_case(zeros(N,1), Ic, ...
        t, dt, Rv, Lv, Cv, Rt);

    [v3, r3] = simulate_case(Vind, Ic, ...
        t, dt, Rv, Lv, Cv, Rt);

    Vpp_ind(n) = v1;
    Vpp_cap(n) = v2;
    Vpp_combined(n) = v3;

    RMS_ind(n) = r1;
    RMS_cap(n) = r2;
    RMS_combined(n) = r3;

    fprintf('Completed: %.0f mm\n',D*1000);
end

%% Results table
Results = table( ...
    spacing'*1000, M_nH, Cm_pF, ...
    Vpp_ind, Vpp_cap, Vpp_combined, ...
    RMS_ind, RMS_cap, RMS_combined, ...
    'VariableNames', { ...
    'Spacing_mm','M_nH','Cm_pF', ...
    'Vpp_Inductive_V','Vpp_Capacitive_V', ...
    'Vpp_Combined_V','RMS_Inductive_V', ...
    'RMS_Capacitive_V','RMS_Combined_V'});

disp('EMC-PREDICT STAGE 5 RESULTS');
disp(Results);

writetable(Results,'EMC_PREDICT_Stage5_Coupling_Separation.csv');

%% Plot peak-to-peak contributions
figure;
plot(spacing*1000,Vpp_ind,'-o','LineWidth',1.6);
hold on;
plot(spacing*1000,Vpp_cap,'-s','LineWidth',1.6);
plot(spacing*1000,Vpp_combined,'-d','LineWidth',1.6);
xlabel('Cable separation (mm)');
ylabel('Victim voltage Vpp (V)');
title('Inductive vs capacitive coupling');
legend('Inductive only','Capacitive only','Combined');
grid on;

fprintf('\nStage 5 completed. Results exported.\n');

%% Simulation function
function [vpp,vrms] = simulate_case(Vind,Ic,t,dt,Rv,Lv,Cv,Rt)

    N = length(t);
    x = zeros(2,1);
    V = zeros(N,1);

    for k = 1:N-1

        u1 = [Vind(k); Ic(k)];
        um = 0.5*([Vind(k);Ic(k)] + ...
                  [Vind(k+1);Ic(k+1)]);

        k1 = rhs(x,u1,Rv,Lv,Cv,Rt);
        k2 = rhs(x+0.5*dt*k1,um,Rv,Lv,Cv,Rt);
        k3 = rhs(x+0.5*dt*k2,um,Rv,Lv,Cv,Rt);
        k4 = rhs(x+dt*k3, ...
            [Vind(k+1);Ic(k+1)],Rv,Lv,Cv,Rt);

        x = x + (dt/6)*(k1+2*k2+2*k3+k4);
        V(k+1) = x(2);
    end

    if any(~isfinite(V))
        error('Non-finite simulation result.');
    end

    idx = t >= 4e-3;
    steady = V(idx);

    vpp = max(steady)-min(steady);
    vrms = sqrt(mean(steady.^2));
end

function dx = rhs(x,u,Rv,Lv,Cv,Rt)

    i = x(1);
    v = x(2);

    di = (u(1)-Rv*i-v)/Lv;
    dv = (i+u(2)-v/Rt)/Cv;

    dx = [di;dv];
end
