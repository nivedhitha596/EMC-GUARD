
%% EMC-PREDICT | Stage 2 - Coupling Sensitivity Analysis
% Independent variation of mutual inductance and capacitance.
% Illustrative parameters, not measured cable data.

clear; clc; close all;

%% Source
Vdc = 5;
fpwm = 5000;
duty = 0.50;
Rload = 47;

%% Time
dt = 10e-9;
T = 5e-3;
t = (0:dt:T)';
N = length(t);

%% PWM with finite rise and fall times
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

%% Victim parameters
Rv = 10;
Lv = 10e-6;
Cv = 100e-12;
Rt = 10e3;

%% Parameter sweeps
M_values = [0.05 0.10 0.15 0.20 0.30]*1e-6;
Cm_values = [5 10 20 40 80]*1e-12;

M_base = 0.15e-6;
Cm_base = 20e-12;

M_results = zeros(length(M_values),2);
Cm_results = zeros(length(Cm_values),2);

%% Sweep mutual inductance
for j = 1:length(M_values)

    [Vpp,Vrms] = runCoupledModel( ...
        M_values(j),Cm_base,dIs,dVs,dt, ...
        Rv,Lv,Cv,Rt);

    M_results(j,:) = [Vpp,Vrms];

end

%% Sweep mutual capacitance
for j = 1:length(Cm_values)

    [Vpp,Vrms] = runCoupledModel( ...
        M_base,Cm_values(j),dIs,dVs,dt, ...
        Rv,Lv,Cv,Rt);

    Cm_results(j,:) = [Vpp,Vrms];

end

%% Display results
M_table = table(M_values'*1e6, ...
    M_results(:,1),M_results(:,2), ...
    'VariableNames', ...
    {'Mutual_Inductance_uH','Victim_Vpp_V','Victim_RMS_V'});

Cm_table = table(Cm_values'*1e12, ...
    Cm_results(:,1),Cm_results(:,2), ...
    'VariableNames', ...
    {'Mutual_Capacitance_pF','Victim_Vpp_V','Victim_RMS_V'});

disp('MUTUAL INDUCTANCE SENSITIVITY');
disp(M_table);

disp('MUTUAL CAPACITANCE SENSITIVITY');
disp(Cm_table);

%% Plot 1: Mutual inductance
figure('Name','EMC-PREDICT M Sensitivity');

plot(M_values*1e6,M_results(:,1), ...
    '-o','LineWidth',1.8);

xlabel('Mutual Inductance (\muH)');
ylabel('Victim Voltage Vpp (V)');
title('Victim Voltage vs Mutual Inductance');
grid on;

%% Plot 2: Mutual capacitance
figure('Name','EMC-PREDICT Cm Sensitivity');

plot(Cm_values*1e12,Cm_results(:,1), ...
    '-s','LineWidth',1.8);

xlabel('Mutual Capacitance (pF)');
ylabel('Victim Voltage Vpp (V)');
title('Victim Voltage vs Mutual Capacitance');
grid on;

%% Export simulation results
writetable(M_table,'EMC_PREDICT_M_Sensitivity.csv');
writetable(Cm_table,'EMC_PREDICT_Cm_Sensitivity.csv');

fprintf('\nStage 2 completed successfully.\n');

%% Local simulation function
function [Vpp,Vrms] = runCoupledModel( ...
    M,Cm,dIs,dVs,dt,Rv,Lv,Cv,Rt)

    N = length(dIs);

    Vind = M*dIs;
    Ic = Cm*dVs;

    x = zeros(2,1);
    Vvictim = zeros(N,1);

    for k = 1:N-1

        u1 = [Vind(k); Ic(k)];

        um = 0.5*([Vind(k); Ic(k)] + ...
                  [Vind(k+1); Ic(k+1)]);

        k1 = rhs(x,u1,Rv,Lv,Cv,Rt);
        k2 = rhs(x+0.5*dt*k1,um,Rv,Lv,Cv,Rt);
        k3 = rhs(x+0.5*dt*k2,um,Rv,Lv,Cv,Rt);

        k4 = rhs(x+dt*k3, ...
            [Vind(k+1);Ic(k+1)],Rv,Lv,Cv,Rt);

        x = x + (dt/6)*(k1+2*k2+2*k3+k4);

        Vvictim(k+1) = x(2);

    end

    if any(~isfinite(Vvictim))
        error('Non-finite result detected.');
    end

    % Use last 1 ms for consistent comparison
    steady = Vvictim(round(4e-3/dt)+1:end);

    Vpp = max(steady)-min(steady);
    Vrms = sqrt(mean(steady.^2));

end

function dx = rhs(x,u,Rv,Lv,Cv,Rt)

    i = x(1);
    v = x(2);

    Vind = u(1);
    Ic = u(2);

    di = (Vind-Rv*i-v)/Lv;
    dv = (i+Ic-v/Rt)/Cv;

    dx = [di;dv];

end
