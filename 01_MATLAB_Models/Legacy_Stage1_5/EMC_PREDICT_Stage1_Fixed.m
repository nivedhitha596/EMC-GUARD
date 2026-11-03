
%% EMC-PREDICT | Stage 1 - Numerically corrected model
% Simplified coupled-cable transient simulation
% All initial coupling parameters are illustrative assumptions.

clear; clc; close all;

%% 1. Source parameters
Vdc = 5;
fpwm = 5000;
duty = 0.50;
Rload = 47;

%% 2. Simulation settings
dt = 10e-9;
T = 5e-3;
t = (0:dt:T)';
N = length(t);

%% 3. PWM with finite switching transitions
period = 1/fpwm;
tr = 100e-9;
tf = 100e-9;

phase = mod(t,period);
highEnd = duty*period;

rise = min(phase/tr,1);
fall = min(max((highEnd-phase)/tf,0),1);

pwm = min(rise,fall);
Vs = Vdc*pwm;

%% 4. Source current
Is = Vs/Rload;

%% 5. Coupling parameters
% Illustrative values: not extracted or measured.
M = 0.15e-6;
Cm = 20e-12;

Vind = M*gradient(Is,dt);
Ic = Cm*gradient(Vs,dt);

%% 6. Victim cable parameters
Rv = 10;
Lv = 10e-6;
Cv = 100e-12;
Rt = 10e3;

%% 7. RK4 integration
% State vector:
% x(1) = victim inductor current
% x(2) = victim node voltage

x = zeros(2,1);

Ivictim = zeros(N,1);
Vvictim = zeros(N,1);

for k = 1:N-1

    u1 = [Vind(k); Ic(k)];
    um = 0.5*([Vind(k); Ic(k)] + ...
              [Vind(k+1); Ic(k+1)]);

    k1 = victim_rhs(x,u1,Rv,Lv,Cv,Rt);

    k2 = victim_rhs(x + 0.5*dt*k1, ...
                    um,Rv,Lv,Cv,Rt);

    k3 = victim_rhs(x + 0.5*dt*k2, ...
                    um,Rv,Lv,Cv,Rt);

    k4 = victim_rhs(x + dt*k3, ...
                    [Vind(k+1); Ic(k+1)], ...
                    Rv,Lv,Cv,Rt);

    x = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);

    Ivictim(k+1) = x(1);
    Vvictim(k+1) = x(2);
end

%% 8. Validate numerical results
if any(~isfinite(Vvictim))
    error('Numerical instability detected. Check parameters and time step.');
end

%% 9. Calculate output metrics
Vpp = max(Vvictim)-min(Vvictim);
Vrms = sqrt(mean(Vvictim.^2));
Vpeak = max(abs(Vvictim));

fprintf('\nEMC-PREDICT STAGE 1 - FIXED MODEL\n');
fprintf('Victim voltage peak-to-peak: %.8f V\n',Vpp);
fprintf('Victim voltage RMS: %.8f V\n',Vrms);
fprintf('Maximum absolute victim voltage: %.8f V\n',Vpeak);
fprintf('Finite-value validation: PASSED\n');

%% 10. Plot results
figure('Name','EMC-PREDICT Stage 1');

subplot(4,1,1);
plot(t(1:100:end)*1000,Vs(1:100:end));
ylabel('V (V)');
title('PWM source');
grid on;

subplot(4,1,2);
plot(t(1:100:end)*1000,Is(1:100:end));
ylabel('I (A)');
title('Source current');
grid on;

subplot(4,1,3);
plot(t(1:100:end)*1000,Vind(1:100:end));
ylabel('V (V)');
title('Inductive coupling excitation');
grid on;

subplot(4,1,4);
plot(t(1:100:end)*1000,Vvictim(1:100:end));
xlabel('Time (ms)');
ylabel('V (V)');
title('Predicted victim voltage');
grid on;

%% Local function
function dx = victim_rhs(x,u,Rv,Lv,Cv,Rt)

    i = x(1);
    v = x(2);

    Vind = u(1);
    Ic = u(2);

    di = (Vind - Rv*i - v)/Lv;
    dv = (i + Ic - v/Rt)/Cv;

    dx = [di; dv];
end
%% EMC-PREDICT - Steady state diagnostic

idx = t >= 4e-3;

Vsteady = Vvictim(idx);
tsteady = t(idx);

Vpp_steady = max(Vsteady) - min(Vsteady);
Vrms_steady = sqrt(mean(Vsteady.^2));
Vpeak_steady = max(abs(Vsteady));

fprintf('\nSTEADY STATE RESULTS (LAST 1 ms)\n');
fprintf('Peak-to-peak: %.8f V\n',Vpp_steady);
fprintf('RMS: %.8f V\n',Vrms_steady);
fprintf('Maximum absolute: %.8f V\n',Vpeak_steady);

figure;
plot(tsteady*1000,Vsteady);
xlabel('Time (ms)');
ylabel('Victim voltage (V)');
title('EMC-PREDICT steady-state response');
grid on;