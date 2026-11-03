
%% EMC-PREDICT | Stage 4
% Geometry-integrated transient coupling simulation
% Preliminary model: unshielded two-core cable loops in air.

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

%% 3. PWM waveform with finite rise/fall times
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

%% 4. Cable geometry
mu0 = 4*pi*1e-7;
eps0 = 8.854187817e-12;

cableLength = 0.450;
conductorRadius = 0.0005;
pairPitch = 0.006;

spacing = [0.020 0.040 0.080 0.100];

%% 5. Victim circuit
Rv = 10;
Lv = 10e-6;
Cv = 100e-12;
Rt = 10e3;

%% 6. Allocate results
M_values = zeros(length(spacing),1);
Cm_values = zeros(length(spacing),1);
Vpp_values = zeros(length(spacing),1);
Vrms_values = zeros(length(spacing),1);

%% 7. Geometry and transient simulation
for n = 1:length(spacing)

    D = spacing(n);

    sourceX = [-pairPitch/2, pairPitch/2];
    victimX = [D-pairPitch/2, D+pairPitch/2];

    sourceSign = [1,-1];
    victimSign = [1,-1];

    %% Mutual inductance between complete loops
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

    M_values(n) = M;
    Cm_values(n) = Cm;

    %% Coupling excitation
    Vind = M*dIs;
    Ic = Cm*dVs;

    %% RK4 transient simulation
    x = zeros(2,1);
    Vvictim = zeros(N,1);

    for k = 1:N-1

        u1 = [Vind(k); Ic(k)];

        um = 0.5*([Vind(k); Ic(k)] + ...
                  [Vind(k+1); Ic(k+1)]);

        k1 = victim_rhs(x,u1,Rv,Lv,Cv,Rt);

        k2 = victim_rhs(x+0.5*dt*k1, ...
                        um,Rv,Lv,Cv,Rt);

        k3 = victim_rhs(x+0.5*dt*k2, ...
                        um,Rv,Lv,Cv,Rt);

        k4 = victim_rhs(x+dt*k3, ...
                        [Vind(k+1);Ic(k+1)], ...
                        Rv,Lv,Cv,Rt);

        x = x + (dt/6)*(k1+2*k2+2*k3+k4);

        Vvictim(k+1) = x(2);
    end

    %% Numerical validation
    if any(~isfinite(Vvictim))
        error('Invalid numerical result at spacing %.1f mm',D*1000);
    end

    %% Last 1 ms metrics
    idx = t >= 4e-3;
    Vsteady = Vvictim(idx);

    Vpp_values(n) = max(Vsteady)-min(Vsteady);
    Vrms_values(n) = sqrt(mean(Vsteady.^2));

    fprintf('Completed spacing: %.0f mm\n',D*1000);

end

%% 8. Results table
Results = table( ...
    spacing'*1000, ...
    M_values*1e9, ...
    Cm_values*1e12, ...
    Vpp_values, ...
    Vrms_values, ...
    'VariableNames', ...
    {'Spacing_mm','M_nH','Cm_pF', ...
     'Victim_Vpp_V','Victim_RMS_V'});

disp('EMC-PREDICT STAGE 4 RESULTS');
disp(Results);

%% 9. Save dataset
writetable(Results,'EMC_PREDICT_Stage4_Integrated.csv');

%% 10. Plot results
figure;

plot(spacing*1000,Vpp_values,'-o','LineWidth',1.8);

xlabel('Cable centre separation (mm)');
ylabel('Victim voltage Vpp (V)');
title('EMC-PREDICT: Geometry-based interference prediction');
grid on;

figure;

plot(spacing*1000,Vrms_values,'-s','LineWidth',1.8);

xlabel('Cable centre separation (mm)');
ylabel('Victim voltage RMS (V)');
title('EMC-PREDICT: RMS interference vs separation');
grid on;

fprintf('\nStage 4 completed. Results exported.\n');

%% Local function
function dx = victim_rhs(x,u,Rv,Lv,Cv,Rt)

    i = x(1);
    v = x(2);

    Vind = u(1);
    Ic = u(2);

    di = (Vind-Rv*i-v)/Lv;
    dv = (i+Ic-v/Rt)/Cv;

    dx = [di;dv];

end
