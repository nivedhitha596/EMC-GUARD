
%% EMC-GUARD | STAGE 7
% Coupled source-victim lumped circuit model
% Preliminary engineering model; not experimentally validated.
%
% Model includes:
% 1. Source resistance, load and cable inductance
% 2. Victim resistance, inductance, capacitance and termination
% 3. Signed mutual inductance
% 4. Lumped mutual capacitance between cable nodes
%
% NOTE:
% Geometry-based Cm remains a simplified air approximation.
% All electrical parameters must eventually be measured or calibrated.

clear; clc; close all;

%% 1. Source parameters
Vdc = 5;
fpwm = 5000;
duty = 0.50;
Rload = 47;

Rs = 1;                 % Source resistance (ohm)
Ls = 2e-6;              % Source loop inductance (H), assumed

%% 2. Victim parameters
Rv = 10;                % Victim resistance (ohm)
Lv = 10e-6;             % Victim loop inductance (H)
Cg = 100e-12;           % Victim capacitance to reference (F)
Rt = 10e3;              % Victim termination (ohm)

%% 3. Simulation settings
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

pwm = min(rise,fall);
Vs = Vdc*pwm;

%% 4. Cable geometry
mu0 = 4*pi*1e-7;
eps0 = 8.854187817e-12;

cableLength = 0.450;
conductorRadius = 0.0005;
pairPitch = 0.006;

spacing = [0.020 0.040 0.080 0.100];

%% 5. Results storage
nCases = length(spacing);

M_nH = zeros(nCases,1);
Cm_pF = zeros(nCases,1);

Vpp = zeros(nCases,1);
Vrms = zeros(nCases,1);
Vmax = zeros(nCases,1);

%% 6. Run each cable configuration
for n = 1:nCases

    D = spacing(n);

    sourceX = [-pairPitch/2, pairPitch/2];
    victimX = [D-pairPitch/2, D+pairPitch/2];

    sourceSign = [1,-1];
    victimSign = [1,-1];

    %% Signed mutual inductance
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

    M = Mloop;

    %% Preliminary mutual capacitance approximation
    Cm = pi*eps0*cableLength / ...
        acosh(D/(2*conductorRadius));

    %% Physical consistency check
    if Ls*Lv <= M^2
        error('Invalid inductance matrix: check Ls, Lv and M.');
    end

    %% Coupled inductance matrix
    Lmatrix = [Ls M; M Lv];

    %% Initial states
    % x = [source current; victim current; victim voltage]
    x = zeros(3,1);

    Vvictim = zeros(N,1);

    %% RK4 integration
    for k = 1:N-1

        u1 = Vs(k);
        um = 0.5*(Vs(k)+Vs(k+1));
        u4 = Vs(k+1);

        k1 = coupled_rhs(x,u1,Lmatrix,Rs,Rload,...
            Rv,Cg,Cm,Rt);

        k2 = coupled_rhs(x+0.5*dt*k1,um,Lmatrix,...
            Rs,Rload,Rv,Cg,Cm,Rt);

        k3 = coupled_rhs(x+0.5*dt*k2,um,Lmatrix,...
            Rs,Rload,Rv,Cg,Cm,Rt);

        k4 = coupled_rhs(x+dt*k3,u4,Lmatrix,...
            Rs,Rload,Rv,Cg,Cm,Rt);

        x = x + (dt/6)*(k1+2*k2+2*k3+k4);

        Vvictim(k+1) = x(3);
    end

    %% Numerical checks
    if any(~isfinite(Vvictim))
        error('Non-finite voltage result at %.0f mm.',D*1000);
    end

    %% Steady-state measurements
    idx = t >= (T-0.5e-3);
    steady = Vvictim(idx);

    Vpp(n) = max(steady)-min(steady);
    Vrms(n) = sqrt(mean(steady.^2));
    Vmax(n) = max(abs(steady));

    M_nH(n) = M*1e9;
    Cm_pF(n) = Cm*1e12;

    fprintf('Completed spacing: %.0f mm\n',D*1000);
end

%% 7. Results table
Results = table( ...
    spacing'*1000,M_nH,Cm_pF,...
    Vpp,Vrms,Vmax,...
    'VariableNames',{...
    'Spacing_mm','M_nH','Cm_pF',...
    'Victim_Vpp_V','Victim_RMS_V',...
    'Victim_MaxAbs_V'});

disp('========== EMC-GUARD STAGE 7 RESULTS ==========');
disp(Results);

%% 8. Export results
outputFolder = fullfile(pwd,'02_Simulation_Results');

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

outputFile = fullfile(outputFolder,...
    'EMC_GUARD_Stage7_CoupledLumped.csv');

writetable(Results,outputFile);

%% 9. Plot results
figure;

plot(spacing*1000,Vpp,'-o','LineWidth',1.6);

xlabel('Cable separation (mm)');
ylabel('Victim voltage peak-to-peak (V)');
title('EMC-GUARD Stage 7: Coupled-circuit prediction');
grid on;

saveas(gcf,fullfile(outputFolder,...
    'EMC_GUARD_Stage7_VictimVoltage.png'));

fprintf('\nStage 7 completed.\n');
fprintf('Results saved to: %s\n',outputFile);

%% 10. Coupled circuit differential equations
function dx = coupled_rhs(x,Vs,Lmatrix,Rs,Rload,...
    Rv,Cg,Cm,Rt)

    is = x(1);
    iv = x(2);
    v = x(3);

    % Source loop and victim loop
    b = [Vs-(Rs+Rload)*is;
         -Rv*iv-v];

    currentsDerivative = Lmatrix\b;

    dis = currentsDerivative(1);
    div = currentsDerivative(2);

    % Source node voltage = Rload * source current
    % Capacitive coupling current = Cm*d(Vsource-v)/dt
    dv = (iv + Cm*Rload*dis - v/Rt)/(Cg+Cm);

    dx = [dis;div;dv];
end
