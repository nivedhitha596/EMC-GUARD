
%% EMC-GUARD STAGE 9B REVISED
% Fixed total cable length, variable parallel overlap.
% Four-conductor lumped coupled-circuit model.
%
% Important:
% This is an exploratory engineering model, not a validated
% elevator installation or EMC compliance model.

clear; clc; close all;

%% 1. Physical constants
mu0 = 4*pi*1e-7;
eps0 = 8.854187817e-12;

%% 2. Fixed cable geometry
Ltotal = 0.450;       % Total cable length (m)
a = 0.0005;           % Conductor radius (m)
pitch = 0.006;        % Pair conductor spacing (m)
h = 0.050;            % Ground-plane height (m)
er = 2.3;             % Assumed homogeneous dielectric
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

%% 4. Circuit topology
B = [1 0;
    -1 0;
     0 1;
     0 -1];

Rloop = [Rs; Rv];

Gmatrix = B*diag([1/Rload,1/Rt])*B';

%% 5. Search variables
separationList = (20:20:100)*1e-3;
overlapList = [0.100 0.200 0.300 0.450];

nD = numel(separationList);
nL = numel(overlapList);
nCases = nD*nL;

%% 6. Simulation settings
T = 2e-3;
dtOutput = 2.5e-9;
t = (0:dtOutput:T)';

maxSteps = [10 5 2.5 1.25]*1e-9;
x0 = zeros(6,1);

options = odeset( ...
    'RelTol',1e-6, ...
    'AbsTol',[1e-10 1e-10 1e-7 1e-7 1e-7 1e-7], ...
    'MaxStep',5e-9);

%% 7. Non-parallel cable capacitance
% Two separate two-conductor cable pairs.
% Mutual coupling between the two bundles is neglected here.

xPair = [-pitch/2, pitch/2];

Ppair = zeros(2,2);

for i = 1:2
    for j = 1:2

        if i == j
            Ppair(i,i) = log(2*h/a)/(2*pi*eps);
        else
            rij = abs(xPair(i)-xPair(j));
            rImage = sqrt(rij^2+(2*h)^2);

            Ppair(i,j) = log(rImage/rij)/(2*pi*eps);
        end

    end
end

CpairPerM = Ppair \ eye(2);

% Four-conductor uncoupled-section capacitance matrix.
CuncoupledPerM = zeros(4,4);

CuncoupledPerM(1:2,1:2) = CpairPerM;
CuncoupledPerM(3:4,3:4) = CpairPerM;

%% 8. Result storage
Results = zeros(nCases,8);
caseNo = 0;

%% 9. Evaluate configurations
for il = 1:nL

    Lparallel = overlapList(il);
    Lremaining = Ltotal-Lparallel;

    for id = 1:nD

        D = separationList(id);
        caseNo = caseNo+1;

        %% 9A. Four conductor coordinates in parallel section
        xpos = [-D/2-pitch/2, ...
                -D/2+pitch/2, ...
                 D/2-pitch/2, ...
                 D/2+pitch/2];

        %% 9B. Parallel-section potential coefficient matrix
        Pparallel = zeros(4,4);

        for i = 1:4
            for j = 1:4

                if i == j

                    Pparallel(i,i) = ...
                        log(2*h/a)/(2*pi*eps);

                else

                    rij = abs(xpos(i)-xpos(j));
                    rImage = sqrt(rij^2+(2*h)^2);

                    Pparallel(i,j) = ...
                        log(rImage/rij)/(2*pi*eps);

                end

            end
        end

        %% 9C. Parallel-section capacitance per metre
        CparallelPerM = Pparallel \ eye(4);

        %% 9D. Total lumped capacitance matrix
        % Parallel overlap + remaining uncoupled sections.

        Cmatrix = ...
            CparallelPerM*Lparallel + ...
            CuncoupledPerM*Lremaining;

        Cmatrix = 0.5*(Cmatrix+Cmatrix');

        if min(eig(Cmatrix)) <= 0
            error('Capacitance matrix is not positive definite.');
        end

        %% 9E. Mutual inductance of parallel section
        sourceX = [-pitch/2 pitch/2];
        victimX = [D-pitch/2 D+pitch/2];

        sourceSign = [1 -1];
        victimSign = [1 -1];

        M = 0;

        for i = 1:2
            for j = 1:2

                r = abs(sourceX(i)-victimX(j));

                Mp = mu0/(2*pi)*( ...
                    Lparallel*asinh(Lparallel/r) ...
                    -sqrt(Lparallel^2+r^2)+r);

                M = M + ...
                    sourceSign(i)*victimSign(j)*Mp;

            end
        end

        %% 9F. Fixed self-inductance, variable mutual inductance
        Lmatrix = [Ls M; M Lv];

        if min(eig(Lmatrix)) <= 0
            error('Inductance matrix is not positive definite.');
        end

        %% 9G. Solve coupled circuit
        [tSol,xSol] = ode15s( ...
            @(tt,xx) rhs9b_revised( ...
            tt,xx,Lmatrix,Cmatrix,B,Rloop, ...
            Gmatrix,Vdc,fpwm,duty), ...
            t,x0,options);

        Vvictim = xSol(:,5)-xSol(:,6);

        if any(~isfinite(Vvictim))
            error('Non-finite result at D = %.1f mm.',D*1000);
        end

        %% 9H. Steady-state measurements
        idx = tSol >= T-0.5e-3;
        steady = Vvictim(idx);

        Vpp = max(steady)-min(steady);
        Vrms = sqrt(mean(steady.^2));
        Vmax = max(abs(steady));

        %% 9I. Store results
        Results(caseNo,:) = [ ...
            D*1000, ...
            Lparallel*1000, ...
            M*1e9, ...
            Cmatrix(1,3)*1e12, ...
            Vpp*1000, ...
            Vrms*1000, ...
            Vmax*1000, ...
            Lremaining*1000];

        fprintf(['Completed %d/%d | D = %.0f mm | ', ...
            'Overlap = %.0f mm | RMS = %.6f mV\n'], ...
            caseNo,nCases,D*1000, ...
            Lparallel*1000,Vrms*1000);

    end
end

%% 10. Results table
Tresults = array2table(Results, ...
    'VariableNames',{ ...
    'Separation_mm', ...
    'ParallelOverlap_mm', ...
    'MutualInductance_nH', ...
    'C13_pF', ...
    'Victim_Vpp_mV', ...
    'Victim_RMS_mV', ...
    'Victim_MaxAbs_mV', ...
    'RemainingLength_mm'});

disp(Tresults);

%% 11. Find lowest predicted RMS
[bestRMS,bestIndex] = min(Tresults.Victim_RMS_mV);

fprintf('\n====================================\n');
fprintf('STAGE 9B REVISED RESULT\n');
fprintf('====================================\n');

fprintf('Best tested separation: %.1f mm\n', ...
    Tresults.Separation_mm(bestIndex));

fprintf('Best tested overlap: %.1f mm\n', ...
    Tresults.ParallelOverlap_mm(bestIndex));

fprintf('Minimum predicted RMS: %.6f mV\n',bestRMS);

%% 12. Heatmap
RMSmatrix = reshape(Tresults.Victim_RMS_mV,nD,nL);

figure;

imagesc(overlapList*1000,separationList*1000,RMSmatrix);

set(gca,'YDir','normal');

colorbar;

xlabel('Parallel overlap length (mm)');
ylabel('Cable separation (mm)');

title('EMC-GUARD Stage 9B Revised - Victim RMS (mV)');

%% 13. Export
outFolder = fullfile(pwd,'02_Simulation_Results');

if ~exist(outFolder,'dir')
    mkdir(outFolder);
end

writetable(Tresults,fullfile(outFolder, ...
    'EMC_GUARD_Stage9B_Revised.csv'));

fprintf('\nSTAGE 9B REVISED COMPLETED.\n');

%% 14. Coupled circuit equations
function dx = rhs9b_revised(t,x,Lmatrix,Cmatrix, ...
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

    di = Lmatrix\(e-Rloop.*i-B'*v);
    dv = Cmatrix\(B*i-Gmatrix*v);

    dx = [di;dv];

end
