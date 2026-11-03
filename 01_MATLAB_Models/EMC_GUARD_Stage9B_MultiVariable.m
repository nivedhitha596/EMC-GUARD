
%% EMC-GUARD STAGE 9B
% Two-variable cable routing optimisation
% Variables: separation and parallel overlap length
% Based on corrected Stage 8C circuit topology.

clear; clc; close all;

%% 1. Constants
eps0 = 8.854187817e-12;
mu0 = 4*pi*1e-7;

%% 2. Fixed geometry
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
Ls_ref = 2e-6;

Rv_ref = 10;
Rt = 10e3;
Lv_ref = 10e-6;

Lref = 0.450;

%% 4. Circuit topology
B = [1 0;
    -1 0;
     0 1;
     0 -1];

Gmatrix = B*diag([1/Rload,1/Rt])*B';

%% 5. Search variables
separationList = (20:20:100)*1e-3;
lengthList = [0.100 0.200 0.300 0.450];

nD = length(separationList);
nL = length(lengthList);
nCases = nD*nL;

%% 6. Solver settings
T = 2e-3;
dtOutput = 10e-9;
t = (0:dtOutput:T)';

x0 = zeros(6,1);

options = odeset( ...
    'RelTol',1e-6, ...
    'AbsTol',[1e-10 1e-10 1e-7 1e-7 1e-7 1e-7], ...
    'MaxStep',5e-9);

%% 7. Result storage
Results = zeros(nCases,7);
caseNo = 0;

%% 8. Evaluate all configurations
for il = 1:nL

    L = lengthList(il);

    % Scale assumed self-inductances and victim loop resistance
    % with the investigated cable length.
    scale = L/Lref;

    Ls = Ls_ref*scale;
    Lv = Lv_ref*scale;
    Rv = Rv_ref*scale;

    Rloop = [Rs; Rv];

    for id = 1:nD

        D = separationList(id);

        caseNo = caseNo+1;

        %% 8A. Four conductor coordinates
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

        %% 8C. Capacitance matrix
        Cmatrix = (P\eye(4))*L;
        Cmatrix = 0.5*(Cmatrix+Cmatrix');

        if min(eig(Cmatrix)) <= 0
            error('Invalid capacitance matrix.');
        end

        %% 8D. Geometry-dependent mutual inductance
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

        %% 8E. Coupled inductance matrix
        Lmatrix = [Ls M; M Lv];

        if min(eig(Lmatrix)) <= 0
            error('Invalid inductance matrix.');
        end

        %% 8F. Simulate
        [tSol,xSol] = ode15s( ...
            @(tt,xx) rhs9b(tt,xx,Lmatrix,Cmatrix, ...
            B,Rloop,Gmatrix,Vdc,fpwm,duty), ...
            t,x0,options);

        Vvictim = xSol(:,5)-xSol(:,6);

        if any(~isfinite(Vvictim))
            error('Invalid result at separation %.1f mm, length %.1f mm.', ...
                D*1000,L*1000);
        end

        %% 8G. Steady-state metrics
        idx = tSol >= T-0.5e-3;
        steady = Vvictim(idx);

        Vpp = max(steady)-min(steady);
        Vrms = sqrt(mean(steady.^2));
        Vmax = max(abs(steady));

        %% 8H. Store results
        Results(caseNo,:) = [ ...
            D*1000, ...
            L*1000, ...
            M*1e9, ...
            Vpp*1000, ...
            Vrms*1000, ...
            Vmax*1000, ...
            scale];

        fprintf(['Completed %d/%d | D = %.0f mm | ', ...
            'Length = %.0f mm | RMS = %.6f mV\n'], ...
            caseNo,nCases,D*1000,L*1000,Vrms*1000);

    end
end

%% 9. Convert to table
Tresults = array2table(Results, ...
    'VariableNames', { ...
    'Separation_mm', ...
    'ParallelLength_mm', ...
    'MutualInductance_nH', ...
    'Victim_Vpp_mV', ...
    'Victim_RMS_mV', ...
    'Victim_MaxAbs_mV', ...
    'LengthScale'});

disp(Tresults);

%% 10. Find best tested configuration
[bestRMS,bestIndex] = min(Tresults.Victim_RMS_mV);

fprintf('\n====================================\n');
fprintf('STAGE 9B OPTIMISATION RESULT\n');
fprintf('====================================\n');

fprintf('Best tested separation: %.1f mm\n', ...
    Tresults.Separation_mm(bestIndex));

fprintf('Best tested parallel length: %.1f mm\n', ...
    Tresults.ParallelLength_mm(bestIndex));

fprintf('Minimum predicted RMS: %.6f mV\n',bestRMS);

%% 11. Heatmap
RMSmatrix = reshape(Tresults.Victim_RMS_mV,nD,nL);

figure;

imagesc(lengthList*1000,separationList*1000,RMSmatrix);

set(gca,'YDir','normal');

colorbar;

xlabel('Parallel overlap length (mm)');
ylabel('Cable separation (mm)');

title('EMC-GUARD Stage 9B - Predicted victim RMS (mV)');

%% 12. Export results
outFolder = fullfile(pwd,'02_Simulation_Results');

if ~exist(outFolder,'dir')
    mkdir(outFolder);
end

writetable(Tresults,fullfile(outFolder, ...
    'EMC_GUARD_Stage9B_MultiVariable.csv'));

fprintf('\nSTAGE 9B COMPLETED.\n');

%% 13. Circuit differential equations
function dx = rhs9b(t,x,Lmatrix,Cmatrix, ...
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
