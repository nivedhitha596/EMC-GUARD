
%% EMC-GUARD STAGE 9B - OUTPUT GRID CONVERGENCE TEST
clear; clc;

mu0 = 4*pi*1e-7;
eps0 = 8.854187817e-12;

Ltotal = 0.450;
a = 0.0005;
pitch = 0.006;
h = 0.050;
eps = eps0*2.3;

Vdc = 5;
fpwm = 5000;
duty = 0.50;

Rs = 1;
Rload = 47;
Ls = 2e-6;

Rv = 10;
Rt = 1e4;
Lv = 10e-6;

B = [1 0;-1 0;0 1;0 -1];
Rloop = [Rs;Rv];
G = B*diag([1/Rload,1/Rt])*B';

% Test configurations: [separation_mm, overlap_mm]
cases = [20 450; 100 100];

% Output sampling intervals
outputSteps = [10 5 2.5 1.25]*1e-9;

% Keep internal solver maximum step fixed
solverMaxStep = 1e-9;

T = 2e-3;
x0 = zeros(6,1);

% 2 configurations x 4 output intervals
Results = zeros(8,5);
row = 0;

for c = 1:size(cases,1)

    D = cases(c,1)*1e-3;
    Lp = cases(c,2)*1e-3;
    Lremaining = Ltotal-Lp;

    xpos = [-D/2-pitch/2, ...
            -D/2+pitch/2, ...
             D/2-pitch/2, ...
             D/2+pitch/2];

    P = zeros(4,4);

    for i = 1:4
        for j = 1:4
            if i == j
                P(i,i) = log(2*h/a)/(2*pi*eps);
            else
                r = abs(xpos(i)-xpos(j));
                ri = sqrt(r^2+(2*h)^2);
                P(i,j) = log(ri/r)/(2*pi*eps);
            end
        end
    end

    Cparallel = P\eye(4);

    % Uncoupled cable sections
    xp = [-pitch/2 pitch/2];
    Ppair = zeros(2,2);

    for i = 1:2
        for j = 1:2
            if i == j
                Ppair(i,i) = log(2*h/a)/(2*pi*eps);
            else
                r = abs(xp(i)-xp(j));
                ri = sqrt(r^2+(2*h)^2);
                Ppair(i,j) = log(ri/r)/(2*pi*eps);
            end
        end
    end

    Cpair = Ppair\eye(2);
    Cuncoupled = zeros(4,4);
    Cuncoupled(1:2,1:2) = Cpair;
    Cuncoupled(3:4,3:4) = Cpair;

    C = Cparallel*Lp + Cuncoupled*Lremaining;
    C = 0.5*(C+C');

    % Mutual inductance
    sx = [-pitch/2 pitch/2];
    vx = [D-pitch/2 D+pitch/2];
    ss = [1 -1];
    vs = [1 -1];

    M = 0;

    for i = 1:2
        for j = 1:2
            r = abs(sx(i)-vx(j));

            Mp = mu0/(2*pi)*( ...
                Lp*asinh(Lp/r)-sqrt(Lp^2+r^2)+r);

            M = M+ss(i)*vs(j)*Mp;
        end
    end

    Lmatrix = [Ls M;M Lv];

    for k = 1:numel(outputSteps)

        dtOutput = outputSteps(k);
        t = (0:dtOutput:T)';

        opts = odeset( ...
            'RelTol',1e-6, ...
            'AbsTol',[1e-10 1e-10 1e-7 1e-7 1e-7 1e-7], ...
            'MaxStep',solverMaxStep);

        [ts,xs] = ode15s( ...
            @(tt,xx) rhsConv(tt,xx,Lmatrix,C,B, ...
            Rloop,G,Vdc,fpwm,duty), ...
            t,x0,opts);

        vv = xs(:,5)-xs(:,6);

        % Last 0.5 ms of simulation
        idx = ts >= T-0.5e-3;
        tss = ts(idx);
        vss = vv(idx);

        vpp = max(vss)-min(vss);

        % RMS calculated using numerical integration
        vrms = sqrt(trapz(tss,vss.^2)/(tss(end)-tss(1)));

        row = row+1;

        Results(row,:) = [ ...
            cases(c,1), ...
            cases(c,2), ...
            dtOutput*1e9, ...
            vpp*1000, ...
            vrms*1000];

        fprintf(['Case %d | Output %.2f ns | ' ...
            'RMS %.9f mV\n'], ...
            c,dtOutput*1e9,vrms*1000);

    end
end

Tresults = array2table(Results, ...
    'VariableNames',{'Separation_mm','Overlap_mm', ...
    'OutputStep_ns','Vpp_mV','RMS_mV'});

disp(Tresults);

% Compare successive output resolutions
for c = 1:2

    r = (c-1)*4;

    fprintf('\nCase %d output-grid convergence:\n',c);

    for k = 1:3

        change = abs(Results(r+k,5)-Results(r+k+1,5)) ...
            /Results(r+k+1,5)*100;

        fprintf('%.2f ns vs %.2f ns: %.6f %%\n', ...
            Results(r+k,3),Results(r+k+1,3),change);
    end
end

outFolder = fullfile(pwd,'02_Simulation_Results');

if ~exist(outFolder,'dir')
    mkdir(outFolder);
end

writetable(Tresults,fullfile(outFolder, ...
    'EMC_GUARD_Stage9B_OutputGrid_Convergence.csv'));

fprintf('\nSTAGE 9B OUTPUT GRID TEST COMPLETED.\n');

function dx = rhsConv(t,x,L,C,B,R,G,Vdc,fpwm,duty)

    i = x(1:2);
    v = x(3:6);

    period = 1/fpwm;
    tr = 100e-9;
    tf = 100e-9;

    phase = mod(t,period);
    rise = min(phase/tr,1);
    fall = min(max((duty*period-phase)/tf,0),1);

    pwm = min(rise,fall);
    e = [Vdc*pwm;0];

    di = L\(e-R.*i-B'*v);
    dv = C\(B*i-G*v);

    dx = [di;dv];
end
