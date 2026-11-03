
%% EMC-GUARD STAGE 10A - MATRIX VERIFICATION
clear; clc;

mu0 = 4*pi*1e-7;
eps0 = 8.854187817e-12;

% Geometry and material assumptions
Ltotal = 0.450;
a = 0.0005;
pitch = 0.006;
h = 0.050;
eps = eps0*2.3;

% Assumed self-inductances
Ls = 2e-6;
Lv = 10e-6;

% Configuration sweep
separations = 20:10:100;       % mm
overlaps = [100 200 300 450];  % mm

Results = [];
row = 0;

for Dmm = separations

    for Lpmm = overlaps

        D = Dmm*1e-3;
        Lp = Lpmm*1e-3;
        Lremaining = Ltotal-Lp;

        %% 1. Potential coefficient matrix

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

        %% 2. Uncoupled sections

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

        %% 3. Mutual inductance

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

        %% 4. Inductance matrix

        Lmatrix = [Ls M; M Lv];
%% Additional numerical checks

finiteL = all(isfinite(Lmatrix(:)));
finiteC = all(isfinite(C(:)));

symmetryL_raw = norm(Lmatrix-Lmatrix','fro');
symmetryC_raw = norm(C-C','fro');

minEigL = min(eig((Lmatrix+Lmatrix')/2));
minEigC = min(eig((C+C')/2));

if ~finiteL || ~finiteC
    error('Non-finite matrix values detected.');
end

if symmetryL_raw > 1e-10
    warning('Inductance matrix symmetry issue.');
end

if symmetryC_raw > 1e-10
    warning('Capacitance matrix symmetry issue.');
end

if minEigL <= 0 || minEigC <= 0
    warning('Positive definiteness check failed.');
end

if Lp < 0 || Lp > Ltotal
    error('Overlap must be between 0 and total cable length.');
end
        %% 5. Numerical verification

        eigL = eig(0.5*(Lmatrix+Lmatrix'));
        eigC = eig(C);

        symmetryL = norm(Lmatrix-Lmatrix','fro');
        symmetryC = norm(C-C','fro');

        condL = cond(Lmatrix);
        condC = cond(C);

        mutualRatio = abs(M)/sqrt(Ls*Lv);

        passL = all(eigL > 0);
        passC = all(eigC > 0);

        row = row+1;

        Results(row,:) = [ ...
            Dmm, Lpmm, M*1e9, ...
            min(eigL)*1e9, ...
            min(eigC)*1e12, ...
            symmetryL, symmetryC, ...
            mutualRatio, condL, condC, ...
            passL, passC];

    end
end

%% 6. Display results

Tresults = array2table(Results, ...
    'VariableNames',{ ...
    'Separation_mm', ...
    'Overlap_mm', ...
    'MutualInductance_nH', ...
    'MinEigen_L_nH', ...
    'MinEigen_C_pF', ...
    'SymmetryError_L', ...
    'SymmetryError_C', ...
    'MutualRatio', ...
    'ConditionNumber_L', ...
    'ConditionNumber_C', ...
    'L_PositiveDefinite', ...
    'C_PositiveDefinite'});

disp(Tresults);

%% 7. Overall verification

allL = all(Tresults.L_PositiveDefinite);
allC = all(Tresults.C_PositiveDefinite);

fprintf('\n========== STAGE 10A RESULTS ==========\n');

fprintf('Total configurations: %d\n',height(Tresults));

fprintf('All inductance matrices positive definite: %d\n',allL);
fprintf('All capacitance matrices positive definite: %d\n',allC);

fprintf('Maximum inductance symmetry error: %.4e\n', ...
    max(Tresults.SymmetryError_L));

fprintf('Maximum capacitance symmetry error: %.4e\n', ...
    max(Tresults.SymmetryError_C));

fprintf('Maximum mutual inductance ratio: %.8f\n', ...
    max(Tresults.MutualRatio));

outFolder = fullfile(pwd,'02_Simulation_Results');

if ~exist(outFolder,'dir')
    mkdir(outFolder);
end

writetable(Tresults,fullfile(outFolder, ...
    'EMC_GUARD_Stage10A_Matrix_Verification.csv'));

fprintf('\nSTAGE 10A COMPLETED.\n');
