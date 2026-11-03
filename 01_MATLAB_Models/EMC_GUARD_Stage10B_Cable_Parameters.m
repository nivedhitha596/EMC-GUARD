%% EMC-GUARD STAGE 10B
% Representative cable parameter register
% Source: Macrotherm composite travelling cable
% Research reference, not KONE-specific cable data

clear;
clc;

Cable.Name = ...
    'Composite Flat Elevator Travelling Cable';

Cable.Source = 'Macrotherm Industries';

% Documented cable architecture
Cable.Control_Cores = 10;
Cable.Control_Area_mm2 = 0.75;

Cable.Data_Pairs = 2;
Cable.Data_Area_mm2 = 0.50;

Cable.Data_Insulation = 'PE';
Cable.Data_Shield = 'Aluminium Mylar';
Cable.Drain_Wire = 'ATC';

Cable.Outer_Insulation = 'PVC Elastomer Blend';

% Approximate overall cable dimensions
Cable.Height_mm = 5.80;
Cable.Width_mm = 35.45;

% Parameters not yet verified
Cable.Conductor_Radius_mm = NaN;
Cable.Conductor_Centres_mm = NaN;
Cable.Pair_Pitch_mm = NaN;
Cable.Dielectric_Constant = NaN;
Cable.Shield_Thickness_mm = NaN;

Cable.Ground_Configuration = 'To be verified';

% Record model assumptions separately
Cable.Model_Length_mm = 450;
Cable.Model_Configuration = ...
    'Two-pair simplified coupling model';

Cable.Validation_Status = ...
    'Architecture identified; geometry pending';

disp('===== STAGE 10B CABLE REGISTER =====');
disp(Cable);
%% STAGE 10B-2: DERIVED GEOMETRY

% Equivalent solid copper radii
Cable.Control_Equivalent_Radius_mm = ...
    sqrt(Cable.Control_Area_mm2/pi);

Cable.Data_Equivalent_Radius_mm = ...
    sqrt(Cable.Data_Area_mm2/pi);

% Geometry verification status
Cable.Geometry_Status = ...
    'Equivalent radii calculated; physical positions pending';

Cable.Actual_Conductor_Radius_mm = NaN;
Cable.Actual_Conductor_Centres_mm = NaN;

fprintf('\n===== GEOMETRY DERIVATION =====\n');

fprintf('Control conductor equivalent radius: %.4f mm\n', ...
    Cable.Control_Equivalent_Radius_mm);

fprintf('Data conductor equivalent radius: %.4f mm\n', ...
    Cable.Data_Equivalent_Radius_mm);

fprintf('Actual conductor positions: NOT VERIFIED\n');
save('EMC_GUARD_Stage10B_Cable_Register.mat', ...
    'Cable');

fprintf('\nStage 10B parameter register created.\n');
fprintf('Unverified parameters remain NaN.\n');