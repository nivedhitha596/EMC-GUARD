
function result = EMC_GUARD_Predict(separation,pitch,overlap,ground)

% EMC-GUARD exact-grid prediction engine
% Uses frozen Stage 12A model coefficients and feature matrices.
%
% Supported inputs are restricted to the original 216 configurations.
% This function does not interpolate or extrapolate.

persistent F S

frozenFile = fullfile( ...
    'D:\Downloads\kone\01_MATLAB_Models', ...
    '06_Stage11A_Results', ...
    '07_Stage12A_ModelFreeze', ...
    'Stage12A_Selected_Models_FROZEN.mat');

sourceFile = fullfile( ...
    'D:\Downloads\kone\01_MATLAB_Models', ...
    '06_Stage11A_Results', ...
    'Stage11D_Complete.mat');

% Load model assets once
if isempty(F) || isempty(S)

    if ~isfile(frozenFile)
        error('Frozen Stage 12A model file not found.');
    end

    if ~isfile(sourceFile)
        error('Stage 11D source file not found.');
    end

    F = load(frozenFile);
    S = load(sourceFile);

    requiredFrozen = ...
        {'Xcap','Xmut','betaCap','betaMut'};

    for k = 1:numel(requiredFrozen)
        if ~isfield(F,requiredFrozen{k})
            error('Missing frozen model variable: %s', ...
                requiredFrozen{k});
        end
    end

    if ~isfield(S,'T')
        error('Stage 11D geometry table T is missing.');
    end

end

% Validate scalar numeric inputs
inputs = [separation,pitch,overlap,ground];

if ~isnumeric(inputs) || ...
        ~isreal(inputs) || ...
        numel(inputs) ~= 4 || ...
        any(~isfinite(inputs))

    error('All four inputs must be finite real numeric scalars.');

end

if any(~arrayfun(@isscalar, ...
        {separation,pitch,overlap,ground}))

    error('Each geometry input must be a scalar.');

end

% Match exact geometry in original 216-point design
T = S.T;

match = ...
    T.PairSeparation_mm == separation & ...
    T.PairPitch_mm == pitch & ...
    T.Overlap_mm == overlap & ...
    T.GroundClearance_mm == ground;

idx = find(match);

if isempty(idx)
    error(['Unsupported geometry. Use only the original ', ...
        '216 tested configurations. No extrapolation is allowed.']);
end

if numel(idx) ~= 1
    error('Geometry lookup is not unique.');
end

% Extract exact feature rows
xC = F.Xcap(idx,:);
xM = F.Xmut(idx,:);

% Frozen-model predictions
Cpred = xC * F.betaCap;
Mpred = xM * F.betaMut;

% Return structured result
result.Separation_mm = separation;
result.Pitch_mm = pitch;
result.Overlap_mm = overlap;
result.GroundClearance_mm = ground;

result.Capacitance_pF_per_m = Cpred;
result.AbsMutualInductance_nH = Mpred;

result.CapacitanceModel = 'B2_SP_Ground';
result.MutualInductanceModel = 'B1_SP_Overlap';

result.DesignIndex = idx;
result.ValidationDomain = 'Original 216-point design grid only';

end
