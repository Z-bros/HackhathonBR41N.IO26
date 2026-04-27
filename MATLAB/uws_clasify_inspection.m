%% =========================================================
% UWS Vibro-Tactile P300 BCI Analysis Pipeline
% Step 4: Classifier Inspection (Spatial + Temporal Weights, Topographies)
% =========================================================
% Inputs:  processed_data.mat
% Outputs: classifier_inspection.pdf
% =========================================================

clear all; close all; clc;

% Darker text on figures
set(groot, 'DefaultAxesXColor', 'k');
set(groot, 'DefaultAxesYColor', 'k');
set(groot, 'DefaultAxesGridColor', [0.15 0.15 0.15]);
set(groot, 'DefaultAxesFontSize', 11);
set(groot, 'DefaultTextColor', 'k');

%% =========================================================
% SECTION 0: LOAD DATA
% =========================================================

load('processed_data.mat', 'results', 'cfg');
n_files = length(results);

%% =========================================================
% SECTION 1: 10-20 ELECTRODE COORDINATES (for topoplot)
% =========================================================
% 2D approximation of standard 10-20 positions (top-down view).
% x = left/right, y = front/back. Range roughly [-1, 1].

elec_xy = struct();
elec_xy.Fz  = [ 0.00,  0.45];
elec_xy.C3  = [-0.45,  0.00];
elec_xy.Cz  = [ 0.00,  0.00];
elec_xy.C4  = [ 0.45,  0.00];
elec_xy.CP1 = [-0.25, -0.22];
elec_xy.CPz = [ 0.00, -0.22];
elec_xy.CP2 = [ 0.25, -0.22];
elec_xy.Pz  = [ 0.00, -0.45];

% Build coordinate matrix in channel order
chan_coords = zeros(cfg.n_channels, 2);
for ch = 1:cfg.n_channels
    chan_coords(ch, :) = elec_xy.(cfg.chan_labels{ch});
end

%% =========================================================
% SECTION 2: PER-FILE: TRAIN LDA, EXTRACT WEIGHTS, COMPUTE TOPOGRAPHIES
% =========================================================

fprintf('=== Training LDA on each file and extracting weights ===\n');

inspection = struct();

for f = 1:n_files
    r = results(f);
    if isempty(r.epochs), continue; end

    fprintf('  %s ... ', r.fname);

    % --- Build feature matrix using DOWNSAMPLED features ---
    % We use 'ds' (downsampled epochs) so weights have a clean
    % time-by-channel structure that we can reshape and visualize.
    epochs = r.epochs;                        % [time x chan x trial]
    labels = r.labels;

    % Keep only target and non-target
    keep = (labels == cfg.label_target) | (labels == cfg.label_nontarget);
    epochs = epochs(:,:,keep);
    labels = labels(keep);
    y_bin  = (labels == cfg.label_target);

    % Downsample by factor 4 to get a manageable feature dimension
    ds_factor = 4;
    epochs_ds = epochs(1:ds_factor:end, :, :);
    n_time = size(epochs_ds, 1);
    n_chan = size(epochs_ds, 2);
    n_tr   = size(epochs_ds, 3);

    % Reshape to [trials x (time*chan)] with time-first ordering
    % so reshape back later is unambiguous
    X = reshape(permute(epochs_ds, [3, 1, 2]), [n_tr, n_time * n_chan]);

    if sum(y_bin) < 10 || sum(~y_bin) < 10
        fprintf('skipped (too few trials)\n');
        continue;
    end

    % --- Train LDA on full data (we want weights for visualization, not CV) ---
    mdl = fitcdiscr(X, y_bin, ...
        'DiscrimType', 'pseudoLinear', 'Prior', 'uniform');

    % LDA weight vector: difference of class means projected through
    % covariance inverse. fitcdiscr stores it in mdl.Coeffs(1,2).Linear.
    w = mdl.Coeffs(1,2).Linear;   % [n_features x 1]

    % --- Reshape weights into [time x channel] ---
    W = reshape(w, [n_time, n_chan]);

    % --- Activation pattern (Haufe et al. 2014) ---
    % Raw weights are misleading — they include suppression of correlated
    % noise, so a high weight doesn't mean the signal is strong there.
    % The "activation pattern" A = Σ_X · w gives an interpretable map of
    % where the signal actually lives.
    cov_X = cov(X);
    activation = cov_X * w;       % [n_features x 1]
    A = reshape(activation, [n_time, n_chan]);

    % --- Time axis for plots ---
    epoch_len = size(epochs, 1);
    time_axis_full = ((-cfg.epoch_pre + 1) : cfg.epoch_post) / cfg.fs * 1000;
    time_axis_ds = time_axis_full(1:ds_factor:end);

    % --- Channel-wise activation magnitude ---
    % To get a single topography per file, take RMS of activation over
    % the P300 window (200-500 ms post-stimulus)
    p300_mask = time_axis_ds >= 200 & time_axis_ds <= 500;
    chan_activation = sqrt(mean(A(p300_mask, :).^2, 1));   % [1 x n_chan]

    % --- Channel-wise weight magnitude (raw, for comparison) ---
    chan_weight = sqrt(mean(W(p300_mask, :).^2, 1));

    % Store
    inspection(f).fname = r.fname;
    inspection(f).patient = r.patient;
    inspection(f).cond = r.cond;
    inspection(f).sess = r.sess;
    inspection(f).W = W;
    inspection(f).A = A;
    inspection(f).time_axis = time_axis_ds;
    inspection(f).chan_activation = chan_activation;
    inspection(f).chan_weight = chan_weight;

    fprintf('weights extracted (%d time × %d chan)\n', n_time, n_chan);
end

fprintf('\n');

%% =========================================================
% SECTION 3: PLOTTING — PER-FILE WEIGHT/ACTIVATION HEATMAPS
% =========================================================

fprintf('=== Building inspection figures ===\n');

pdf_out = 'classifier_inspection.pdf';
if exist(pdf_out, 'file'), delete(pdf_out); end

%% ---- 3.1 Time × channel activation heatmap, one figure per file ----

for f = 1:n_files
    if isempty(inspection(f)) || ~isfield(inspection(f),'fname') || ...
            isempty(inspection(f).fname), continue; end

    r = inspection(f);
    fig = figure('Position', [50 50 1100 400], 'Color', 'w');

    % Left: raw weights
    subplot(1,2,1);
    imagesc(r.time_axis, 1:cfg.n_channels, r.W');
    set(gca, 'YDir','normal', 'YTick', 1:cfg.n_channels, ...
        'YTickLabel', cfg.chan_labels);
    colormap(gca, redblue());
    cmax = max(abs(r.W(:)));
    caxis([-cmax cmax]);
    colorbar;
    xline(0, 'k--', 'LineWidth', 1);
    xline(200, 'k:', 'LineWidth', 0.8);
    xline(500, 'k:', 'LineWidth', 0.8);
    xlabel('Time (ms)');
    title('LDA weights (raw)');

    % Right: activation pattern (Haufe-corrected, interpretable)
    subplot(1,2,2);
    imagesc(r.time_axis, 1:cfg.n_channels, r.A');
    set(gca, 'YDir','normal', 'YTick', 1:cfg.n_channels, ...
        'YTickLabel', cfg.chan_labels);
    colormap(gca, redblue());
    amax = max(abs(r.A(:)));
    caxis([-amax amax]);
    colorbar;
    xline(0, 'k--', 'LineWidth', 1);
    xline(200, 'k:', 'LineWidth', 0.8);
    xline(500, 'k:', 'LineWidth', 0.8);
    xlabel('Time (ms)');
    title('Activation pattern (interpretable)');

    sgtitle(sprintf('%s | Patient=%s | Cond=%s | Sess=%d', ...
        r.fname, r.patient, r.cond, r.sess), 'Interpreter','none');

    exportgraphics(fig, pdf_out, 'Append', true, 'ContentType','vector');
    close(fig);
end

%% ---- 3.2 Topographies in 4×2 grid (one panel per file) ----

fig = figure('Position', [50 50 1200 600], 'Color', 'w');
t = tiledlayout(2, 4, 'TileSpacing','compact', 'Padding','compact');

% First find a global color scale across all files for fair comparison
all_act = [];
for f = 1:n_files
    if isempty(inspection(f)) || isempty(inspection(f).fname), continue; end
    all_act = [all_act, inspection(f).chan_activation];
end
amax_global = max(abs(all_act));

% Order the subplots by patient and condition for visual comparison
plot_order = {'P1_low1.mat','P1_low2.mat','P1_high1.mat','P1_high2.mat', ...
              'P2_low1.mat','P2_low2.mat','P2_high1.mat','P2_high2.mat'};

for k = 1:length(plot_order)
    nexttile;

    % Find this file in inspection
    fidx = [];
    for f = 1:n_files
        if isempty(inspection(f)) || isempty(inspection(f).fname), continue; end
        if strcmp(inspection(f).fname, plot_order{k})
            fidx = f; break;
        end
    end

    if isempty(fidx)
        title(plot_order{k}, 'Interpreter','none');
        axis off;
        continue;
    end

    r = inspection(fidx);
    plot_topography(chan_coords, r.chan_activation, ...
        cfg.chan_labels, amax_global);

    title_str = sprintf('%s\n(%s, %s)', r.fname, r.patient, r.cond);
    title(title_str, 'Interpreter','none', 'FontSize', 10);
end

cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Activation magnitude (a.u.)';

title(t, 'P300-window (200–500 ms) activation topography per file', ...
    'FontWeight','bold');

exportgraphics(fig, pdf_out, 'Append', true, 'ContentType','vector');
close(fig);

%% ---- 3.3 Mean activation curves: which time points are discriminative? ----

fig = figure('Position', [50 50 1100 600], 'Color', 'w');

% Top: P1 - all four conditions
subplot(2,1,1); hold on;
for f = 1:n_files
    if isempty(inspection(f)) || isempty(inspection(f).fname), continue; end
    if ~strcmp(inspection(f).patient, 'P1'), continue; end
    r = inspection(f);

    % RMS activation across all channels at each time point
    activation_curve = sqrt(mean(r.A.^2, 2));

    color = [0.7 0.3 0.3];
    if strcmp(r.cond, 'high'), color = [0.2 0.6 0.2]; end

    plot(r.time_axis, activation_curve, 'Color', color, 'LineWidth', 1.3);
end
xline(0, 'k--', 'LineWidth', 1);
xline(200, 'k:', 'LineWidth', 0.8); xline(500, 'k:', 'LineWidth', 0.8);
ylabel('RMS activation across channels');
title('P1 — discriminative information across time');
legend({'low','low','high','high'}, 'Location', 'northeast');
grid on;

% Bottom: P2
subplot(2,1,2); hold on;
for f = 1:n_files
    if isempty(inspection(f)) || isempty(inspection(f).fname), continue; end
    if ~strcmp(inspection(f).patient, 'P2'), continue; end
    r = inspection(f);

    activation_curve = sqrt(mean(r.A.^2, 2));

    color = [0.7 0.3 0.3];
    if strcmp(r.cond, 'high'), color = [0.2 0.6 0.2]; end

    plot(r.time_axis, activation_curve, 'Color', color, 'LineWidth', 1.3);
end
xline(0, 'k--', 'LineWidth', 1);
xline(200, 'k:', 'LineWidth', 0.8); xline(500, 'k:', 'LineWidth', 0.8);
xlabel('Time (ms)');
ylabel('RMS activation across channels');
title('P2 — discriminative information across time');
legend({'low','low','high','high'}, 'Location', 'northeast');
grid on;

exportgraphics(fig, pdf_out, 'Append', true, 'ContentType','vector');
close(fig);

fprintf('  Saved %s\n\n', pdf_out);

%% =========================================================
% SECTION 4: HELPER FUNCTIONS (must be at end of file in MATLAB)
% =========================================================

function plot_topography(coords, values, labels, vmax)
    % Simple topography plot: interpolated scalp map with electrode positions

    % Define grid
    [xq, yq] = meshgrid(linspace(-0.7, 0.7, 80), linspace(-0.7, 0.7, 80));

    % Mask outside head circle
    head_radius = 0.6;
    mask = sqrt(xq.^2 + yq.^2) <= head_radius;

    % Interpolate using natural neighbor or scattered linear
    F = scatteredInterpolant(coords(:,1), coords(:,2), values(:), ...
        'natural', 'linear');
    vq = F(xq, yq);
    vq(~mask) = NaN;

    % Plot
    imagesc(xq(1,:), yq(:,1), vq);
    set(gca, 'YDir','normal');
    axis equal off;
    if nargin >= 4 && ~isempty(vmax)
        caxis([-vmax vmax]);
    end
    colormap(gca, redblue());

    hold on;

    % Head outline (circle)
    th = linspace(0, 2*pi, 100);
    plot(head_radius*cos(th), head_radius*sin(th), 'k-', 'LineWidth', 1.5);

    % Nose
    plot([0 -0.05 0.05 0], [head_radius head_radius+0.08 head_radius+0.08 head_radius], ...
        'k-', 'LineWidth', 1.5);

    % Ears (simple arcs)
    th_ear = linspace(-pi/3, pi/3, 30);
    plot(-head_radius - 0.05*cos(th_ear), 0.15*sin(th_ear), 'k-', 'LineWidth', 1);
    plot( head_radius + 0.05*cos(th_ear), 0.15*sin(th_ear), 'k-', 'LineWidth', 1);

    % Electrodes
    plot(coords(:,1), coords(:,2), 'ko', 'MarkerSize', 4, ...
        'MarkerFaceColor', 'w');

    % Labels
    for i = 1:size(coords, 1)
        text(coords(i,1), coords(i,2)+0.05, labels{i}, ...
            'HorizontalAlignment', 'center', 'FontSize', 8);
    end

    xlim([-0.75 0.75]); ylim([-0.75 0.75]);
end

function cmap = redblue()
    % Diverging red-white-blue colormap, 256 levels
    n = 256;
    half = n/2;
    r = [linspace(0, 1, half), linspace(1, 0.7, half)]';
    g = [linspace(0.3, 1, half), linspace(1, 0.1, half)]';
    b = [linspace(0.7, 1, half), linspace(1, 0, half)]';
    cmap = [r, g, b];
end