%% =========================================================
% UWS Vibro-Tactile P300 BCI Analysis Pipeline
% Hackathon - Preprocessing
% =========================================================
% Files needed: P1_low1.mat, P1_low2.mat, P1_high1.mat, P1_high2.mat
%               P2_low1.mat, P2_low2.mat, P2_high1.mat, P2_high2.mat
% =========================================================

clear all; close all; clc;

% --- Global figure appearance: darker text for readability ---
set(groot, 'DefaultAxesXColor', 'k');
set(groot, 'DefaultAxesYColor', 'k');
set(groot, 'DefaultAxesZColor', 'k');
set(groot, 'DefaultAxesGridColor', [0.15 0.15 0.15]);
set(groot, 'DefaultAxesFontWeight', 'normal');
set(groot, 'DefaultAxesFontSize', 11);
set(groot, 'DefaultTextColor', 'k');
set(groot, 'DefaultAxesLabelFontSizeMultiplier', 1.1);
set(groot, 'DefaultAxesTitleFontSizeMultiplier', 1.15);

%% =========================================================
% SECTION 0: CONFIGURATION
% =========================================================

cfg = struct();

% --- File settings ---
cfg.files = {
    'P1_low1.mat',  'P1', 'low',  1;
    'P1_low2.mat',  'P1', 'low',  2;
    'P1_high1.mat', 'P1', 'high', 1;
    'P1_high2.mat', 'P1', 'high', 2;
    'P2_low1.mat',  'P2', 'low',  1;
    'P2_low2.mat',  'P2', 'low',  2;
    'P2_high1.mat', 'P2', 'high', 1;
    'P2_high2.mat', 'P2', 'high', 2;
    };

% --- Channel labels ---
cfg.chan_labels = {'Fz','C3','Cz','C4','CP1','CPz','CP2','Pz'};
cfg.n_channels  = 8;

% --- Sampling rate ---
cfg.fs = 256;  % Hz (from papers)

% --- Epoch window ---
cfg.epoch_pre_ms  = 100;   % ms before stimulus
cfg.epoch_post_ms = 600;   % ms after stimulus
cfg.epoch_pre  = round(cfg.epoch_pre_ms  / 1000 * cfg.fs);
cfg.epoch_post = round(cfg.epoch_post_ms / 1000 * cfg.fs);
cfg.epoch_len  = cfg.epoch_pre + cfg.epoch_post;

% --- Filtering ---
cfg.bandpass_low  = 0.1;   % Hz
cfg.bandpass_high = 30;    % Hz
cfg.notch_freq    = 50;    % Hz

% --- Artifact rejection ---
cfg.artifact_thresh = 100; % µV


% --- Class labels ---
cfg.label_target    =  2;
cfg.label_nontarget =  1;
cfg.label_distractor = -1;


fprintf('=== Configuration loaded ===\n');
fprintf('Epoch window: -%dms to +%dms\n', cfg.epoch_pre_ms, cfg.epoch_post_ms);
fprintf('Filter: %.1f-%.1f Hz + %dHz notch\n', ...
    cfg.bandpass_low, cfg.bandpass_high, cfg.notch_freq);
fprintf('Artifact threshold: ±%d µV\n\n', cfg.artifact_thresh);

%% =========================================================
% SECTION 1: BUILDING FILTERS
% =========================================================

fprintf('=== Building filters ===\n');

% --- Bandpass: 0.1-30 Hz, 4th-order IIR in SOS form ---
% designfilt produces a stable filter object even at low cutoff ratios
% where butter() + filtfilt() becomes numerically ill-conditioned.
cfg.bp_filter = designfilt('bandpassiir', ...
    'FilterOrder', 4, ...
    'HalfPowerFrequency1', cfg.bandpass_low, ...
    'HalfPowerFrequency2', cfg.bandpass_high, ...
    'SampleRate', cfg.fs);

fprintf('Bandpass: %.1f-%.1f Hz (4th-order IIR, SOS)\n', ...
    cfg.bandpass_low, cfg.bandpass_high);

% --- Notch at 50 Hz via iirnotch ---
% iirnotch is the standard, numerically stable way to build a narrow
% bandstop. Q controls sharpness; Q=35 gives ~1.4 Hz bandwidth at 50 Hz.
notch_Q = 35;
w0 = cfg.notch_freq / (cfg.fs/2);   % normalized notch frequency
bw = w0 / notch_Q;                  % normalized bandwidth
[cfg.notch_b, cfg.notch_a] = iirnotch(w0, bw);

fprintf('Notch: %d Hz (iirnotch, Q=%d)\n\n', cfg.notch_freq, notch_Q);
%% =========================================================
% SECTION 2: PREPROCESSING EACH FILE
% =========================================================
% first run test_trigger_detection.m
% then test on 1 file preprocessing1file.m
n_files = size(cfg.files, 1);
results = repmat(struct('fname',[], 'patient',[], 'cond',[], 'sess',[], ...
                        'epochs',[], 'labels',[], 'n_rejected',[], 'n_total',[]), ...
                 n_files, 1);

for f = 1:n_files

    fname   = cfg.files{f,1};
    patient = cfg.files{f,2};
    cond    = cfg.files{f,3};
    sess    = cfg.files{f,4};

    fprintf('========================================\n');
    fprintf('Processing: %s (Patient=%s, Cond=%s, Session=%d)\n', ...
        fname, patient, cond, sess);
    fprintf('========================================\n');

    % --- 2.1 Load data ---
    fprintf('  Loading data...\n');
    if ~exist(fname, 'file')
        fprintf('  WARNING: File not found, skipping.\n\n');
        continue;
    end

    raw = load(fname);
    y    = raw.y;    % [samples x channels]
    trig = raw.trig; % [samples x 1]
    fs_check = raw.fs;

    if fs_check ~= cfg.fs
        fprintf('  WARNING: fs in file (%d) differs from config (%d)!\n', ...
            fs_check, cfg.fs);
    end

    fprintf('  Data size: %d samples x %d channels\n', size(y,1), size(y,2));
    fprintf('  Duration: %.1f seconds\n', size(y,1)/cfg.fs);

    % --- 2.2 Filter the continuous data ---
    % --- 2.2 Filter the continuous data ---
    fprintf('  Filtering...\n');
    y_filt = zeros(size(y));
    for ch = 1:cfg.n_channels
        % Notch first to remove mains, then bandpass
        tmp = filtfilt(cfg.notch_b, cfg.notch_a, y(:,ch));
        y_filt(:,ch) = filtfilt(cfg.bp_filter, tmp);
    end

    % --- 2.3 Find stimulus events ---
    fprintf('  Finding stimulus events...\n');
    
    stim_onsets = [];
    stim_labels = [];
    for cls = [cfg.label_distractor, cfg.label_nontarget, cfg.label_target]
        onsets = find(diff([0; trig == cls]) == 1);
        stim_onsets = [stim_onsets; onsets];
        stim_labels = [stim_labels; cls * ones(size(onsets))];
    end
    [stim_onsets, sort_idx] = sort(stim_onsets);
    stim_labels = stim_labels(sort_idx);
    
    n_target     = sum(stim_labels == cfg.label_target);
    n_nontarget  = sum(stim_labels == cfg.label_nontarget);
    n_distractor = sum(stim_labels == cfg.label_distractor);
    n_total      = length(stim_labels);
    
    fprintf('  Events found: %d total\n', n_total);
    fprintf('    Target (2):     %d\n', n_target);
    fprintf('    Non-target (1): %d\n', n_nontarget);
    fprintf('    Distractor(-1): %d\n', n_distractor);

    % --- 2.4 Epoch + baseline + reject ---
    fprintf('  Epoching...\n');
    
    % Preallocate generously, then trim
    max_epochs = length(stim_onsets);
    epochs       = nan(cfg.epoch_len, cfg.n_channels, max_epochs);
    epoch_labels = nan(max_epochs, 1);
    n_kept = 0;
    n_rejected = 0;
    
    for i = 1:length(stim_onsets)
        onset = stim_onsets(i);
        start_idx = onset - cfg.epoch_pre + 1;
        end_idx   = onset + cfg.epoch_post;
    
        if start_idx < 1 || end_idx > size(y_filt,1)
            continue;
        end
    
        epoch = y_filt(start_idx:end_idx, :);
    
        % Baseline correct FIRST (DC offset from filtering would otherwise
        % cause unfair rejections at ±100 µV)
        baseline = mean(epoch(1:cfg.epoch_pre, :), 1);
        epoch = epoch - baseline;
    
        % Then reject on baseline-corrected amplitude
        if max(abs(epoch(:))) > cfg.artifact_thresh
            n_rejected = n_rejected + 1;
            continue;
        end
    
        n_kept = n_kept + 1;
        epochs(:,:,n_kept)     = epoch;
        epoch_labels(n_kept)   = stim_labels(i);
    end
    
    % Trim
    epochs       = epochs(:,:,1:n_kept);
    epoch_labels = epoch_labels(1:n_kept);
    
    fprintf('  Epochs kept: %d / %d (rejected: %d, %.1f%%)\n', ...
        n_kept, length(stim_onsets), n_rejected, ...
        100*n_rejected/length(stim_onsets));

    % --- 2.5 Store processed data ---
    results(f).fname        = fname;
    results(f).patient      = patient;
    results(f).cond         = cond;
    results(f).sess         = sess;
    results(f).epochs       = epochs;
    results(f).labels       = epoch_labels;
    results(f).n_rejected   = n_rejected;
    results(f).n_total      = length(stim_onsets);

    fprintf('  Preprocessing done.\n\n');
end

%% =========================================================
% SECTION 3: COMPUTE & PLOT GRAND AVERAGE ERPs
% =========================================================

fprintf('=== Plotting ERPs ===\n');

% Time axis (corrected: sample 1 is at t = (-pre+1)/fs)
time_axis = ((-cfg.epoch_pre + 1) : cfg.epoch_post) / cfg.fs * 1000;

% Output filenames
pdf_perfile  = 'erp_per_file.pdf';
pdf_summary  = 'erp_summary.pdf';
if exist(pdf_perfile, 'file'), delete(pdf_perfile); end
if exist(pdf_summary, 'file'), delete(pdf_summary); end

% Channels to show in per-file plots: C3, Cz, C4 (matches Paper 3 Fig. 2)
key_chans  = [2, 3, 4];
chan_names = cfg.chan_labels(key_chans);

%% ---- 3.1 Diagnostics page (first page of per-file PDF) ----

fig_diag = figure('Position', [100 100 1000 600], 'Color', 'w');
axis off;

diag_lines = {};
diag_lines{end+1} = '=== Preprocessing Diagnostics ===';
diag_lines{end+1} = '';
diag_lines{end+1} = sprintf('%-16s %-4s %-5s %-5s %-8s %-8s %-8s %-10s', ...
    'File','Pat','Cond','Sess','Targets','NonTgt','Distr','Rejected');
diag_lines{end+1} = repmat('-', 1, 75);

for f = 1:n_files
    r = results(f);
    if isempty(r.epochs)
        diag_lines{end+1} = sprintf('%-16s   (no data)', cfg.files{f,1});
        continue;
    end
    n_t = sum(r.labels == cfg.label_target);
    n_n = sum(r.labels == cfg.label_nontarget);
    n_d = sum(r.labels == cfg.label_distractor);
    rej_pct = 100 * r.n_rejected / r.n_total;
    diag_lines{end+1} = sprintf('%-16s %-4s %-5s %-5d %-8d %-8d %-8d %d (%.1f%%)', ...
        r.fname, r.patient, r.cond, r.sess, n_t, n_n, n_d, ...
        r.n_rejected, rej_pct);
end

text(0.05, 0.95, diag_lines, ...
    'FontName','Courier New', 'FontSize', 10, ...
    'VerticalAlignment','top', 'Interpreter','none');

exportgraphics(fig_diag, pdf_perfile, 'Append', true, 'ContentType', 'vector');
close(fig_diag);

%% ---- 3.2 Per-file ERP pages ----

for f = 1:n_files
    if isempty(results(f).epochs), continue; end

    epochs = results(f).epochs;
    labels = results(f).labels;

    target_avg    = mean(epochs(:,:, labels == cfg.label_target),    3);
    nontarget_avg = mean(epochs(:,:, labels == cfg.label_nontarget), 3);

    fig = figure('Position', [100 100 1200 400], 'Color', 'w');

    for c = 1:length(key_chans)
        subplot(1, length(key_chans), c);
        ch = key_chans(c);

        plot(time_axis, target_avg(:,ch),    'g-', 'LineWidth', 2); hold on;
        plot(time_axis, nontarget_avg(:,ch), 'b-', 'LineWidth', 1.5);
        xline(0, 'r--', 'LineWidth', 1);
        yline(0, 'k-');

        xlabel('Time (ms)');
        ylabel('Amplitude (µV)');
        title(chan_names{c});
        if c == 1, legend('Target', 'Non-target', 'Location', 'northeast'); end
        grid on;
        xlim([-cfg.epoch_pre_ms, cfg.epoch_post_ms]);
    end

    sgtitle(sprintf('%s | Patient=%s | Cond=%s | Sess=%d', ...
        results(f).fname, results(f).patient, ...
        results(f).cond, results(f).sess), 'Interpreter','none');

    exportgraphics(fig, pdf_perfile, 'Append', true, 'ContentType', 'vector');
    close(fig);
end

fprintf('  Per-file ERPs saved to %s\n', pdf_perfile);

%% ---- 3.3 One-page summary mosaic ----

fig_sum = figure('Position', [50 50 1600 1400], 'Color', 'w');
t = tiledlayout(n_files, length(key_chans), ...
    'TileSpacing','compact', 'Padding','compact');

for f = 1:n_files
    if isempty(results(f).epochs)
        for c = 1:length(key_chans), nexttile; axis off; end
        continue;
    end

    epochs = results(f).epochs;
    labels = results(f).labels;
    target_avg    = mean(epochs(:,:, labels == cfg.label_target),    3);
    nontarget_avg = mean(epochs(:,:, labels == cfg.label_nontarget), 3);

    for c = 1:length(key_chans)
        nexttile;
        ch = key_chans(c);
        plot(time_axis, target_avg(:,ch),    'g-', 'LineWidth', 1.3); hold on;
        plot(time_axis, nontarget_avg(:,ch), 'b-', 'LineWidth', 1);
        xline(0, 'r--'); yline(0, 'k-');
        grid on;
        xlim([-cfg.epoch_pre_ms, cfg.epoch_post_ms]);

        if c == 1
            ylabel(sprintf('%s\n%s', results(f).patient, ...
                [results(f).cond num2str(results(f).sess)]), ...
                'FontWeight','bold', 'Interpreter','none');
        end
        if f == 1
            title(chan_names{c});
        end
        if f == n_files
            xlabel('Time (ms)');
        end
    end
end

title(t, 'ERP overview: target (green) vs. non-target (blue)', ...
    'FontWeight','bold');

exportgraphics(fig_sum, pdf_summary, 'ContentType', 'vector');
close(fig_sum);

fprintf('  Summary mosaic saved to %s\n', pdf_summary);
fprintf('  ERP plots created.\n\n');

%% =========================================================
% SECTION 4: FEATURE EXTRACTION
% =========================================================

fprintf('=== Extracting features ===\n');

% We will compare 3 feature sets:
%   F1: Raw amplitude (flattened epoch) - BASELINE from papers
%   F2: Downsampled amplitude (every 4th sample)
%   F3: Mean amplitude in time windows (10 x 70ms windows)

for f = 1:n_files
    if ~isfield(results(f), 'epochs') || isempty(results(f).epochs)
        continue;
    end

    epochs = results(f).epochs;  % [epoch_len x channels x trials]
    n_trials = size(epochs, 3);

    % Reshape: [trials x (epoch_len * channels)]
    epochs_2d = reshape(permute(epochs, [3,1,2]), [n_trials, cfg.epoch_len * cfg.n_channels]);

    % F1: Raw amplitude (full resolution)
    results(f).features_raw = epochs_2d;

    % F2: Downsampled (factor 4 = 64 Hz effective)
    ds_factor = 4;
    epochs_ds = epochs(1:ds_factor:end, :, :);
    results(f).features_ds = reshape(permute(epochs_ds,[3,1,2]), ...
        [n_trials, size(epochs_ds,1)*cfg.n_channels]);

    % F3: Mean amplitude in 10 time windows
    n_windows = 10;
    win_len = floor(cfg.epoch_len / n_windows);
    feat_win = zeros(n_trials, n_windows * cfg.n_channels);
    for w = 1:n_windows
        idx_start = (w-1)*win_len + 1;
        idx_end   = w * win_len;
        win_mean  = squeeze(mean(epochs(idx_start:idx_end,:,:), 1))'; % [trials x chans]
        feat_win(:, (w-1)*cfg.n_channels+1 : w*cfg.n_channels) = win_mean;
    end
    results(f).features_win = feat_win;

    fprintf('  %s: features extracted (raw=%d, ds=%d, win=%d dims)\n', ...
        results(f).fname, size(results(f).features_raw,2), ...
        size(results(f).features_ds,2), size(results(f).features_win,2));
end

fprintf('\n');

%%
save('processed_data.mat', 'results', 'cfg');
fprintf('\nSaved processed_data.mat\n');

%% Sanity Check
% --- 2.7 One-time sanity check (first file only) ---
    %% if f == 1
   %   fig_sanity = figure('Position', [200 200 900 500], 'Color', 'w');
    %   subplot(2,1,1);
    %   plot((1:1000)/cfg.fs, y(1:1000, 3));
    %   title(sprintf('Raw signal | %s | Cz', fname), 'Interpreter','none');
    %   ylabel('µV'); grid on;
    %   subplot(2,1,2);
    %   plot((1:1000)/cfg.fs, y_filt(1:1000, 3));
    %   title('After filtering (0.1-30 Hz bandpass + 50 Hz notch)');
    %   xlabel('Time (s)'); ylabel('µV'); grid on;
    %end

    % fprintf('  Preprocessing done.\n\n');
    % end  

    % end of Section 2 loop