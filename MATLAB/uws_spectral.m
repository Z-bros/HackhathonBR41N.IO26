%% =========================================================
% UWS Vibro-Tactile P300 BCI Analysis Pipeline
% Step 3: Spectral Analysis
% =========================================================
% Inputs:  processed_data.mat         (from uws_preprocessing.m)
%          classification_results.mat (from uws_classification.m)
% Outputs: spectral_results.mat
%          spectral_summary.pdf
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
% SECTION 0: LOAD INPUTS
% =========================================================

fprintf('=== Loading data ===\n');

if ~exist('processed_data.mat', 'file')
    error('processed_data.mat not found. Run uws_preprocessing.m first.');
end
load('processed_data.mat', 'results', 'cfg');

% Classification results (used in Section 3 to identify "good" trials)
have_clf = exist('classification_results.mat', 'file') == 2;
if have_clf
    load('classification_results.mat', 'clf_results', 'group_results');
    fprintf('Classification results loaded.\n');
else
    fprintf('No classification results found — Section 3 will be skipped.\n');
end

n_files = length(results);
fprintf('Loaded %d files.\n\n', n_files);

%% =========================================================
% SECTION 1: SPECTRAL CONFIG
% =========================================================

spec_cfg = struct();

% --- Frequency bands of interest ---
spec_cfg.bands = struct( ...
    'delta', [1 4], ...
    'theta', [4 8], ...
    'alpha', [8 12], ...
    'beta',  [13 25] );
spec_cfg.band_names = fieldnames(spec_cfg.bands);

% --- Welch parameters for continuous PSD ---
spec_cfg.welch_window_s   = 2;        % 2-second windows
spec_cfg.welch_overlap    = 0.5;      % 50% overlap
spec_cfg.welch_nfft       = 512;      % FFT length
spec_cfg.freq_range_plot  = [1 30];   % Hz

% --- Pre-stimulus epoch for trial-by-trial spectral features ---
% We re-epoch the filtered data to get a longer pre-stimulus window
% than the 100 ms used for ERP baseline. 500 ms gives enough samples
% for reliable band power estimation at low frequencies.
spec_cfg.prestim_ms = 500;
spec_cfg.prestim_samples = round(spec_cfg.prestim_ms / 1000 * cfg.fs);

% --- Spectrogram parameters ---
spec_cfg.spec_window_ms = 200;
spec_cfg.spec_overlap   = 0.9;        % heavy overlap for smooth spectrograms
spec_cfg.spec_window    = round(spec_cfg.spec_window_ms / 1000 * cfg.fs);
spec_cfg.spec_noverlap  = round(spec_cfg.spec_window * spec_cfg.spec_overlap);
spec_cfg.spec_nfft      = 256;

% --- Channels of interest for spectral analysis ---
% Pz / CPz / Cz are the strongest P300 sites. Pz also has the strongest
% posterior alpha — relevant for the vigilance hypothesis.
spec_cfg.spectral_chans = [3, 6, 8];   % Cz, CPz, Pz
spec_cfg.spectral_chan_names = cfg.chan_labels(spec_cfg.spectral_chans);

fprintf('=== Spectral config ===\n');
fprintf('Bands: delta(1-4), theta(4-8), alpha(8-12), beta(13-25) Hz\n');
fprintf('Pre-stim window: %d ms (%d samples)\n', ...
    spec_cfg.prestim_ms, spec_cfg.prestim_samples);
fprintf('Spectral channels: %s\n\n', strjoin(spec_cfg.spectral_chan_names, ', '));

%% =========================================================
% SECTION 2: CONTINUOUS-RECORDING PSD (Welch's method)
% =========================================================
% Question: do high-accuracy runs differ in baseline EEG state from
% low-accuracy runs, looking at the WHOLE recording?

fprintf('=== Computing continuous PSD per file ===\n');

psd_results = struct();

for f = 1:n_files
    r = results(f);
    if isempty(r.epochs), continue; end

    % Reload original file to get continuous filtered signal
    % (We need to refilter because results stores epochs only.)
    raw = load(r.fname);
    y = raw.y;

    % Refilter (same as preprocessing — bandpass + notch)
    y_filt = zeros(size(y));
    for ch = 1:cfg.n_channels
        tmp = filtfilt(cfg.notch_b, cfg.notch_a, y(:,ch));
        y_filt(:,ch) = filtfilt(cfg.bp_filter, tmp);
    end

    % Welch on each channel
    win_samples = spec_cfg.welch_window_s * cfg.fs;
    overlap_samples = round(win_samples * spec_cfg.welch_overlap);

    psds = [];
    for ch = 1:cfg.n_channels
        [pxx, freqs] = pwelch(y_filt(:,ch), win_samples, overlap_samples, ...
            spec_cfg.welch_nfft, cfg.fs);
        psds(:,ch) = pxx;
    end

    % Band power: integrate PSD within each band
    band_power = struct();
    for b = 1:length(spec_cfg.band_names)
        bn = spec_cfg.band_names{b};
        band_range = spec_cfg.bands.(bn);
        idx = freqs >= band_range(1) & freqs <= band_range(2);
        band_power.(bn) = sum(psds(idx, :), 1);   % per channel
    end

    psd_results(f).fname = r.fname;
    psd_results(f).patient = r.patient;
    psd_results(f).cond = r.cond;
    psd_results(f).sess = r.sess;
    psd_results(f).freqs = freqs;
    psd_results(f).psds = psds;
    psd_results(f).band_power = band_power;

    fprintf('  %s: alpha(Pz)=%.2f, delta(Pz)=%.2f µV²\n', ...
        r.fname, band_power.alpha(8), band_power.delta(8));
end

fprintf('\n');

%% =========================================================
% SECTION 3: PRE-STIMULUS ALPHA — does it predict trial quality?
% =========================================================
% Question: within a single file, does pre-stimulus alpha power on a
% given trial predict whether that trial is correctly classified?
%
% This is the vigilance hypothesis: high pre-stim alpha = drowsy /
% inattentive = trial likely misclassified.

fprintf('=== Pre-stimulus alpha analysis ===\n');

prestim_results = struct();

for f = 1:n_files
    r = results(f);
    if isempty(r.epochs), continue; end

    fprintf('  %s ... ', r.fname);

    % Re-epoch the filtered data to extract a 500ms pre-stimulus window.
    % We need to redo this from continuous data because the existing
    % epochs only have 100 ms of pre-stimulus.
    raw = load(r.fname);
    y_filt = zeros(size(raw.y));
    for ch = 1:cfg.n_channels
        tmp = filtfilt(cfg.notch_b, cfg.notch_a, raw.y(:,ch));
        y_filt(:,ch) = filtfilt(cfg.bp_filter, tmp);
    end

    trig = raw.trig;
    stim_onsets = [];
    stim_labels = [];
    for cls = [cfg.label_distractor, cfg.label_nontarget, cfg.label_target]
        onsets = find(diff([0; trig == cls]) == 1);
        stim_onsets = [stim_onsets; onsets];
        stim_labels = [stim_labels; cls * ones(size(onsets))];
    end
    [stim_onsets, sort_idx] = sort(stim_onsets);
    stim_labels = stim_labels(sort_idx);

    % Extract pre-stimulus epochs (only those that fit and pass artifact)
    pre = spec_cfg.prestim_samples;
    n_chans = cfg.n_channels;
    prestim_epochs = [];
    prestim_labels = [];

    for i = 1:length(stim_onsets)
        onset = stim_onsets(i);
        start_idx = onset - pre;
        end_idx = onset - 1;
        if start_idx < 1, continue; end

        epoch = y_filt(start_idx:end_idx, :);
        if max(abs(epoch(:))) > cfg.artifact_thresh, continue; end

        prestim_epochs = cat(3, prestim_epochs, epoch);
        prestim_labels = [prestim_labels; stim_labels(i)];
    end

    n_trials = size(prestim_epochs, 3);

    % Compute band power per trial per channel (pwelch on each epoch)
    band_power_trial = struct();
    for b = 1:length(spec_cfg.band_names)
        band_power_trial.(spec_cfg.band_names{b}) = nan(n_trials, n_chans);
    end

    for t = 1:n_trials
        for ch = 1:n_chans
            % Use pwelch with a single window (whole epoch)
            [pxx, freqs] = pwelch(prestim_epochs(:, ch, t), ...
                pre, 0, spec_cfg.welch_nfft, cfg.fs);
            for b = 1:length(spec_cfg.band_names)
                bn = spec_cfg.band_names{b};
                band_range = spec_cfg.bands.(bn);
                idx = freqs >= band_range(1) & freqs <= band_range(2);
                band_power_trial.(bn)(t, ch) = sum(pxx(idx));
            end
        end
    end

    % Log-transform (band power is heavily right-skewed)
    for b = 1:length(spec_cfg.band_names)
        bn = spec_cfg.band_names{b};
        band_power_trial.(bn) = log10(band_power_trial.(bn) + eps);
    end

    prestim_results(f).fname = r.fname;
    prestim_results(f).patient = r.patient;
    prestim_results(f).cond = r.cond;
    prestim_results(f).band_power = band_power_trial;
    prestim_results(f).labels = prestim_labels;
    prestim_results(f).n_trials = n_trials;

    % Quick summary: mean alpha at Pz for target vs non-target
    alpha_pz = band_power_trial.alpha(:, 8);
    is_target = (prestim_labels == cfg.label_target);
    mean_alpha_t  = mean(alpha_pz(is_target), 'omitnan');
    mean_alpha_nt = mean(alpha_pz(~is_target & prestim_labels ~= cfg.label_distractor), 'omitnan');

    fprintf('alpha@Pz: T=%.2f, NT=%.2f log(µV²)\n', mean_alpha_t, mean_alpha_nt);
end

fprintf('\n');

%% =========================================================
% SECTION 4: STIMULUS-LOCKED TIME-FREQUENCY (spectrograms)
% =========================================================
% Question: does the stimulus-locked oscillatory response differ between
% target and non-target, and between high and low conditions?

fprintf('=== Computing event-related spectrograms ===\n');

tf_results = struct();

for f = 1:n_files
    r = results(f);
    if isempty(r.epochs), continue; end

    epochs = r.epochs;          % [time x chan x trial]
    labels = r.labels;
    is_target    = (labels == cfg.label_target);
    is_nontarget = (labels == cfg.label_nontarget);

    % We'll compute spectrograms only for Pz (channel 8) to keep things tractable
    ch = 8;

    % Compute spectrogram for each trial, then average within class
    [~, spec_freqs, spec_times, ~] = spectrogram(epochs(:,ch,1), ...
        spec_cfg.spec_window, spec_cfg.spec_noverlap, ...
        spec_cfg.spec_nfft, cfg.fs);

    n_target    = sum(is_target);
    n_nontarget = sum(is_nontarget);

    target_specs    = zeros(length(spec_freqs), length(spec_times), n_target);
    nontarget_specs = zeros(length(spec_freqs), length(spec_times), n_nontarget);

    t_idx = find(is_target);
    nt_idx = find(is_nontarget);

    for i = 1:n_target
        [~,~,~,P] = spectrogram(epochs(:,ch,t_idx(i)), ...
            spec_cfg.spec_window, spec_cfg.spec_noverlap, ...
            spec_cfg.spec_nfft, cfg.fs);
        target_specs(:,:,i) = P;
    end
    for i = 1:n_nontarget
        [~,~,~,P] = spectrogram(epochs(:,ch,nt_idx(i)), ...
            spec_cfg.spec_window, spec_cfg.spec_noverlap, ...
            spec_cfg.spec_nfft, cfg.fs);
        nontarget_specs(:,:,i) = P;
    end

    % Average across trials, log-transform for visualization
    target_spec_mean    = log10(mean(target_specs, 3) + eps);
    nontarget_spec_mean = log10(mean(nontarget_specs, 3) + eps);

    % Restrict to display freq range
    freq_mask = spec_freqs >= 1 & spec_freqs <= 30;

    tf_results(f).fname = r.fname;
    tf_results(f).patient = r.patient;
    tf_results(f).cond = r.cond;
    tf_results(f).freqs = spec_freqs(freq_mask);
    % Time axis: spec_times is in seconds from start of epoch.
    % Epoch starts at -100ms relative to stimulus, so subtract pre-stim duration.
    tf_results(f).times_ms = (spec_times - cfg.epoch_pre_ms/1000) * 1000;
    tf_results(f).target_spec    = target_spec_mean(freq_mask, :);
    tf_results(f).nontarget_spec = nontarget_spec_mean(freq_mask, :);
    tf_results(f).difference     = tf_results(f).target_spec - tf_results(f).nontarget_spec;

    fprintf('  %s: TF computed (%d targets, %d non-targets)\n', ...
        r.fname, n_target, n_nontarget);
end

fprintf('\n');

%% =========================================================
% SECTION 5: SUMMARY FIGURES
% =========================================================

fprintf('=== Building summary figures ===\n');

pdf_out = 'spectral_summary.pdf';
if exist(pdf_out, 'file'), delete(pdf_out); end

%% ---- 5.1 PSD per condition, per patient ----

fig = figure('Position', [50 50 1200 800], 'Color', 'w');
patients_unique = unique({results.patient});
patients_unique = patients_unique(~cellfun(@isempty, patients_unique));

n_pat = length(patients_unique);
plot_chans = spec_cfg.spectral_chans;

for pi = 1:n_pat
    pat = patients_unique{pi};

    for ci = 1:length(plot_chans)
        ch = plot_chans(ci);
        subplot(n_pat, length(plot_chans), (pi-1)*length(plot_chans) + ci);
        hold on;

        for f = 1:n_files
            if isempty(psd_results(f)) || ~isfield(psd_results(f),'patient') || ...
                    isempty(psd_results(f).patient), continue; end
            if ~strcmp(psd_results(f).patient, pat), continue; end

            color = [0.7 0.3 0.3];   % low = red
            if strcmp(psd_results(f).cond, 'high')
                color = [0.2 0.6 0.2];
            end

            freqs = psd_results(f).freqs;
            mask = freqs >= spec_cfg.freq_range_plot(1) & ...
                   freqs <= spec_cfg.freq_range_plot(2);

            plot(freqs(mask), 10*log10(psd_results(f).psds(mask, ch)), ...
                'Color', color, 'LineWidth', 1.2);
        end

        % Shade frequency bands lightly for reference
        ylims = ylim;
        for b = 1:length(spec_cfg.band_names)
            bn = spec_cfg.band_names{b};
            br = spec_cfg.bands.(bn);
            patch([br(1) br(2) br(2) br(1)], ...
                [ylims(1) ylims(1) ylims(2) ylims(2)], ...
                'k', 'FaceAlpha', 0.03, 'EdgeColor', 'none');
        end

        xlabel('Frequency (Hz)');
        ylabel('Power (dB)');
        title(sprintf('%s — %s', pat, cfg.chan_labels{ch}));
        xlim(spec_cfg.freq_range_plot);
        grid on;
        if pi == 1 && ci == 1
            legend({'low','low','high','high'}, 'Location', 'best');
        end
    end
end

sgtitle('Continuous PSD: high (green) vs low (red) per patient', ...
    'FontWeight', 'bold');
exportgraphics(fig, pdf_out, 'Append', true, 'ContentType', 'vector');
close(fig);

%% ---- 5.2 Pre-stimulus alpha boxplot per file ----

fig = figure('Position', [50 50 1200 500], 'Color', 'w');

% Collect alpha-Pz values per file for boxplot
all_alpha = [];
all_groups = {};
all_colors = [];

for f = 1:n_files
    if isempty(prestim_results(f)) || ~isfield(prestim_results(f),'fname') || ...
            isempty(prestim_results(f).fname), continue; end

    alpha_pz = prestim_results(f).band_power.alpha(:, 8);
    all_alpha = [all_alpha; alpha_pz];
    all_groups = [all_groups; repmat({prestim_results(f).fname}, length(alpha_pz), 1)];
end

if ~isempty(all_alpha)
    boxplot(all_alpha, all_groups, 'Symbol','.');
    xtickangle(35);
    ylabel('log_{10} alpha power at Pz (8-12 Hz)');
    title('Pre-stimulus alpha power per file');
    grid on;
    set(gca, 'TickLabelInterpreter', 'none');
end

exportgraphics(fig, pdf_out, 'Append', true, 'ContentType', 'vector');
close(fig);

%% ---- 5.3 Spectrograms: target/non-target/difference, all files ----

for f = 1:n_files
    if isempty(tf_results(f)) || ~isfield(tf_results(f),'fname') || ...
            isempty(tf_results(f).fname), continue; end

    fig = figure('Position', [50 50 1400 400], 'Color', 'w');

    times = tf_results(f).times_ms;
    freqs = tf_results(f).freqs;

    subplot(1,3,1);
    imagesc(times, freqs, tf_results(f).target_spec); axis xy;
    xline(0, 'w--', 'LineWidth', 1.5);
    xlabel('Time (ms)'); ylabel('Frequency (Hz)');
    title('Target'); colorbar;

    subplot(1,3,2);
    imagesc(times, freqs, tf_results(f).nontarget_spec); axis xy;
    xline(0, 'w--', 'LineWidth', 1.5);
    xlabel('Time (ms)');
    title('Non-target'); colorbar;

    subplot(1,3,3);
    imagesc(times, freqs, tf_results(f).difference); axis xy;
    xline(0, 'k--', 'LineWidth', 1.5);
    xlabel('Time (ms)');
    title('Target − Non-target'); colorbar;
    colormap(gca, 'parula');

    sgtitle(sprintf('%s | Patient=%s | Cond=%s — Pz spectrogram', ...
        tf_results(f).fname, tf_results(f).patient, tf_results(f).cond), ...
        'Interpreter','none');

    exportgraphics(fig, pdf_out, 'Append', true, 'ContentType','vector');
    close(fig);
end

fprintf('  Saved %s\n', pdf_out);

%% =========================================================
% SECTION 6: SAVE RESULTS
% =========================================================

save('spectral_results.mat', 'psd_results', 'prestim_results', ...
    'tf_results', 'spec_cfg');
fprintf('  Saved spectral_results.mat\n\n');

%% =========================================================
% SECTION 7: PRINT BAND-POWER SUMMARY
% =========================================================

fprintf('=== Band power summary (whole-recording, channel Pz) ===\n');
fprintf('%-16s %-4s %-5s %-8s %-8s %-8s %-8s\n', ...
    'File','Pat','Cond','delta','theta','alpha','beta');
fprintf('%s\n', repmat('-', 1, 65));
for f = 1:n_files
    if isempty(psd_results(f)) || ~isfield(psd_results(f),'fname') || ...
            isempty(psd_results(f).fname), continue; end
    p = psd_results(f);
    fprintf('%-16s %-4s %-5s %-8.2f %-8.2f %-8.2f %-8.2f\n', ...
        p.fname, p.patient, p.cond, ...
        p.band_power.delta(8), p.band_power.theta(8), ...
        p.band_power.alpha(8), p.band_power.beta(8));
end
fprintf('\n');

%% =========================================================
% SECTION 8: STATISTICAL TEST — alpha by condition, per patient
% =========================================================

fprintf('=== Mann-Whitney U test: pre-stim alpha at Pz, high vs low ===\n');

for pat_idx = 1:length(patients_unique)
    pat = patients_unique{pat_idx};

    alpha_high = [];
    alpha_low  = [];

    for f = 1:n_files
        if isempty(prestim_results(f)) || ~isfield(prestim_results(f),'patient'), continue; end
        if ~strcmp(prestim_results(f).patient, pat), continue; end

        a = prestim_results(f).band_power.alpha(:, 8);   % Pz
        if strcmp(prestim_results(f).cond, 'high')
            alpha_high = [alpha_high; a];
        else
            alpha_low  = [alpha_low; a];
        end
    end

    if isempty(alpha_high) || isempty(alpha_low), continue; end

    [p_val, ~, stats] = ranksum(alpha_high, alpha_low);

    fprintf('  %s: high median=%.3f (n=%d), low median=%.3f (n=%d), ranksum p=%.4f\n', ...
        pat, median(alpha_high), length(alpha_high), ...
        median(alpha_low), length(alpha_low), p_val);
end
fprintf('\n');