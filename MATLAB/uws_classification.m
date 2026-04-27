%% =========================================================
% UWS Vibro-Tactile P300 BCI Analysis Pipeline
% Step 2: Single-Trial AND Group-of-8 Classification
% =========================================================
% Inputs:  processed_data.mat (from uws_preprocessing.m)
% Outputs: classification_results.mat
%          classification_summary.pdf (per-file + aggregated)
%          group_classification_perfile.pdf (standalone)
%          group_classification_aggregated.pdf (standalone)
% =========================================================

clear all; close all; clc;

% Darker text on figures for readability
set(groot, 'DefaultAxesXColor', 'k');
set(groot, 'DefaultAxesYColor', 'k');
set(groot, 'DefaultAxesGridColor', [0.15 0.15 0.15]);
set(groot, 'DefaultAxesFontSize', 11);
set(groot, 'DefaultTextColor', 'k');

%% =========================================================
% SECTION 0: LOAD PREPROCESSED DATA
% =========================================================

fprintf('=== Loading preprocessed data ===\n');

if ~exist('processed_data.mat', 'file')
    error('processed_data.mat not found. Run uws_preprocessing.m first.');
end

load('processed_data.mat', 'results', 'cfg');
n_files = length(results);
fprintf('Loaded %d files.\n\n', n_files);

%% =========================================================
% SECTION 1: CLASSIFICATION CONFIG
% =========================================================

clf_cfg = struct();
clf_cfg.feature_set        = 'win';
clf_cfg.n_folds            = 5;
clf_cfg.n_repeats          = 10;
clf_cfg.exclude_distractor = true;
clf_cfg.classifier         = 'lda';
rng(42);

fprintf('=== Classification config ===\n');
fprintf('Feature set: %s\n', clf_cfg.feature_set);
fprintf('CV: %d-fold × %d repeats\n', clf_cfg.n_folds, clf_cfg.n_repeats);
fprintf('Classifier: %s\n\n', clf_cfg.classifier);

%% =========================================================
% SECTION 2: SINGLE-TRIAL CLASSIFICATION (target vs non-target)
% =========================================================

fprintf('=== Single-trial classification per file ===\n');

clf_results = struct();

for f = 1:n_files
    r = results(f);
    if isempty(r.epochs)
        fprintf('  %s: skipped (no data)\n', r.fname);
        continue;
    end

    fprintf('  %s ... ', r.fname);

    switch clf_cfg.feature_set
        case 'raw',  X = r.features_raw;
        case 'ds',   X = r.features_ds;
        case 'win',  X = r.features_win;
    end
    y_lab = r.labels;

    if clf_cfg.exclude_distractor
        keep = (y_lab == cfg.label_target) | (y_lab == cfg.label_nontarget);
        X = X(keep, :);
        y_lab = y_lab(keep);
    end

    y_bin = (y_lab == cfg.label_target);
    n_targ = sum(y_bin);
    n_nont = sum(~y_bin);

    if n_targ < 10 || n_nont < 10
        fprintf('skipped (too few trials)\n');
        continue;
    end

    auc_per_fold = nan(clf_cfg.n_repeats, clf_cfg.n_folds);
    acc_per_fold = nan(clf_cfg.n_repeats, clf_cfg.n_folds);

    for rep = 1:clf_cfg.n_repeats
        cv = cvpartition(y_bin, 'KFold', clf_cfg.n_folds, 'Stratify', true);

        for k = 1:clf_cfg.n_folds
            train_idx = training(cv, k);
            test_idx  = test(cv, k);

            X_train = X(train_idx, :);
            y_train = y_bin(train_idx);
            X_test  = X(test_idx, :);
            y_test  = y_bin(test_idx);

            mdl = fitcdiscr(X_train, y_train, ...
                'DiscrimType', 'pseudoLinear', 'Prior', 'uniform');
            [pred, score] = predict(mdl, X_test);

            acc_per_fold(rep, k) = mean(pred == y_test);
            if length(unique(y_test)) == 2
                [~,~,~,auc] = perfcurve(y_test, score(:,2), true);
                auc_per_fold(rep, k) = auc;
            end
        end
    end

    clf_results(f).fname    = r.fname;
    clf_results(f).patient  = r.patient;
    clf_results(f).cond     = r.cond;
    clf_results(f).sess     = r.sess;
    clf_results(f).n_target    = n_targ;
    clf_results(f).n_nontarget = n_nont;
    clf_results(f).mean_auc = mean(auc_per_fold(:), 'omitnan');
    clf_results(f).std_auc  = std(auc_per_fold(:),  'omitnan');
    clf_results(f).mean_acc = mean(acc_per_fold(:), 'omitnan');
    clf_results(f).std_acc  = std(acc_per_fold(:),  'omitnan');
    clf_results(f).auc_per_fold = auc_per_fold;
    clf_results(f).acc_per_fold = acc_per_fold;

    fprintf('AUC=%.3f±%.3f  (n_t=%d, n_nt=%d)\n', ...
        clf_results(f).mean_auc, clf_results(f).std_auc, n_targ, n_nont);
end
fprintf('\n');

%% =========================================================
% SECTION 3: PERMUTATION TEST FOR SINGLE-TRIAL AUC
% =========================================================

fprintf('=== Permutation test (n=200 shuffles per file) ===\n');
n_perm = 200;

for f = 1:n_files
    if isempty(clf_results(f).fname), continue; end
    r = results(f);

    switch clf_cfg.feature_set
        case 'raw',  X = r.features_raw;
        case 'ds',   X = r.features_ds;
        case 'win',  X = r.features_win;
    end
    y_lab = r.labels;
    if clf_cfg.exclude_distractor
        keep = (y_lab == cfg.label_target) | (y_lab == cfg.label_nontarget);
        X = X(keep, :);
        y_lab = y_lab(keep);
    end
    y_bin = (y_lab == cfg.label_target);

    perm_aucs = nan(n_perm, 1);
    for p = 1:n_perm
        y_shuf = y_bin(randperm(length(y_bin)));
        cv = cvpartition(y_shuf, 'KFold', 5, 'Stratify', true);
        fold_aucs = nan(5,1);
        for k = 1:5
            tr = training(cv, k); te = test(cv, k);
            mdl = fitcdiscr(X(tr,:), y_shuf(tr), ...
                'DiscrimType', 'pseudoLinear', 'Prior', 'uniform');
            [~, s] = predict(mdl, X(te,:));
            if length(unique(y_shuf(te))) == 2
                [~,~,~,a] = perfcurve(y_shuf(te), s(:,2), true);
                fold_aucs(k) = a;
            end
        end
        perm_aucs(p) = mean(fold_aucs, 'omitnan');
    end

    clf_results(f).perm_p         = mean(perm_aucs >= clf_results(f).mean_auc);
    clf_results(f).perm_null_mean = mean(perm_aucs);
    clf_results(f).perm_null_95   = quantile(perm_aucs, 0.95);

    fprintf('  %s: AUC=%.3f, null mean=%.3f, 95th=%.3f, p=%.3f\n', ...
        clf_results(f).fname, clf_results(f).mean_auc, ...
        clf_results(f).perm_null_mean, clf_results(f).perm_null_95, ...
        clf_results(f).perm_p);
end
fprintf('\n');

%% =========================================================
% SECTION 4: GROUP-OF-8 CLASSIFICATION (paper-faithful)
% =========================================================
% In VT3, stimuli arrive in groups of 8: 1 target, 1 non-target, 6 distractors.
% Paper's task: given one such group, predict which trial was the target.
% Chance = 1/8 = 12.5%. 95% binomial significance threshold ≈ 23%.

fprintf('=== Group-of-8 classification (leave-one-group-out CV) ===\n');

group_results = struct();

for f = 1:n_files
    r = results(f);
    if isempty(r.epochs), continue; end

    fprintf('  %s ... ', r.fname);

    raw = load(r.fname);
    trig = raw.trig;

    stim_onsets = []; stim_labels = [];
    for cls = [cfg.label_distractor, cfg.label_nontarget, cfg.label_target]
        onsets = find(diff([0; trig == cls]) == 1);
        stim_onsets = [stim_onsets; onsets];
        stim_labels = [stim_labels; cls * ones(size(onsets))];
    end
    [stim_onsets, sort_idx] = sort(stim_onsets);
    stim_labels = stim_labels(sort_idx);

    n_stim = length(stim_onsets);
    n_groups = floor(n_stim / 8);

    group_idx = zeros(n_stim, 1);
    valid_group = false(n_groups, 1);

    for g = 1:n_groups
        idx = (g-1)*8 + 1 : g*8;
        group_idx(idx) = g;
        labs = stim_labels(idx);
        if sum(labs == cfg.label_target) == 1 && ...
           sum(labs == cfg.label_nontarget) == 1 && ...
           sum(labs == cfg.label_distractor) == 6
            valid_group(g) = true;
        end
    end

    n_valid = sum(valid_group);
    fprintf('groups=%d (valid=%d) ... ', n_groups, n_valid);

    if n_valid < 10
        fprintf('skipped (too few valid groups)\n');
        continue;
    end

    % Align kept trials back to original onsets
    n_samples_total = size(raw.y, 1);
    pre  = cfg.epoch_pre;
    post = cfg.epoch_post;

    kept_to_orig = nan(size(r.epochs, 3), 1);
    j = 1;
    for i = 1:n_stim
        if j > length(r.labels), break; end
        if stim_onsets(i) - pre + 1 < 1 || stim_onsets(i) + post > n_samples_total
            continue;
        end
        if stim_labels(i) == r.labels(j)
            kept_to_orig(j) = i;
            j = j + 1;
        end
    end

    if any(isnan(kept_to_orig))
        fprintf('alignment failed\n');
        continue;
    end

    kept_group_idx = group_idx(kept_to_orig);

    switch clf_cfg.feature_set
        case 'raw',  X = r.features_raw;
        case 'ds',   X = r.features_ds;
        case 'win',  X = r.features_win;
    end
    y_lab = r.labels;
    y_target = (y_lab == cfg.label_target);

    valid_group_ids = find(valid_group);
    n_correct = 0;
    n_tested  = 0;

    for vg = 1:length(valid_group_ids)
        gid = valid_group_ids(vg);

        test_mask  = (kept_group_idx == gid);
        train_mask = ~test_mask & ~isnan(kept_group_idx);

        if sum(test_mask) < 2, continue; end
        if sum(y_target(test_mask) == 1) ~= 1, continue; end
        if sum(y_target(train_mask)) < 5 || sum(~y_target(train_mask)) < 5
            continue;
        end

        X_train = X(train_mask, :);
        y_train = y_target(train_mask);
        X_test  = X(test_mask, :);
        y_test  = y_target(test_mask);

        mdl = fitcdiscr(X_train, y_train, ...
            'DiscrimType', 'pseudoLinear', 'Prior', 'uniform');
        [~, score] = predict(mdl, X_test);
        target_score = score(:, 2);

        [~, pred_idx] = max(target_score);
        true_idx = find(y_test == 1);

        if pred_idx == true_idx
            n_correct = n_correct + 1;
        end
        n_tested = n_tested + 1;
    end

    if n_tested == 0
        fprintf('no testable groups\n');
        continue;
    end

    group_acc = n_correct / n_tested;

    group_results(f).fname           = r.fname;
    group_results(f).patient         = r.patient;
    group_results(f).cond            = r.cond;
    group_results(f).sess            = r.sess;
    group_results(f).n_groups_tested = n_tested;
    group_results(f).n_correct       = n_correct;
    group_results(f).group_acc       = group_acc;

    fprintf('acc=%.1f%% (%d/%d, chance=12.5%%)\n', ...
        100*group_acc, n_correct, n_tested);
end
fprintf('\n');

%% =========================================================
% SECTION 5: SUMMARY FIGURES (per-file + patient-level aggregated)
% =========================================================

fprintf('=== Building summary figures ===\n');

% --- Collect data ---
fnames     = {};
patients   = {};
conds      = {};
aucs       = [];
auc_stds   = [];
group_accs = [];
pvals      = [];

for f = 1:n_files
    if isempty(clf_results(f).fname), continue; end
    fnames{end+1}   = clf_results(f).fname;
    patients{end+1} = clf_results(f).patient;
    conds{end+1}    = clf_results(f).cond;
    aucs(end+1)     = clf_results(f).mean_auc;
    auc_stds(end+1) = clf_results(f).std_auc;
    pvals(end+1)    = clf_results(f).perm_p;

    if length(group_results) >= f && ~isempty(group_results(f)) && ...
            isfield(group_results(f),'group_acc') && ~isempty(group_results(f).group_acc)
        group_accs(end+1) = 100 * group_results(f).group_acc;
    else
        group_accs(end+1) = NaN;
    end
end

% --- Color per file by condition ---
colors = zeros(length(aucs), 3);
for i = 1:length(aucs)
    if strcmp(conds{i}, 'high')
        colors(i,:) = [0.2 0.6 0.2];
    else
        colors(i,:) = [0.7 0.3 0.3];
    end
end

% --- FIGURE 1: per-file group-of-8 accuracy ---
fig1 = figure('Position', [100 100 1100 500], 'Color', 'w');
hold on;
x = 1:length(group_accs);
for i = 1:length(group_accs)
    bar(x(i), group_accs(i), 'FaceColor', colors(i,:), 'EdgeColor','none');
end
yline(12.5, 'k--', 'Chance (12.5%)', 'LineWidth', 1.2);
yline(23,   'b:',  '95% sig (23%)',  'LineWidth', 1.2);

h_high = bar(NaN, NaN, 'FaceColor', [0.2 0.6 0.2], 'EdgeColor','none');
h_low  = bar(NaN, NaN, 'FaceColor', [0.7 0.3 0.3], 'EdgeColor','none');
legend([h_high, h_low], {'high (>95%)', 'low (<5%)'}, 'Location', 'northeast');

set(gca, 'XTick', x, 'XTickLabel', fnames, 'TickLabelInterpreter','none');
xtickangle(35);
ylabel('Group-of-8 accuracy (%)');
ylim([0 100]);
title('Per-file group-of-8 classification (LOGO-CV)');
grid on;

% Save to multi-page PDF + standalone PDF
if exist('classification_summary.pdf','file'), delete('classification_summary.pdf'); end
exportgraphics(fig1, 'classification_summary.pdf', 'ContentType','vector');
exportgraphics(fig1, 'group_classification_perfile.pdf', 'ContentType','vector');
fprintf('  Saved per-file chart\n');

% --- FIGURE 2: patient × condition aggregated ---
fig2 = figure('Position', [100 100 700 500], 'Color', 'w');
hold on;

unique_pats = unique(patients);
group_means = nan(length(unique_pats), 2);
group_stds  = nan(length(unique_pats), 2);

for pi = 1:length(unique_pats)
    p = unique_pats{pi};
    mask_low  = strcmp(patients, p) & strcmp(conds, 'low');
    mask_high = strcmp(patients, p) & strcmp(conds, 'high');
    group_means(pi, 1) = mean(group_accs(mask_low),  'omitnan');
    group_means(pi, 2) = mean(group_accs(mask_high), 'omitnan');
    group_stds(pi, 1)  = std(group_accs(mask_low),   'omitnan');
    group_stds(pi, 2)  = std(group_accs(mask_high),  'omitnan');
end

b = bar(group_means);
b(1).FaceColor = [0.7 0.3 0.3];
b(2).FaceColor = [0.2 0.6 0.2];

% Error bars
[ngroups, nbars] = size(group_means);
x_err = nan(nbars, ngroups);
for i = 1:nbars
    x_err(i,:) = b(i).XEndPoints;
end
errorbar(x_err', group_means, group_stds, 'k.', 'LineWidth', 1, 'CapSize', 8);

yline(12.5, 'k--', 'Chance (12.5%)');
yline(23,   'b:',  '95% sig (23%)');
set(gca, 'XTick', 1:length(unique_pats), 'XTickLabel', unique_pats);
ylabel('Mean accuracy across runs (%)');
ylim([0 100]);
title('Patient × condition (mean ± SD across runs)');
legend({'Low','High','Chance','95% sig'}, 'Location', 'northeast');
grid on;

exportgraphics(fig2, 'classification_summary.pdf', 'Append', true, 'ContentType','vector');
exportgraphics(fig2, 'group_classification_aggregated.pdf', 'ContentType','vector');
fprintf('  Saved aggregated chart\n');

%% =========================================================
% SECTION 6: SAVE RESULTS
% =========================================================

save('classification_results.mat', 'clf_results', 'clf_cfg', 'group_results');
fprintf('  Saved classification_results.mat\n\n');

%% =========================================================
% SECTION 7: PRINT FINAL TABLE
% =========================================================

fprintf('=== FINAL SUMMARY ===\n');
fprintf('%-16s %-4s %-5s %-9s %-9s %-9s %-9s\n', ...
    'File', 'Pat', 'Cond', 'AUC', '±SD', 'p-val', 'Group %');
fprintf('%s\n', repmat('-', 1, 65));

for f = 1:n_files
    if isempty(clf_results(f).fname), continue; end
    r = clf_results(f);

    if length(group_results) >= f && ~isempty(group_results(f)) && ...
            isfield(group_results(f),'group_acc') && ~isempty(group_results(f).group_acc)
        ga_str = sprintf('%.1f%%', 100 * group_results(f).group_acc);
        ga_sig = ''; if group_results(f).group_acc > 0.23, ga_sig = '*'; end
    else
        ga_str = 'NaN'; ga_sig = '';
    end

    sig = ''; if r.perm_p < 0.05, sig = '*'; end

    fprintf('%-16s %-4s %-5s %-9.3f %-9.3f %-9.3f %-9s%s\n', ...
        r.fname, r.patient, r.cond, ...
        r.mean_auc, r.std_auc, r.perm_p, ga_str, ga_sig);
end
fprintf('\nGroup chance = 12.5%%, significance threshold = 23%%\n');
fprintf('* = above significance threshold\n\n');