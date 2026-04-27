%% =========================================================
% UWS Vibro-Tactile P300 BCI Analysis Pipeline
% Step 5: Classifier Comparison (Methodological Variants)
% =========================================================
% Compares: pseudoLinear LDA (baseline) vs shrinkage LDA vs
%           xDAWN+LDA vs Riemannian MDM
% Metric: group-of-8 accuracy with leave-one-group-out CV
% Inputs:  processed_data.mat
% Outputs: classifier_comparison.pdf, classifier_comparison.mat
% =========================================================

clear all; close all; clc;

set(groot, 'DefaultAxesXColor', 'k');
set(groot, 'DefaultAxesYColor', 'k');
set(groot, 'DefaultAxesFontSize', 11);
set(groot, 'DefaultTextColor', 'k');

load('processed_data.mat', 'results', 'cfg');
n_files = length(results);
rng(42);

%% =========================================================
% SECTION 1: HELPER — group-of-8 evaluation for arbitrary classifier
% =========================================================
% Wraps the LOGO-CV procedure so we can plug in different classifiers.
% A "classifier" here is a function handle that takes (X_train, y_train,
% X_test) and returns (target_score_per_test_trial).

function group_acc = evaluate_group_of_8(epochs, labels, fname, cfg, classifier_fn)
    % Reload to get group structure
    raw = load(fname);
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

    % Align kept trials to original stim indices
    pre = cfg.epoch_pre; post = cfg.epoch_post;
    n_total = size(raw.y, 1);
    kept_to_orig = nan(size(epochs, 3), 1);
    j = 1;
    for i = 1:n_stim
        if j > length(labels), break; end
        if stim_onsets(i) - pre + 1 < 1 || stim_onsets(i) + post > n_total
            continue;
        end
        if stim_labels(i) == labels(j)
            kept_to_orig(j) = i;
            j = j + 1;
        end
    end
    if any(isnan(kept_to_orig))
        group_acc = NaN; return;
    end
    kept_group_idx = group_idx(kept_to_orig);
    y_target = (labels == cfg.label_target);

    % LOGO-CV
    valid_ids = find(valid_group);
    n_correct = 0; n_tested = 0;
    for vg = 1:length(valid_ids)
        gid = valid_ids(vg);
        test_mask = (kept_group_idx == gid);
        train_mask = ~test_mask & ~isnan(kept_group_idx);
        if sum(test_mask) < 2, continue; end
        if sum(y_target(test_mask) == 1) ~= 1, continue; end
        if sum(y_target(train_mask)) < 5 || sum(~y_target(train_mask)) < 5
            continue;
        end

        % Pass epoch tensor to classifier (some methods need raw epochs,
        % not just feature vectors)
        epochs_train = epochs(:,:,train_mask);
        epochs_test  = epochs(:,:,test_mask);
        y_train      = y_target(train_mask);
        y_test       = y_target(test_mask);

        try
            target_scores = classifier_fn(epochs_train, y_train, epochs_test);
        catch ME
            warning('Classifier failed: %s', ME.message);
            continue;
        end

        [~, pred_idx] = max(target_scores);
        true_idx = find(y_test == 1);
        if pred_idx == true_idx, n_correct = n_correct + 1; end
        n_tested = n_tested + 1;
    end

    if n_tested == 0
        group_acc = NaN;
    else
        group_acc = n_correct / n_tested;
    end
end

%% =========================================================
% SECTION 2: CLASSIFIER IMPLEMENTATIONS
% =========================================================

% --- Classifier 1: pseudoLinear LDA on windowed features (baseline) ---
function scores = clf_pseudoLDA(epochs_tr, y_tr, epochs_te)
    X_tr = epochs_to_windowed_features(epochs_tr);
    X_te = epochs_to_windowed_features(epochs_te);
    mdl = fitcdiscr(X_tr, y_tr, 'DiscrimType', 'pseudoLinear', 'Prior', 'uniform');
    [~, s] = predict(mdl, X_te);
    scores = s(:, 2);
end

% --- Classifier 2: shrinkage LDA on windowed features ---
function scores = clf_shrinkLDA(epochs_tr, y_tr, epochs_te)
    X_tr = epochs_to_windowed_features(epochs_tr);
    X_te = epochs_to_windowed_features(epochs_te);
    % 'linear' with Gamma optimized via cross-validation = shrinkage
    mdl = fitcdiscr(X_tr, y_tr, 'DiscrimType', 'linear', ...
        'Prior', 'uniform', 'Gamma', 0.3);   % moderate shrinkage
    [~, s] = predict(mdl, X_te);
    scores = s(:, 2);
end

% --- Classifier 3: xDAWN + LDA ---
function scores = clf_xdawn(epochs_tr, y_tr, epochs_te)
    % xDAWN: find spatial filters that maximize target ERP energy
    n_filters = 4;   % typical choice for P300

    % Average target ERP
    target_avg = mean(epochs_tr(:,:, y_tr == 1), 3);   % [T x C]

    % Stack all training epochs: [T*N_trials x C]
    [T, C, N] = size(epochs_tr);
    X_stack = reshape(permute(epochs_tr, [1 3 2]), [T*N, C]);

    % Build "Toeplitz design" simply: target template repeated where targets occur
    D = zeros(T*N, T);
    for n = 1:N
        if y_tr(n) == 1
            rows = (n-1)*T + (1:T);
            D(rows, :) = eye(T);
        end
    end

    % xDAWN spatial filter via generalized eigendecomposition of
    % (D'*D) and X'*X relations. Simple closed form:
    % Project target template onto data, then SVD.
    A = pinv(D' * D) * D' * X_stack;   % [T x C], best target-ERP estimate
    [U, S, V] = svd(A, 'econ');
    W = V(:, 1:min(n_filters, size(V,2)));   % [C x n_filters] spatial filters

    % Project epochs through filters: [T x n_filters x N]
    epochs_tr_proj = zeros(T, n_filters, N);
    for n = 1:N
        epochs_tr_proj(:,:,n) = epochs_tr(:,:,n) * W;
    end
    epochs_te_proj = zeros(T, n_filters, size(epochs_te, 3));
    for n = 1:size(epochs_te, 3)
        epochs_te_proj(:,:,n) = epochs_te(:,:,n) * W;
    end

    % Now extract windowed features on the projected data
    X_tr = epochs_to_windowed_features(epochs_tr_proj);
    X_te = epochs_to_windowed_features(epochs_te_proj);

    mdl = fitcdiscr(X_tr, y_tr, 'DiscrimType', 'pseudoLinear', 'Prior', 'uniform');
    [~, s] = predict(mdl, X_te);
    scores = s(:, 2);
end

% --- Classifier 4: Riemannian MDM (covariance-based) ---
function scores = clf_mdm(epochs_tr, y_tr, epochs_te)
    % Compute per-trial covariance matrices, then classify via distance
    % to class-mean covariance on the manifold.
    [~, C, N_tr] = size(epochs_tr);
    N_te = size(epochs_te, 3);

    % Per-trial covariance matrices
    cov_tr = zeros(C, C, N_tr);
    for n = 1:N_tr
        x = epochs_tr(:,:,n);
        cov_tr(:,:,n) = (x' * x) / (size(x,1) - 1);
        % Regularize to avoid singular matrices
        cov_tr(:,:,n) = cov_tr(:,:,n) + 1e-6 * eye(C);
    end
    cov_te = zeros(C, C, N_te);
    for n = 1:N_te
        x = epochs_te(:,:,n);
        cov_te(:,:,n) = (x' * x) / (size(x,1) - 1);
        cov_te(:,:,n) = cov_te(:,:,n) + 1e-6 * eye(C);
    end

    % Class mean covariances (use simple Euclidean mean — log-Euclidean
    % is more principled but requires more code; this approximation is
    % close enough for a comparison)
    mean_target    = mean(cov_tr(:,:, y_tr == 1), 3);
    mean_nontarget = mean(cov_tr(:,:, y_tr == 0), 3);

    % Score = log-Euclidean distance to non-target minus distance to target
    % (higher score = more target-like)
    scores = zeros(N_te, 1);
    log_mean_t  = real(logm(mean_target));
    log_mean_nt = real(logm(mean_nontarget));
    for n = 1:N_te
        log_cov = real(logm(cov_te(:,:,n)));
        d_t  = norm(log_cov - log_mean_t,  'fro');
        d_nt = norm(log_cov - log_mean_nt, 'fro');
        scores(n) = d_nt - d_t;   % positive = closer to target
    end
end

% --- Helper: epochs tensor → windowed features ---
function X = epochs_to_windowed_features(epochs)
    % epochs: [T x C x N]
    [T, C, N] = size(epochs);
    n_windows = 10;
    win_len = floor(T / n_windows);
    X = zeros(N, n_windows * C);
    for w = 1:n_windows
        idx_start = (w-1)*win_len + 1;
        idx_end   = w * win_len;
        win_mean  = squeeze(mean(epochs(idx_start:idx_end, :, :), 1))';
        X(:, (w-1)*C+1 : w*C) = win_mean;
    end
end

%% =========================================================
% SECTION 3: RUN ALL CLASSIFIERS ON ALL FILES
% =========================================================

fprintf('=== Comparing 4 classifiers across all files ===\n');

clf_names = {'pseudoLDA', 'shrinkLDA', 'xDAWN+LDA', 'Riemann-MDM'};
clf_fns   = {@clf_pseudoLDA, @clf_shrinkLDA, @clf_xdawn, @clf_mdm};
n_clf = length(clf_names);

% Results table: [n_files x n_classifiers] of group-of-8 accuracy
acc_matrix = nan(n_files, n_clf);
file_labels = {};

for f = 1:n_files
    r = results(f);
    if isempty(r.epochs)
        file_labels{f} = '';
        continue;
    end
    file_labels{f} = r.fname;

    fprintf('  %s\n', r.fname);
    for c = 1:n_clf
        fprintf('    %-15s ... ', clf_names{c});
        try
            acc = evaluate_group_of_8(r.epochs, r.labels, r.fname, cfg, clf_fns{c});
            acc_matrix(f, c) = acc;
            fprintf('%.1f%%\n', 100*acc);
        catch ME
            fprintf('FAILED (%s)\n', ME.message);
        end
    end
end

%% =========================================================
% SECTION 4: COMPARISON FIGURE
% =========================================================

fprintf('\n=== Building comparison figure ===\n');

valid_files = find(~cellfun(@isempty, file_labels));
acc_plot = 100 * acc_matrix(valid_files, :);
labels_plot = file_labels(valid_files);

fig = figure('Position', [50 50 1300 600], 'Color', 'w');
b = bar(acc_plot, 'grouped');

% Color the classifier groups
clf_colors = [
    0.30 0.30 0.30;   % pseudoLDA - dark grey (baseline)
    0.20 0.50 0.80;   % shrinkLDA - blue
    0.85 0.40 0.30;   % xDAWN+LDA - orange
    0.30 0.65 0.40    % Riemann-MDM - green
];
for c = 1:n_clf
    b(c).FaceColor = clf_colors(c, :);
    b(c).EdgeColor = 'none';
end

yline(12.5, 'k--', 'Chance (12.5%)', 'LineWidth', 1.2);
yline(23,   'b:',  '95% sig (23%)', 'LineWidth', 1.2);

set(gca, 'XTick', 1:length(labels_plot), 'XTickLabel', labels_plot, ...
    'TickLabelInterpreter', 'none');
xtickangle(35);
ylabel('Group-of-8 accuracy (%)');
ylim([0 100]);
title('Classifier comparison: 4 methods, same data, same task');
legend(clf_names, 'Location', 'eastoutside');
grid on;

exportgraphics(fig, 'classifier_comparison.pdf', 'ContentType','vector');
fprintf('  Saved classifier_comparison.pdf\n');

save('classifier_comparison.mat', 'acc_matrix', 'clf_names', 'file_labels');
fprintf('  Saved classifier_comparison.mat\n\n');

%% =========================================================
% SECTION 5: PRINT SUMMARY TABLE
% =========================================================

fprintf('=== CLASSIFIER COMPARISON TABLE (group-of-8 accuracy %%) ===\n');
fprintf('%-16s', 'File');
for c = 1:n_clf, fprintf(' %-12s', clf_names{c}); end
fprintf('\n%s\n', repmat('-', 1, 16 + 13*n_clf));

for f = 1:n_files
    if isempty(file_labels{f}) || strcmp(file_labels{f}, ''), continue; end
    fprintf('%-16s', file_labels{f});
    for c = 1:n_clf
        if isnan(acc_matrix(f, c))
            fprintf(' %-12s', 'NaN');
        else
            sig = ' '; if acc_matrix(f, c) > 0.23, sig = '*'; end
            fprintf(' %-12s', sprintf('%.1f%%%s', 100*acc_matrix(f,c), sig));
        end
    end
    fprintf('\n');
end

% Mean per classifier
fprintf('%-16s', 'MEAN');
for c = 1:n_clf
    m = mean(acc_matrix(:, c), 'omitnan');
    fprintf(' %-12s', sprintf('%.1f%%', 100*m));
end
fprintf('\n\n');

%% =========================================================
% SECTION 6: SANITY CHECK — compare pseudoLDA here vs. earlier run
% =========================================================

if exist('classification_results.mat', 'file')
    earlier = load('classification_results.mat', 'group_results');
    fprintf('=== Diagnostic: pseudoLDA accuracy, this run vs. earlier ===\n');
    fprintf('%-16s  %-20s %-20s\n', 'File', 'earlier-pseudoLDA', 'comparison-pseudoLDA');
    fprintf('%s\n', repmat('-', 1, 60));
    for f = 1:length(earlier.group_results)
        gr = earlier.group_results(f);
        if isempty(gr) || ~isfield(gr,'group_acc') || isempty(gr.group_acc), continue; end
        if isnan(acc_matrix(f, 1)), continue; end
        fprintf('%-16s  %.1f%%               %.1f%%\n', ...
            gr.fname, 100*gr.group_acc, 100*acc_matrix(f, 1));
    end
    fprintf('\n');
end