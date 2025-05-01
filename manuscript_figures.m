clear all; close all; clc;
base_folder = 'C:/Users/liu_s/policycompression_neuraldim';
cd(base_folder)
addpath(genpath(base_folder))


% Figure and font default setting
set(0,'units','inches');
Inch_SS = get(0,'screensize');
set(0,'units','pixels');
figsize = get(0, 'ScreenSize');
Res = figsize(3)./Inch_SS(3);
set(groot, 'DefaultAxesTickDir', 'out', 'DefaultAxesTickDirMode', 'manual');
fontsize=12;
set(groot,'DefaultAxesFontName','Arial','DefaultAxesFontSize',fontsize);
set(groot,'DefaultLegendFontSize',fontsize-2,'DefaultLegendFontSizeMode','manual')

alpha = 0.5; 
markersize = 10; linewidth=1.5;

% paths
figformat = "svg";
figpath = "figs\"; %"newplots\"
png_dpi = 500;

% Color palettes
cmap = brewermap(3, 'Set1');
cmap = cmap([1,3,2],:);
cmap_subj = brewermap(200, 'Set1');
cmap_exp3 = brewermap(10, 'Set2');
cmap_exp3 = cmap_exp3([2,1,3,4,5,6,9,7,8,10],:);

%% Load behavioral data
% Load policy cost information, computed by block.
T = readtable("BEH\"+"T_policycost_Paforeachblock.txt");
% Keep only trials that have been responded / responsive trials
T = T(T.OMIT ~= 1, :);
T.cond_surprisal = T.surprisal-T.cost;
T.RT_minus_SOA = T.RT - T.SOA;
T.cost_abs = abs(T.cost);
T.logRT = log(T.RT);


%% Load neural dimensionality files
table_path = "EEG_paper_analysis\new_dimensionality_files\";
subjids = [304:307,309:345];
for subjid =subjids
    subjid
    dimfile_savename = "A"+subjid+"_DIM_READOUT_RL_processed.txt";
    if(subjid==subjids(1))
        T_dim = readtable(table_path+dimfile_savename);
    else
        T_dim = [T_dim; readtable(table_path+dimfile_savename)];
    end
end
% Remove dimensionality estimates after action onset
T_dim = T_dim(logical((T_dim.time<0)),:);
%T_dim2 = T_dim(logical((T_dim.time<600)),:);

% For each trial, either take the mean or max dimensionality over timesteps.
dimdata_bytrial= groupsummary(T_dim, {'SUBID', 'BLOCK','TRIAL'}, {'mean','max'}, {'dimensionality_estimate'});



%% Behavioral analysis: 
n_states=12;
n_actions=4;
p_state = ones(1,n_states)./n_states;
Q = zeros(n_states,n_actions);
for action=1:n_actions
    Q(((action-1)*3+1):(action*3),action) = 1;
end
beta_set = linspace(0,10,50);
[R1, V1, Pa1, optimal_policy1] = blahut_arimoto(p_state,Q,beta_set);

% Count the number of trials by SUBID and SOACODE (equivalent to R's count function)
ntrials_bysubjcond = groupsummary(T, {'SUBID', 'SOACODE'}, 'IncludeEmptyGroups', true, 'IncludeMissingGroups', true);
% Rename the 'GroupCount' column to 'n_trials'
ntrials_bysubjcond.Properties.VariableNames{'GroupCount'} = 'n_trials';
% Calculate accuracy (mean of ACC) grouped by SUBID and SOACODE
acc_bysubjcond = varfun(@mean, T, 'InputVariables', 'ACCRESP', ...
                        'GroupingVariables', {'SUBID', 'SOACODE'}, ...
                        'OutputFormat', 'table');
acc_bysubjcond.Properties.VariableNames{'mean_ACCRESP'} = 'accuracy';
% Calculate reaction time (mean of RT) grouped by SUBID and SOACODE
rt_bysubjcond = varfun(@mean, T, 'InputVariables', 'RT', ...
                       'GroupingVariables', {'SUBID', 'SOACODE'}, ...
                       'OutputFormat', 'table');
rt_bysubjcond.Properties.VariableNames{'mean_RT'} = 'rt';
% Merge ntrials_bysubcond and acc_bysubjcond by SUBID and SOACODE
stats_bysubjcond = outerjoin(ntrials_bysubjcond, acc_bysubjcond, 'Keys', {'SUBID', 'SOACODE'}, ...
                            'MergeKeys', true, 'Type', 'Left');
% Merge the resulting table with rt_bysubjcond
stats_bysubjcond = outerjoin(stats_bysubjcond, rt_bysubjcond, 'Keys', {'SUBID', 'SOACODE'}, ...
                            'MergeKeys', true, 'Type', 'Left');
% Remove any extra GroupCount columns
colsToRemove = startsWith(stats_bysubjcond.Properties.VariableNames, 'GroupCount');
stats_bysubjcond(:, colsToRemove) = [];

ntrials_bytaskstim = groupsummary(T, {'TASK', 'STIMPOS'}, 'IncludeEmptyGroups', true, 'IncludeMissingGroups', true);
ntrials_bytaskstim.Properties.VariableNames{'GroupCount'} = 'n_trials';

% Additionally stratify by subjects and condition
ntrials_bysubjcondtaskstim = groupsummary(T, {'SUBID', 'SOACODE','TASK', 'STIMPOS'}, 'IncludeEmptyGroups', true, 'IncludeMissingGroups', true);
ntrials_bysubjcondtaskstim.Properties.VariableNames{'GroupCount'} = 'n_trials';

% Mutual information
subj_idxs = unique(T.SUBID,'stable');
n_subj = length(subj_idxs);
n_conds = 3; % SOACODE column: Short, Medium, or Long SOA. 
n_stims = 4; % 4 possible stimuli locations.

complexity = zeros(n_subj*n_conds,1);
cond_entropy = zeros(n_subj*n_conds,1);
repeat_actions = zeros(n_subj*n_conds,1);
P_a_given_s_bycond = zeros(n_subj,n_conds, n_states,n_actions);
P_a_bycond = zeros(n_subj,n_conds, n_actions);
idx=0;
for subj=1:n_subj
    for c=1:n_conds
        T_subjcond = T(T.SOACODE == c & T.SUBID==subj_idxs(subj), :);
        state = (T_subjcond.TASK-1).*n_stims + T_subjcond.STIMPOS; % This treats each rule + stimulus as one state. Hence 12 distinct states.
        action = T_subjcond.RESP;
        idx=idx+1;
        complexity(idx)=mutual_information(round(state),round(action),0.1)./log(2);
        cond_entropy(idx) = condEntropy(round(action), round(state));
        repeat_actions(idx) = mean(action(2:end)==action(1:(end-1)));

        P_a_bycond(subj,c,:) = histcounts(action, 0.5:1:(n_actions+0.5));
        P_a_bycond(subj,c,:) = P_a_bycond(subj,c,:)./sum(P_a_bycond(subj,c,:));
        for s=1:n_states
            state_relevant_idxs = (state==s);
            P_a_given_s_bycond(subj,c,s,:) = histcounts(action(state_relevant_idxs), 0.5:1:(n_actions+0.5));
            P_a_given_s_bycond(subj,c,s,:) = P_a_given_s_bycond(subj,c,s,:)./sum(P_a_given_s_bycond(subj,c,s,:));
        end
    end
end
stats_bysubjcond.complexity = complexity;
stats_bysubjcond.cond_entropy = cond_entropy;
stats_bysubjcond.repeat_actions = repeat_actions;


% Between-condition t-tests
conds = 1:n_conds;
complexity_bycond = reshape(stats_bysubjcond.complexity,n_conds,n_subj)';
rt_bycond = reshape(stats_bysubjcond.rt,n_conds,n_subj)'./1000;
accuracy_bycond = reshape(stats_bysubjcond.accuracy,n_conds,n_subj)';
cond_entropy_bycond = reshape(stats_bysubjcond.cond_entropy,n_conds,n_subj)';
repeat_actions_bycond = reshape(stats_bysubjcond.repeat_actions,n_conds,n_subj)';


%% RT LMEs
T_rt = table(repmat(subj_idxs,n_conds,1), complexity_bycond(:), rt_bycond(:), VariableNames={'SUBID','Complexity','RT'});
lme0_rt_complexity = fitlme(T_rt, "RT~1+(1|SUBID)")
lme_rt_complexity = fitlme(T_rt, "RT~Complexity+(Complexity|SUBID)")
compare(lme0_rt_complexity,lme_rt_complexity)
model1 = lme_rt_complexity; model2 = lme0_rt_complexity;
deltaAIC = model1.ModelCriterion.AIC - model2.ModelCriterion.AIC;
deltaBIC = model1.ModelCriterion.BIC - model2.ModelCriterion.BIC;
disp(['ΔAIC: ', num2str(deltaAIC), '; ΔBIC: ', num2str(deltaBIC)]);



%% Figure 3
alpha = 0.5; line_alpha = 0.05; ttl_fontsize = 12; ttl_position_xshift = -0.19; ttl_position_yshift = 0.99;
figure("Position", [10,10,1000,600])
tiledlayout(2,3, 'Padding', 'compact', 'TileSpacing', 'compact'); 

% Rate distortion curve
nexttile; hold on;
plot(R1,V1,"k-","HandleVisibility","off")
for cond=1:n_conds
    scatter(complexity_bycond(:,cond),accuracy_bycond(:,cond),10,'MarkerFaceColor',cmap(cond,:),'MarkerEdgeColor',cmap(cond,:),'MarkerFaceAlpha',alpha,'MarkerEdgeAlpha',alpha)
end
ylim([0.2,1])
xlim([0,log2(4)])
xlabel("Policy complexity (bits)")
ylabel("Trial-averaged reward")
lgd = legend("Short","Medium","Long","location","southeast");
title(lgd, {"SOA condition"})
ttl = title("A", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')



% Policy complexity, by RT deadline cond
[a,b,c,d]=ttest(complexity_bycond(:,1), complexity_bycond(:,3))
CohensD_complexity_withinsubj = table2cell(meanEffectSize(complexity_bycond(:,1), complexity_bycond(:,3), Effect="cohen", Paired=true));
nexttile; hold on;
[se,m] = wse(complexity_bycond,2);
errorbar(conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
ylim([0.5,2])
xlim([min(conds)-0.5,max(conds)+0.5])
xticks(conds)
xticklabels(["Short","Medium","Long"])
xlabel("SOA condition")
ylabel("Policy complexity (bits)")
ttl = title("B", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')


[a,b,c,d]=ttest(rt_bycond(:,1), rt_bycond(:,3))
nexttile; hold on;
[se,m] = wse(rt_bycond,2);
errorbar(conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
ylim([0.4,0.8])
% yticks(0.2:0.1:0.5)
xlim([min(conds)-0.5,max(conds)+0.5])
xticks(conds)
xticklabels(["Short","Medium","Long"])
xlabel("SOA condition")
ylabel("Response time (sec)")
yticks(0.4:0.1:0.8)
ttl = title("C", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')


% Accuracy, by RT deadline cond
[a,b,c,d]=ttest(accuracy_bycond(:,1), accuracy_bycond(:,3))
nexttile; hold on;
[se,m] = wse(accuracy_bycond,2);
errorbar(conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
yticks(0.7:0.1:1)
xlim([min(conds)-0.5,max(conds)+0.5])
xticks(conds)
xticklabels(["Short","Medium","Long"])
xlabel("SOA condition")
ylabel("Accuracy")
ttl = title("D", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')


% H(A|S), by RT deadline cond
[a,b,c,d]=ttest(cond_entropy_bycond(:,1), cond_entropy_bycond(:,3))
nexttile; hold on;
[se,m] = wse(cond_entropy_bycond,2);
errorbar(conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
ylim([0,1.2])
% yticks(0.2:0.1:0.5)
xlim([min(conds)-0.5,max(conds)+0.5])
xticks(conds)
xticklabels(["Short","Medium","Long"])
xlabel("SOA condition")
ylabel("H(A|S) (bits)")
ttl = title("E", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')

% RT LME partial residual plots
nexttile(6); hold on;
[partial_residual_plot] = partial_residual_plots(lme_rt_complexity, T_rt,"Complexity", 0:0.2:1);
errorbar(partial_residual_plot.bin_centers,partial_residual_plot.mu_ys,partial_residual_plot.sem_ys,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
xlabel('Policy complexity (bits)');
ylabel('Partial Residuals');
xlim([0.5,2])
yticks(0.2:0.1:0.5)
ttl = title("F", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')

saveas(gca, figpath+'Fig3.fig')
exportgraphics(gcf,figpath+'Fig3.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'Fig3.pdf',"ContentType","vector");


%% Perseveration analysis: only analyze trials that i) is not the first responded trial in a block AND ii) its previous trial has been responded. Perseveration based on past subject choice.
% Get all trials that were responded (RT<Inf). 
T_resp = readtable('HAL2017_READOUT_TEST_BehP.txt');
T_resp = T_resp(:, {'SUBID', 'BLOCK', 'TRIAL', 'CUETYPE', 'STIMPOS', 'TASK', ...
            'CORRESP', 'RESP', 'RT', 'ACCRESP', 'SOA', 'SOACODE', 'VALID','OMIT','PRELATE'});

% Only consider trials where the previous trial is responsive.
T_resp = T_resp(T_resp.OMIT ~= 1, :);

% There are 23 trials where the response is larger than 4. Not sure why---will remove these trials.
T_resp = T_resp(T_resp.RESP<=4,:);

% Correct mappings. Rows are different tasks/rules (1 to 3); Cols are the correct actions
% for each stim (cols).
n_rules = 3;
correct_actions = [4,3,2,1; 2,1,4,3; 3,4,1,2];
repeat_rule = zeros(n_subj*n_conds,1);
repeat_rule_byrule = zeros(n_subj,n_conds,n_rules);
repeat_stim = zeros(n_subj*n_conds,1);
repeat_actions = zeros(n_subj*n_conds,1);
repeat_actions_byaction = zeros(n_subj,n_conds,n_actions);
repeat_rule_aggcond = zeros(n_subj,1);
repeat_stim_aggcond = zeros(n_subj,1);
repeat_actions_aggcond = zeros(n_subj,1);
num_trials = zeros(n_subj,1);
num_trials_analyzed = zeros(n_subj,1);
num_trials_chooseimpossibleaction = zeros(n_subj,1);
idx=0;
for subj=1:n_subj
    T_subj = T_resp(T_resp.SUBID==subj_idxs(subj), :);
    state = (T_subj.TASK-1).*n_conds + T_subj.STIMPOS; % This treats each rule + stimulus as one state. Hence 12 distinct states.
    action = T_subj.RESP;
    num_trials(subj) = length(action);
    correct_action_assuming_prev_trial_task = zeros(height(T_subj)-1,1);
    correct_action_assuming_prev_trial_stim = zeros(height(T_subj)-1,1);
    prev_respondedtrial_sameblock = ones(height(T_subj)-1,1);
    prevtrial_consistentwithsomerule = ones(height(T_subj)-1,1);
    trial_isvalid = T_subj.VALID(2:end);
    trial_isnotlate = T_subj.OMIT ~= 1 & (T_subj.VALID==1 | (T_subj.RT<=T_subj.SOA));
    trial_isnotlate = trial_isnotlate(2:end);

    prevtrials_appliedrule = zeros(height(T_subj)-1,1);
    for trial=2:height(T_subj)
        prev_trial_block = T_subj.BLOCK(trial-1);
        current_trial_block = T_subj.BLOCK(trial);
        prev_trial_blocktrial = T_subj.TRIAL(trial-1);
        current_trial_blocktrial = T_subj.TRIAL(trial);
        if(current_trial_block~=prev_trial_block)
            prev_respondedtrial_sameblock(trial-1) = 0;
        elseif(current_trial_blocktrial~=(prev_trial_blocktrial+1))
            prev_respondedtrial_sameblock(trial-1) = 0;
        end

        prev_trial_stim = T_subj.STIMPOS(trial-1);
        prev_trial_action = action(trial-1);
        current_task = T_subj.TASK(trial);
        current_stim = T_subj.STIMPOS(trial);

        % Infer which rule the agent used in the previous trial
        prevtrial_corraction_underrules = correct_actions(:,prev_trial_stim);
        prevtrial_appliedrule = find(prevtrial_corraction_underrules == prev_trial_action);
        if(isempty(prevtrial_appliedrule)) % Subject clicked the same location as the stim, which is incorrect under all trials
            % Do not analyze this trial.
            prevtrial_appliedrule = NaN;
            prevtrial_consistentwithsomerule(trial-1) = 0;
            % Count the total number of such trials per subject.
            num_trials_chooseimpossibleaction(subj) = num_trials_chooseimpossibleaction(subj)+1;
        else
            % Assess rule-perseverance w.r.t. to the inferred rule used in
            % the previous trial
            correct_action_assuming_prev_trial_task(trial-1) = correct_actions(prevtrial_appliedrule, current_stim);
        end
        prevtrials_appliedrule(trial-1) =  prevtrial_appliedrule;

        % This line remains unchanged...
        correct_action_assuming_prev_trial_stim(trial-1) = correct_actions(current_task, prev_trial_stim);
    end
    % Only analyze trials who are i) responsive; ii) not the first trial in a new block; AND iii) previous responsive trial is in the same block;
    thistrial_inclusion_criterion = prev_respondedtrial_sameblock;
    analyzed_trials = find(thistrial_inclusion_criterion==1) + 1; % +1 to reflect the fact that [prev_respondedtrial_sameblock,trial_isvalid] both start from Trial 2 that the subject did.
    num_trials_analyzed(subj) = length(analyzed_trials); % Count how many trials were used for perseveration analysis. 
    repeat_prevtrial_action_subj = (action(analyzed_trials)==action(analyzed_trials-1)); % First idx is Trial 2 of the subject.
    repeat_rule_alltrials_subj = (action(analyzed_trials)==correct_action_assuming_prev_trial_task(analyzed_trials-1)); 
    repeat_stim_alltrials_subj = (action(analyzed_trials)==correct_action_assuming_prev_trial_stim(analyzed_trials-1));
    repeat_actions_aggcond(subj) = mean(repeat_prevtrial_action_subj);
    repeat_rule_aggcond(subj) = mean(repeat_rule_alltrials_subj);
    repeat_stim_aggcond(subj) = mean(repeat_stim_alltrials_subj);

    for c=1:n_conds
        idx=idx+1;
        relevant_trials_subjcond_raw = (T_subj.SOACODE==c);
        relevant_trials_subjcond_raw = relevant_trials_subjcond_raw(2:end); % Cannot analyze Trial 1 of the subject.
        relevant_trials_subjcond = relevant_trials_subjcond_raw .* thistrial_inclusion_criterion;
        relevant_trials_subjcond = find(relevant_trials_subjcond==1) + 1; % The resultant vector's Entry 1 is Trial 1 of the subject.

        repeat_actions(idx) = mean(action(relevant_trials_subjcond)==action(relevant_trials_subjcond-1)); 
        repeat_rule(idx) = mean(action(relevant_trials_subjcond)==correct_action_assuming_prev_trial_task(relevant_trials_subjcond-1)); 
        repeat_stim(idx) = mean(action(relevant_trials_subjcond)==correct_action_assuming_prev_trial_stim(relevant_trials_subjcond-1));
        
        % Compute rule and action perseveration, but stratify by which
        % rule/action is being used on the previous trial. 
        for rule_idx=1:n_rules
            relevant_trials_subjcondrule = relevant_trials_subjcond_raw .* thistrial_inclusion_criterion .* (prevtrials_appliedrule==rule_idx);
            relevant_trials_subjcondrule = find(relevant_trials_subjcondrule==1)+1;  % The resultant vector's Entry 1 is Trial 1 of the subject.
            repeat_rule_byrule(subj,c,rule_idx) =  mean(action(relevant_trials_subjcondrule)==correct_action_assuming_prev_trial_task(relevant_trials_subjcondrule-1)); 
        end
        for a=1:n_actions
            relevant_trials_subjcondaction = relevant_trials_subjcond_raw .* thistrial_inclusion_criterion .* (action(1:end-1)==a);
            relevant_trials_subjcondaction = find(relevant_trials_subjcondaction==1)+1;  % The resultant vector's Entry 1 is Trial 1 of the subject.
            repeat_actions_byaction(subj,c,a) =  mean(action(relevant_trials_subjcondaction)==action(relevant_trials_subjcondaction-1)); 
        end
    end
end
stats_bysubjcond.repeat_actions = repeat_actions;
stats_bysubjcond.repeat_rule = repeat_rule;
stats_bysubjcond.repeat_stim = repeat_stim;
repeat_actions_bycond = reshape(stats_bysubjcond.repeat_actions,n_conds,n_subj)';
repeat_rule_bycond = reshape(stats_bysubjcond.repeat_rule,n_conds,n_subj)';
repeat_stim_bycond = reshape(stats_bysubjcond.repeat_stim,n_conds,n_subj)';

%% Figure 4
alpha = 0.5; line_alpha = 0.05; ttl_fontsize = 12; ttl_position_xshift = -0.25; ttl_position_yshift = 0.99;
figure("Position", [0,0,1180,800])
tiledlayout(2,3, 'Padding', 'compact', 'TileSpacing', 'compact'); 
nexttile; hold on;
[a,b,c,d]=ttest(repeat_actions_bycond(:,1), repeat_actions_bycond(:,3))
[se,m] = wse(repeat_actions_bycond,2);
errorbar(conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
%plot([min(conds)-2,max(conds)+2], [1/4,1/4],"k--")
xlim([min(conds)-0.5,max(conds)+0.5])
xticks(conds)
xticklabels(["Short","Medium","Long"])
xlabel("SOA condition")
ylabel("P(repeat prev. trial's action)")
plot([-0.5,4],[1/n_actions, 1/n_actions],"k--","HandleVisibility","off")
ylim([0.235,0.255])
yticks([0.235:0.005:0.255])
% title("Two-sided paired t-test: p="+round(b,5))
ttl = title("A", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')

nexttile; hold on;
[a,b,c,d]=ttest(repeat_rule_bycond(:,1), repeat_rule_bycond(:,3))
[se,m] = wse(repeat_rule_bycond,2);
errorbar(conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
%plot([min(conds)-2,max(conds)+2], [1/4,1/4],"k--")
xlim([min(conds)-0.5,max(conds)+0.5])
xticks(conds)
xticklabels(["Short","Medium","Long"])
xlabel("SOA condition")
ylabel({"P(use prev. applied rule","on current stim)"})
plot([-0.5,4],[1/n_rules, 1/n_rules],"k--","HandleVisibility","off")
% title("Two-sided paired t-test: p="+round(b,5))
ttl = title("B", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')

nexttile; hold on;
[se,m] = wse(repeat_stim_bycond,2);
[a,b,c,d]=ttest(repeat_stim_bycond(:,1), repeat_stim_bycond(:,3))
errorbar(conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
%plot([min(conds)-2,max(conds)+2], [1/4,1/4],"k--")
xlim([min(conds)-0.5,max(conds)+0.5])
xticks(conds)
xticklabels(["Short","Medium","Long"])
xlabel("SOA condition")
ylabel({"P(choose action correct for", "prev. stim under current rule)"})
ylim([0.245,0.265])
yticks([0.245:0.005:0.265])
plot([-0.5,4],[1/n_actions, 1/n_actions],"k--","HandleVisibility","off")
% title("Two-sided paired t-test: p="+round(b,5))
ttl = title("C", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')

nexttile; hold on;
for c=1:n_conds
    [se,m] = wse(squeeze(P_a_bycond(:,c,:)),2);
    errorbar(1:n_actions,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color',cmap(c,:))
end
plot([0,5], [0.25,0.25],"k--","HandleVisibility","off")
xlim([0.5,n_actions+0.5])
ylim([0.22,0.28])
yticks(0.22:0.02:0.28)
ylabel("P(a)")
xlabel("Actions")
xticklabels(["A_1","A_2","A_3","A_4"])
lgd = legend("Short","Medium","Long");
title(lgd, {"SOA condition"})
%sgtitle({"Assuming each task-stimulus combination is a unique state",trial_inclusion_rule})
[a,b,c,d]=ttest(mean(P_a_bycond(:,1,1:2),3), mean(P_a_bycond(:,1,3:4),3))
ttl = title("D", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')

% Plot the same as Row 1, but stratified by the specific action/rule being repeated.
nexttile; hold on;
pvals = zeros(n_actions,1);
for action_idx=1:n_actions
    [a,b,c,d]=ttest(repeat_actions_byaction(:,1,action_idx), repeat_actions_byaction(:,3,action_idx))
    pvals(action_idx) = b;
    [se,m] = wse(squeeze(repeat_actions_byaction(:,:,action_idx)),2);
    errorbar(conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color',cmap_exp3(action_idx+3,:))
end
%plot([min(conds)-2,max(conds)+2], [1/4,1/4],"k--")
xlim([min(conds)-0.5,max(conds)+0.5])
xticks(conds)
ylim([0.2,0.325])
yticks(0.2:0.025:0.325)
xticklabels(["Short","Medium","Long"])
xlabel("SOA condition")
ylabel("P(repeat prev. trial's action)")
lgd = legend("A_1","A_2","A_3","A_4");
title(lgd, {"Prev. trial's", "chosen action"})
lgd.Color = [1 1 1];  % Set the color to white
lgd.Box = 'on';  % Ensure the box is on
lgd.BoxFace.ColorType = 'truecoloralpha';  % Enable transparency
lgd.BoxFace.ColorData = uint8([255; 255; 255; 127]);  % [R; G; B; Alpha] with 127/255 transparency
plot([-0.5,4],[1/n_actions, 1/n_actions],"k--","HandleVisibility","off")
%title({"Two-sided paired t-test:","p=["+strjoin(string(round(pvals,5)),',')+"]"})
ttl = title("E", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')

nexttile; hold on;
pvals = zeros(n_rules,1);
for rule_idx=1:n_rules
    [a,b,c,d]=ttest(repeat_rule_byrule(:,1,rule_idx), repeat_rule_byrule(:,3,rule_idx))
    pvals(rule_idx) = b;
    [se,m] = wse(squeeze(repeat_rule_byrule(:,:,rule_idx)),2);
    errorbar(conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color',cmap_exp3(rule_idx,:))
end
xlim([min(conds)-0.5,max(conds)+0.5])
xticks(conds)
xticklabels(["Short","Medium","Long"])
xlabel("SOA condition")
ylabel({"P(use prev. applied rule","on current stim)"})
plot([-0.5,4],[1/n_rules, 1/n_rules],"k--","HandleVisibility","off")
lgd = legend("Vertical","Horizontal","Diagonal");
title(lgd, {"Prev. trial's", "applied rule"})
lgd.Color = [1 1 1];  % Set the color to white
lgd.Box = 'on';  % Ensure the box is on
lgd.BoxFace.ColorType = 'truecoloralpha';  % Enable transparency
lgd.BoxFace.ColorData = uint8([255; 255; 255; 127]);  % [R; G; B; Alpha] with 127/255 transparency
ttl = title("F", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')

saveas(gca, figpath+'Fig4.fig')
exportgraphics(gcf,figpath+'Fig4.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'Fig4.pdf',"ContentType","vector");










%% Below is all neural-level code. Join the behavioral and dimensionality data
Dim_Behavior = innerjoin(T, dimdata_bytrial);



%% Trial-by-trial policy cost measures
lme0 = fitlme(Dim_Behavior,'mean_dimensionality_estimate~1+(1|SUBID)')
lme1 = fitlme(Dim_Behavior,'mean_dimensionality_estimate~cost+(cost|SUBID)')
lme_full = fitlme(Dim_Behavior,'mean_dimensionality_estimate~cost+BLOCK+TRIAL+SOACODE+(cost+BLOCK+TRIAL+SOACODE|SUBID)')
lme_full0 = fitlme(Dim_Behavior,'mean_dimensionality_estimate~BLOCK+TRIAL+SOACODE+(BLOCK+TRIAL+SOACODE|SUBID)')

%% Model comparison
model1 = lme1; model2 = lme_full0;
deltaAIC = model1.ModelCriterion.AIC - model2.ModelCriterion.AIC;
deltaBIC = model1.ModelCriterion.BIC - model2.ModelCriterion.BIC;
disp(['ΔAIC: ', num2str(deltaAIC), '; ΔBIC: ', num2str(deltaBIC)]);


%% Below is LME, aggregated over trials:

% Mutual information computed in the usual way
% Take the mean over (mean/max dimensionality over time) over trials. 
Dim_Behavior_bycond = groupsummary(Dim_Behavior, {'SUBID', 'SOACODE'}, {'mean'}, {'mean_dimensionality_estimate','max_dimensionality_estimate','cost'});
Dim_Behavior_bycond = renamevars(Dim_Behavior_bycond,["mean_cost"],["Complexity"]);
Dim_Behavior_bycond.Complexity = complexity;
lme0 = fitlme(Dim_Behavior_bycond,'mean_max_dimensionality_estimate~1+(1|SUBID)')
lme = fitlme(Dim_Behavior_bycond,'mean_max_dimensionality_estimate~Complexity+(Complexity|SUBID)')

model1 = lme; model2 = lme0;
deltaAIC = model1.ModelCriterion.AIC - model2.ModelCriterion.AIC;
deltaBIC = model1.ModelCriterion.BIC - model2.ModelCriterion.BIC;
disp(['ΔAIC: ', num2str(deltaAIC), '; ΔBIC: ', num2str(deltaBIC)]);

%% Partial residual plot for Dimensionality vs. Policy cost / I(S;A) LMEs
figure("Position",[10,10,1200,400]);
tiledlayout(1,3, 'Padding', 'compact', 'TileSpacing', 'compact'); 
nexttile(1); hold on;
[partial_residual_plot] = partial_residual_plots(lme1, Dim_Behavior,"cost", 0:0.2:1);
errorbar(partial_residual_plot.bin_centers,partial_residual_plot.mu_ys,partial_residual_plot.sem_ys,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
xlabel('Policy cost (bits)');
ylabel('Partial Residuals');
xlim([-3,3])
ttl = title("A", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift+0.03; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')

nexttile(3); hold on;
[partial_residual_plot] = partial_residual_plots(lme, Dim_Behavior_bycond,"Complexity", 0:0.2:1);
errorbar(partial_residual_plot.bin_centers,partial_residual_plot.mu_ys,partial_residual_plot.sem_ys,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
xlabel('Policy complexity (bits)');
ylabel('Partial Residuals');
xlim([0.5,2])
ttl = title("C", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift+0.03; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')

% Correlate I(S;A) with dimensionality
temp = groupsummary(Dim_Behavior, {'SUBID', 'SOACODE'}, {'mean'}, {'mean_dimensionality_estimate','max_dimensionality_estimate','cost'});
temp.Complexity = complexity;
dimensionality_bycond_max = reshape(Dim_Behavior_bycond.mean_max_dimensionality_estimate,n_conds,n_subj)';
[a,b,c,d]=ttest(dimensionality_bycond_max(:,1), dimensionality_bycond_max(:,3))
[a,b] = corr(temp.mean_cost, temp.mean_max_dimensionality_estimate, type="Pearson")


% Dimensionality, by RT deadline cond
[a,b,c,d]=ttest(dimensionality_bycond_max(:,1), dimensionality_bycond_max(:,3))
CohensD_complexity_withinsubj = table2cell(meanEffectSize(dimensionality_bycond_max(:,1), dimensionality_bycond_max(:,3), Effect="cohen", Paired=true));
nexttile(2); hold on;
[se,m] = wse(dimensionality_bycond_max,2);
errorbar(1:n_conds,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
% ylim([0.5,3.5])
xlim([1-0.5,n_conds+0.5])
xticks(1:n_conds)
xticklabels(["Short","Medium","Long"])
ylim([0.68,0.682])
yticks(0.68:0.0005:0.682)
xlabel("SOA condition")
ylabel("Trial-averaged max. accuracy")
ttl = title("B", "Fontsize", ttl_fontsize);
ttl.Units = 'Normalize'; 
ttl.Position(1) = ttl_position_xshift-0.05; % use negative values (ie, -0.1) to move further left
ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
set(gca,'box','off')

saveas(gca, figpath+'Fig5.fig')
exportgraphics(gcf,figpath+'Fig5.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'Fig5.pdf',"ContentType","vector");


%% Helper functions
function [partial_residual_plot] = partial_residual_plots(lme,X,predictor_name, bin_quantiles)
    % Given a LME object, require the partial residual plot stats for a
    % particular predictor's fixed effects, binned according to quantiles
    % of the predictor's value.

    % Compute partial residuals for x1
    residuals_lme = residuals(lme);
    [beta_x1,betanames] = fixedEffects(lme);
    regression_idx = find(strcmp(predictor_name,betanames.Name));
    beta_x1 = beta_x1(regression_idx);
    partial_residuals = residuals_lme + beta_x1 * X.(predictor_name);

    % Bin into quantiles based on value of predictor.
    bin_edges = quantile(X.(predictor_name), bin_quantiles);%-4:1.6:4; for equal width bins centered at 0
    bin_centers = (bin_edges(2:end) + bin_edges(1:(end-1)))./2;
    bin_idxs = discretize(X.(predictor_name),bin_edges);
    mu_xs = zeros(1,length(bin_edges)-1);
    sem_xs = zeros(1,length(bin_edges)-1);
    mu_ys = zeros(1,length(bin_edges)-1);
    sem_ys = zeros(1,length(bin_edges)-1);
    for bin_idx = 1:(length(bin_edges)-1)
        relevant_trials = find(bin_idxs==bin_idx);
        mu_xs(bin_idx) = mean(X.(predictor_name)(relevant_trials));
        sem_xs(bin_idx) = std(X.(predictor_name)(relevant_trials))./sqrt(length(relevant_trials));
        mu_ys(bin_idx) = mean(partial_residuals(relevant_trials));
        sem_ys(bin_idx) = std(partial_residuals(relevant_trials))./sqrt(length(relevant_trials));
    end
    partial_residual_plot.bin_edges = bin_edges;
    partial_residual_plot.bin_centers = bin_centers;
    partial_residual_plot.bin_idxs = bin_idxs;
    partial_residual_plot.mu_xs = mu_xs;
    partial_residual_plot.sem_xs = sem_xs;
    partial_residual_plot.mu_ys = mu_ys;
    partial_residual_plot.sem_ys = sem_ys;

end
