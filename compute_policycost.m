clear all; close all;
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
figpath = "figures\"; %"newplots\"
decodingpath = "EEG_paper_analysis\Decoding";
RSApath = "EEG_paper_analysis\RSA";
png_dpi = 500;

% Color palettes
cmap = brewermap(3, 'Set1');
cmap = cmap([1,3,2],:);
cmap_subj = brewermap(200, 'Set1');
cmap_exp3 = brewermap(10, 'Set2');
cmap_exp3 = cmap_exp3([2,1,3,4,5,6,9,7,8,10],:);

%% Compute policy cost of each trial (only need to do once to save .txt files
T_raw = readtable("BEH\"+'HAL2017_READOUT_TEST_BehP.txt');
T_raw = T_raw(:, {'SUBID', 'BLOCK', 'TRIAL', 'CUETYPE', 'STIMPOS', 'TASK', ...
            'CORRESP', 'RESP', 'RT', 'ACC', 'ACCRESP', 'SOA', 'SOACODE', 'VALID','OMIT','PRELATE'});
T_raw = T_raw(T_raw.RESP<=4,:);
subjs = unique(T_raw.SUBID);
n_subj = length(subjs);
n_conds = length(unique(T_raw.SOACODE));
n_stims = 4; 
T_raw = T_raw(T_raw.OMIT ~= 1, :);

% Compute policy cost for all retained trials, assuming that they are adjacent.
T_raw.cost = zeros(height(T_raw),1);
T_raw.marginal = zeros(height(T_raw),1);
T_raw.surprisal = zeros(height(T_raw),1);
T_SUBJ_BLOCKS = T_raw;
for subj =1:n_subj
    subj
    T_subj = T_raw(T_raw.SUBID==subjs(subj),:);
    blocks = unique(T_subj.BLOCK);

    % Fill in T_SUBJBLOCKS: computes P(a) of the current block.
    for block=1:length(blocks)
        subjblock_trials = find(logical((T_SUBJ_BLOCKS.SUBID==subjs(subj)).*(T_SUBJ_BLOCKS.BLOCK==block))==1);
        T_subjblock = T_SUBJ_BLOCKS(subjblock_trials,:);
        states = (T_subjblock.TASK-1).*n_stims + T_subjblock.STIMPOS;
        actions = T_subjblock.RESP;
        if(~isempty(states)) % No trials for this subject in this block
            [T_subjblock.cost, T_subjblock.marginal, T_subjblock.surprisal] = compute_cost_running(states,actions);
            T_SUBJ_BLOCKS(subjblock_trials,:) = T_subjblock;
        end
    end
end
writetable(T_SUBJ_BLOCKS, "BEH\"+"T_policycost_Paforeachblock.txt")

