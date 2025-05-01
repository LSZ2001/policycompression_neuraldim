base_folder = 'C:/Users/liu_s/policycompression_neuraldim';
cd(base_folder)
addpath(genpath(base_folder))

rawfile_path = "EEG_paper_analysis\Dimension\";
table_path = "EEG_paper_analysis\new_dimensionality_files\";
subjids = [304:307,309:332]; % One subject excluded, hence no file

for subjid =subjids
    subjid
    raw_filename = "A"+subjid+"_DIM_READOUT_RL_B_pred_DIM.h5";
    dimfile_savename = "A"+subjid+"_DIM_READOUT_RL_processed.txt";
    dimensionality_estimates = compute_neural_dimensionality_fun(rawfile_path+raw_filename);
    writetable(dimensionality_estimates, table_path+dimfile_savename)
end


%%
function [dimensionality_estimates] = compute_neural_dimensionality_fun(h5_file)
    num_datafiles = length(h5info(h5_file).Datasets);
    num_partitions = zeros(num_datafiles,1);

    for datafile_idx =1:num_datafiles
        datafile_idx
        data_temp = h5read(h5_file, "/cmbSG"+datafile_idx);
        if(datafile_idx==1)
            % Extract info common to all data for the subject
            SUBID = data_temp.SUBID;
            BLOCK = data_temp.BLOCK;
            TRIAL = data_temp.TRIAL;
            time = data_temp.time;
        end
        % Extract partition names
        partition_idxs = fieldnames(data_temp);
        partition_idxs = partition_idxs(5:end); % Remove SUBID, BLOCK, TRIAL, time
        num_partitions(datafile_idx) = length(partition_idxs);

        data_temp = rmfield(data_temp,{'SUBID','BLOCK','TRIAL','time'});
        % Sum the matrix along the second dimension (column-wise)
        if(datafile_idx==1)
            accuracies_sumoverpartitions = sum(cell2mat(struct2cell(data_temp)'), 2); % Result is a 200x1 double vector
        else
            accuracies_sumoverpartitions = accuracies_sumoverpartitions + sum(cell2mat(struct2cell(data_temp)'), 2); % Result is a 200x1 double vector
        end
        clear data_temp partition_idxs
    end
    % Take the average accuracy over all partitions
    dimensionality_estimate = accuracies_sumoverpartitions ./ sum(num_partitions);
    dimensionality_estimates = table(SUBID, BLOCK, TRIAL, time, dimensionality_estimate);
    
end