function [data] = remove_nonresponsive_trials(data, nonresponsive_colname, remove_nonresponsive)
% If remove_nonresponsive = true, remove these trials.
% If = false, rewrite NaNs into -1.
    if(nargin==1)
        nonresponsive_colname = "a";
        remove_nonresponsive=true;
    elseif(nargin==2)
        remove_nonresponsive=true;
    end
    
    n_subj = length(data);
    for s=1:n_subj
        if(remove_nonresponsive)
        orig_n_trials = length(data(s).(nonresponsive_colname));
        valid_trials = find(~isnan(data(s).(nonresponsive_colname)));

        % Find cols whose entries are trial-by-trial vectors. 
        fields = fieldnames(data(s));
        for j = 1:numel(fields)
            % Get the value of the current field
            value = data(s).(fields{j});
            % Check if the field is a vector of size 60
            if (isvector(value) && length(value) == orig_n_trials)
                data(s).(fields{j}) = data(s).(fields{j})(valid_trials);
            end
        end
        

        % data(s).cond = data(s).cond(valid_trials);
        % data(s).s = data(s).s(valid_trials);
        % data(s).a = data(s).a(valid_trials);
        % data(s).corrchoice = data(s).corrchoice(valid_trials);
        % data(s).acc = data(s).acc(valid_trials);
        % data(s).r = data(s).r(valid_trials);
        % data(s).rt = data(s).rt(valid_trials);
        % data(s).tt = data(s).tt(valid_trials);
        % data(s).block = data(s).block(valid_trials);
        else
            invalid_trials = find(isnan(data(s).(nonresponsive_colname)));
            data(s).(nonresponsive_colname)(invalid_trials) = -1;
        end

    end

end