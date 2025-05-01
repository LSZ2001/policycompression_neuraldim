%% This code is identical to compute_cost(), but computes P(a) based only on the trials before the current trial. Also, log is computed on base 2.

function [cost, marginal, surprisal] = compute_cost_running(stimulus,action)
    
    % Compute trial-by-trial policy cost, marginal distribution, and surprisal

    [S,~,si] = unique(stimulus);
    [A,~,ai] = unique(action);
    alpha = 0.1;

    nSA = zeros(length(S),length(A)) + alpha;
    pSA = nSA./sum(nSA(:));
    pS = sum(pSA,2);
    pA = sum(pSA);

    cost = zeros(size(stimulus));
    surprisal = zeros(size(stimulus));
    marginal = zeros(size(stimulus));
    

    for i = 1:length(stimulus)
        % Compute cost, marginal, and surprisal for taking the action in
        % the current trial, based on P(a) of all previous trials.
        cost(i) = log2(pSA(si(i),ai(i))) - log2(pS(si(i))) - log2(pA(ai(i)));
        marginal(i) = pA(ai(i));
        surprisal(i) = -log2(pA(ai(i)));
        
        % Update P(s), P(a) according to the current trial's state and
        % action.
        nSA(si(i),ai(i)) = nSA(si(i),ai(i))+1;
        pSA = nSA./sum(nSA(:));
        pS = sum(pSA,2);
        pA = sum(pSA);
    end

    if nargout > 1
        marginal = zeros(size(stimulus)) + pA(2);
    end
end