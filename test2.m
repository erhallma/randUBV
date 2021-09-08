% Experiment 2
% Speed on rectangular matrices
% Accuracy of randUBV vs randQB
% -----------------------------

rng(1)      % set the random seed for (limited) reproducibility

Ms = 24000; 
N = 4000;  
K = 600; 
D = 0.008; 
b = 20; 
ntrials = 10; 

ms = 8000*(1:5);
ks = 200*(1:5); 
ds = 0.004*(1:5); 
labels = {'Rows', 'Rank', 'Density'}; 
tUBV = zeros(ntrials,5); 
tQB  = zeros(ntrials,5); 

%% Part 1: fix size and sparsity, vary the rank
for itn = 1:3
    fprintf("Variation: %d\n", itn); 
    for i = 1:5
        fprintf("Step: %d\n", i); 
        if itn == 1
            m = ms(i); k = ks(3); d = ds(2); 
        elseif itn == 2
            m = ms(3); k = ks(i); d = ds(2);
        else
            m = ms(3); k = ks(3); d = ds(i);
        end

        for j = 1:ntrials
            fprintf("Trial: %d\n", j); 
            A = sprandn(m, N, d); 
            tic 
            [U,B,V] = randUBV_k(A, k, b);
            tUBV(j,i) = toc; 

            tic
            [Q0, B0] = randQB_EI_k(A, k, b, 0);
            tQB(j,i) = toc; 
        end
    end

    muUBV = mean(tUBV) 
    muQB  = mean(tQB)

    figure, hold on
    if itn == 1
        xpts = ms; 
        xlabel('Number of rows')
    elseif itn == 2
        xpts = ks; 
        xlabel('Approximation rank')
    else
        xpts = 100*ds; 
        xlabel('% nonzero elements')
    end
    plot(xpts,muUBV,'k','linewidth',1)
    plot(xpts,muQB,'k--','linewidth',2)
    ylabel('Time (s)')
    lgd = legend('UBV','QB(P=0)'); 
    lgd.Location = 'best'; 
    lgd.FontSize = 15; 
    ax = gca; 
    ax.FontSize = 16; 
    
    s = sprintf('plots/time%s',labels{itn}); 
    print(s,'-dpng')
    
end

