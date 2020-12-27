% speedTests.m
% Runtime of randUBV vs randQB
%  on a dense matrix
% --------------------------------
% Code to produce Figure 3
% --------------------------------

rng(1); % Set the seed for reproducibility

Ns = [4000,16000]; 
b = 20; 
nTrials = 10;
Ms = {5:5:50,2:2:20};

%%
for matType = 1:2
%% Form the matrix 
    N = Ns(matType); 
    maxits = Ms{matType}; 

    if matType == 1
        A = randn(N); 
    else
        A = sprand(N,N,0.01);
    end


    %% Speed tests
    SUBV = zeros(nTrials,length(maxits)); 
    SQB0 = zeros(nTrials,length(maxits));
    SQB1 = zeros(nTrials,length(maxits));
    SQB2 = zeros(nTrials,length(maxits));
    %%
    for itn = 1:length(maxits)
        fprintf("itn: %d\n", itn); 
        k = b*maxits(itn);
        for j = 1:nTrials
            fprintf("Trial: %d\n", j); 

            tic
            [U,B,V] = randUBV_k(A,k,b);
            SUBV(j,itn) = toc;

            tic
            [Q0,B0] = randQB_EI_k(A, k, b, 0);
            SQB0(j,itn) = toc; 

            tic
            [Q1,B1] = randQB_EI_k(A, k, b, 1);
            SQB1(j,itn) = toc; 

            tic
            [Q2,B2] = randQB_EI_k(A, k, b, 2);
            SQB2(j,itn) = toc; 

        end
    end

    tUBV = mean(SUBV); 
    tQB0 = mean(SQB0); 
    tQB1 = mean(SQB1); 
    tQB2 = mean(SQB2); 

    %% Plot time results
    figure
    plot(b*maxits, tUBV,'k','linewidth',2), hold on
    plot(b*maxits, tQB0,'k--','linewidth',2)
    plot(b*maxits, tQB1,'k:','linewidth',2)
    plot(b*maxits, tQB2,'k-.','linewidth',2)

    xlabel('k','fontsize',16)
    ylabel('Time (s)','fontsize',16)
    ax = gca; 
    ax.FontSize = 16; 
    lgd = legend('UBV','QB(P=0)','QB(P=1)','QB(P=2)'); 
    lgd.Location = 'northwest'; 
    lgd.FontSize = 15;
    if matType == 1
        xlim([100,1000])
    else
        xlim([40,400])
    end
    
end
