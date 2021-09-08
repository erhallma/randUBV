% Experiment 4b
% Test the effect of lowering the stopping
%  tolerance on the eventual truncated rank
% Test case: SuiteSparse matrix
% -----------------------------

rng(1) % Set the random seed for reproducibility

load lp_cre_b.mat
load lp_cre_b_SVD.mat

A = (Problem.A)'; 
sa = S.s; 
normAf = norm(sa); 
 

%% Spectrum information for A

figure
plot(sa(1:800),'k','Linewidth',1); 
xlabel('k','fontsize',18); 
ylabel('$$\sigma_k$$','fontsize',24,'interpreter','latex')
ylim([60,200])
print('plots/lp_cre_b','-dpng'); 

figure
ds = abs(diff(sa(1:801))); 
rs = ds./sa(1:800); 
semilogy(rs,'k','Linewidth',1); 
xlabel('k','fontsize',18); 
ylabel('$$(\sigma_k-\sigma_{k+1})/\sigma_{k+1}$$','fontsize',24,'interpreter','latex')

fprintf("Median gap: %.4e\n", median(rs)); 
fprintf("Smallest gap: %.4e\n", min(rs)); 

%%
% TRIAL 1: (0.45, 0.5)
% TRIAL 2: (0.14, 0.15)

%tol0 = 0.45; tol = 0.5; 
tol0 = 0.14; tol = 0.15; 


b = 50; 
csa = cumsum(sa.^2); 
csa = csa/csa(end); 
errf = sqrt(1-csa);
ropt = find(errf<tol,1,'first'); 

fprintf("Stopping Tolerance: %.2f\n", tol0); 
fprintf("Truncation Tolerance: %.2f\n", tol); 
fprintf("Optimal Rank: %d\n\n", ropt); 

%% UBV factorization
for itn = 1:2
    if itn == 1
        tstop = tol0;
    else
        tstop = tol;
    end

    tic
    [U,B,V] = randUBV(A,tstop,b); 
    t1 = toc;
    rel_err = norm(A-U*B*V','fro')/norm(A,'fro'); 
    fprintf("Approximation error: %.4e\n", rel_err);
    fprintf("U error: %.4e\n", norm(U'*U-eye(size(U,2))));
    tic
    %[Ub,S,Vb] = svd(B,'econ'); 
    [Ub,S,Vb] = eigSVD(B); 
    s  = diag(S); 
    s  = sort(s,'descend');
    r  = length(s);  
    err= sqrt(1 - cumsum(s.^2)/normAf^2); 
    rT = find(err<tol,1,'first');
    U = U*Ub(:,1:rT); 
    V = V*Vb(:,1:rT); 
    t2 = toc; 


    fprintf("UBV, tStop = %.4f\n",tstop)
    fprintf("Factorization time: %.4f\n", t1);  
    fprintf("SVD time: %.4f\n", t2); 
    fprintf("Total time: %.4f\n", t1+t2);
    fprintf("Initial rank: %d\n", r); 
    fprintf("Truncated rank: %d\n\n", rT);

end


%% QB factorizations

for P = 0:2
    tic
    [Q,B] = randQB_EI_auto(A', tol, b, P);
    t1 = toc; 
    
    tic
    %[Ub,S,Vb] = svd(B,'econ');
    [Ub,S,Vb] = eigSVD(B); 
    s  = diag(S); 
    s  = sort(s,'descend');
    r  = length(s);  
    errqb = sqrt(1 - cumsum(s.^2)/normAf^2); 
    rT = find(errqb<tol,1,'first'); 
    U = Q*Ub(:,1:rT); 
    t2 = toc;

    fprintf("QB, P=%d\n", P)
    fprintf("Factorization time: %.4f\n", t1);
    fprintf("SVD time: %.4f\n", t2); 
    fprintf("Total time: %.4f\n", t1+t2);
    fprintf("Initial rank: %d\n", r); 
    fprintf("Truncated rank: %d\n\n", rT);

end


function [U,S,V] = eigSVD(A)
    tflag = false;
    if size(A,1)<size(A,2)
        A = A'; 
        tflag = true; 
    end
    B = A'*A; 
    [V,D] = eig(B,'vector'); 
    S = sqrt(D); 
    U = A*(V./S'); 
    if tflag
        tmp = U; 
        U = V; 
        V = tmp; 
    end
    S = diag(S); 
end