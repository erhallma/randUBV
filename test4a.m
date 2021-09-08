% Experiment 4a
% Test the effect of lowering the stopping
%  tolerance on the eventual truncated rank
% Test case: photograph
% ---------------------

rng(1); % set the seed for reproducibility

A = double(rgb2gray(imread('pinus_glabra.jpg')))';

%% SVD information for I
tic
[U,S,V] = svd(A,'econ'); 
sa = diag(S); 
tfull = toc; 

fprintf("Time for photo SVD: %.4f\n", tfull); 
normAf = norm(sa);
csa = cumsum(sa.^2); 
csa = csa/csa(end); 
errf = sqrt(1-csa);

%% Stopping tolerances

tol0 = 0.09; 
tol  = 0.1; 
b    = 20; 
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
    [Q,B] = randQB_EI_auto(A, tol, b, P);
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
