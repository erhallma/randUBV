% Experiment 1
% Convergence by iteration on a variety of matrices
% Accuracy of randUBV vs randQB
% -----------------------------
rng(1)      % set the random seed for reproducibility

N = 2000; 
b = 10; 
k = 200;  

%% Generate random matrices U,V

Ua = orth(randn(N)); 
Va = orth(randn(N)); 

%% Set the matrix A
% Method 1: fast decay
% Method 2: slow decay
% Method 3: slower decay
% Method 4: Devil's stairs

SA = { exp(-(1:N)/20),(1:N).^(-2),(1:N).^(-1), devils_svd(N,30)}; 
labels = {'Fast','Slow','Slower','Stairs'}; 
xlims = {[175,200],[100,200],[100,200],[110,200]}; 
%%
for i = 1:4

sa = SA{i};
A = (Ua.*sa)*Va';

% Randomized methods
%% UBV
[U,B,V,errUBV,defl] = randUBV_k(A,k,b); 


%% accuracy of error indicator
fprintf("i = %d\n", i)
pred_err = errUBV(end); 
act_err = norm(A-U*B*V','fro'); 
fprintf("Deflations: %d\n", defl); 
X = U'*U - eye(size(U,2)); 
loc_err = 0; 
for j = 0:k/b-1
    xj = X(j*b+1:j*b+b,j*b+1:j*b+b); 
    loc_err = max(loc_err,norm(xj));
    if j < k/b-1
        xj = X(j*b+1:j*b+b,j*b+b+1:j*b+2*b);
        loc_err = max(loc_err,norm(xj)); 
    end
end
fprintf("Total Loss of Orthogonality in U: %.4e\n", norm(X)); 
fprintf("Local Loss of Orthogonality in U: %.4e\n", loc_err);
fprintf("Squared predicted error: %.4e\n", pred_err^2); 
fprintf("Absolute squared difference: %.4e\n", abs(pred_err^2-act_err^2)); 
fprintf("Theoretical bound: %.4e\n", 4*loc_err*norm(A,'fro')^2); 

%% QB, P=0
[Q0, B0, err0,~,err_id] = randQB_EI_k(A, k, b, 0);  

%% QB, P=1
[Q1, B1, err1] = randQB_EI_k(A, k, b, 1); 

%% QB, P=2
[Q2, B2, err2] = randQB_EI_k(A, k, b, 2); 

%% Normalize the errors
errUBV = errUBV/norm(sa); 
err0 = err0/norm(sa);
err1 = err1/norm(sa);
err2 = err2/norm(sa);

csa = cumsum(sa.^2); 
csa = csa/csa(end); 
errfOpt = sqrt(1-csa);


%% Plot results
close all
figure
semilogy(errfOpt(1:k),'k','linewidth',2), hold on
%%
plot(b:b:k, errUBV,'ko')
plot(b:b:k, err0,'k+')
plot(b:b:k, err1,'ks')
plot(b:b:k, err2,'kp')
xlim([0,k])
xlabel('k','fontsize',16)
ylabel('||A-QB||_F/||A||_F','fontsize',16)
lgd1 = legend('SVD','UBV','QB(P=0)','QB(P=1)','QB(P=2)'); 
lgd1.FontSize = 15; 
ax = gca; 
ax.FontSize = 16; 

%%
s = sprintf('plots/acc%s',labels{i}); 
print(s,'-dpng')

%% Plot of the singular values at convergence
figure
semilogy(sa(1:k),'k','linewidth',1), hold on
plot(svd(B),'k--','linewidth',1)
plot(svd(B0),'k:','linewidth',1)
plot(svd(B1),'k:','linewidth',1)
plot(svd(B2),'k:','linewidth',1)
xlim(xlims{i})
lgd = legend('SVD','UBV','QB(P=0,1,2)');
lgd.Location = 'southwest'; 
lgd.FontSize = 15; 
xlabel('k','fontsize',16)
ylabel('$$\sigma_k(B_{200})$$','FontSize',18,'interpreter','latex')
ax = gca; 
ax.FontSize = 16; 

%%
s = sprintf('plots/sval%s',labels{i}); 
print(s,'-dpng')

end


% -------------------
% Auxiliary functions
% -------------------

function sa = devils_svd(n,L)
    % Creates an n-vector with clustered singular values
    % in clusters of size L. 
    
    s = zeros(n,1); 
    Nst = floor(n/L); 
    
    for i = 1:Nst
        s(1+L*(i-1):L*i) = -0.6*(i-1); 
    end
    s(L*Nst:end) = -0.6 * (Nst-1); 
    sa = 10.^s; 
end