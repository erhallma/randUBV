% accTest.m
% Convergence of randUBV vs randQB
% --------------------------------
% Code to produce Figures 1 and 2
% --------------------------------
rng(1)      % set the random seed for reproducibility

N = 2000; 
b = 10; 
k = 200;  

%% Generate random matrices U,V

Ua = orth(randn(N)); 
Va = orth(randn(N)); 

%% Set the matrix A
% Method 1: slow decay
% Method 2: fast decay
% Method 3: Devil's stairs

SA = {(1:N).^(-2), exp(-(1:N)/20), devils_svd(N,30)}; 
labels = {'Slow','Fast','Stairs'}; 
xlims = {[100,200],[175,200],[110,200]}; 
%%
for i = 1:3

sa = SA{i};
A = (Ua.*sa)*Va';

%% Randomized methods

%% UBV
[U,B,V,errs,errU,errUloc,errV,errEst,defl] = randUBV_k(A,k,b); 
fprintf("Trial %d deflations: %d\n", i, defl); 

%% QB, P=0
P = 0; 
[Q0, B0, err0, errQ0, ~] = randQB_EI_k(A, k, b, P);  

%% QB, P=1
P = 1; 
[Q1, B1, err1, errQ1, ~] = randQB_EI_k(A, k, b, P); 

%% QB, P=2
P = 2; 
[Q2, B2, err2, errQ2, ~] = randQB_EI_k(A, k, b, P); 

%% Normalize the errors
errs = errs/norm(sa); 
err0 = err0/norm(sa);
err1 = err1/norm(sa);
err2 = err2/norm(sa);
errEst = errEst/norm(sa); 

%% Plot results
csa = cumsum(sa.^2); 
csa = csa/csa(end); 
errfOpt = [1,sqrt(1-csa)]; 

figure('Position',[200 200 800 350])

subplot(1,2,1)
semilogy(0:k,errfOpt(1:k+1),'k','linewidth',2), hold on
%%
plot(b:b:k, errs,'ko')
plot(b:b:k, err0,'k+')
plot(b:b:k, err1,'ks')
plot(b:b:k, err2,'kp')
xlim([0,k])
xlabel('k','fontsize',16)
ylabel('||A-QB||_F/||A||_F','fontsize',16)
lgd1 = legend('SVD','UBV','QB(P=0)','QB(P=1)','QB(P=2)'); 
lgd1.FontSize = 15; 
if i == 3
    lgd1.Location = 'southwest'; 
end
ax = gca; 
ax.FontSize = 16; 

%%
subplot(1,2,2)
%%
semilogy(b:b:k,errU,'k--','Linewidth',0.8), hold on 
plot(b:b:k,errV(1:end-1),'k','Linewidth',0.8)
plot(b:b:k,errUloc,'k.','Linewidth',0.8)

xlabel('k','fontsize',16)
ylabel('Loss of Orthogonality','fontsize',16)
lgd2 = legend('$$\|U^TU-I\|_2$$','$$\|V^TV-I\|_2$$',...
                                '$$\varepsilon_k$$');
lgd2.FontSize = 15; 
lgd2.Interpreter = 'latex'; 
lgd2.Location = 'northwest'; 
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
xlabel('k','fontsize',20)
ylabel('$$\sigma_k(B_{200})$$','FontSize',24,'interpreter','latex')
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
    % Creates an nxn vector with clustered singular values
    % in clusters of size L. 
    
    s = zeros(1,n); 
    Nst = floor(n/L); 
    
    for i = 1:Nst
        s(1+L*(i-1):L*i) = -0.6*(i-1); 
    end
    s(L*Nst:end) = -0.6 * (Nst-1); 
    sa = 10.^s; 
end