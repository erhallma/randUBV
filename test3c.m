% Experiment 3b
% Effect of block size 
% on speed and convergence
% Test case: SuiteSparse matrix
% -------------------------

rng(1); % set the seed for reproducibility

load lp_cre_b.mat
load lp_cre_b_SVD.mat

A = (Problem.A)'; 
sa = S.s; 

relerr = 0.5; 

csa = cumsum(sa.^2); 
csa = csa/csa(end); 
errfOpt = sqrt(1-csa);
rmin = find(errfOpt<relerr,1,'first'); 

%%
bs = [1,2,4,8,12,16,20,24,28,32,48,64,96,128]; 
ts = zeros(size(bs)); 
rs = zeros(size(bs)); 

tsqb = zeros(size(bs)); 
rsqb = zeros(size(bs)); 

tsqb0 = zeros(size(bs)); 
rsqb0 = zeros(size(bs)); 

for i = 1:length(bs)
    b = bs(i); 
    tic
    [U,B,V,E] = randUBV(A,relerr,b); 
    ts(i) = toc;
    rs(i) = size(U,2);
    fprintf("block size: %d, rank: %d\n", b, size(U,2))
    
    tic
    [Q,B] = randQB_EI_auto(A,relerr,b,0); % P = 0
    tsqb0(i) = toc; 
    rsqb0(i) = size(Q,2); 
    fprintf("qb rank: %d\n", rsqb0(i)); 
    
    tic
    [Q,B] = randQB_EI_auto(A,relerr,b,1); % P = 1
    tsqb(i) = toc; 
    rsqb(i) = size(Q,2); 
    fprintf("qb rank: %d\n", rsqb(i)); 
end

%%
close all
figure
semilogx(bs,rs,'k-o','linewidth',1), hold on
semilogx(bs,rsqb0(1,:),'k:*','linewidth',1)
semilogx(bs,rsqb(1,:),'k--x','linewidth',1)
yline(rmin,'k:','linewidth',2)
ylim([500,1200])
xlabel('Block size','fontsize',16)
ylabel('Approximation rank')
lgd1 = legend('UBV','QB(P=0)','QB(P=1)','Optimal'); 
lgd1.FontSize = 15; 
lgd1.Location = 'best'; 
ax = gca; 
ax.FontSize = 16; 
print('plots/blockrank3','-dpng')

figure
semilogx(bs,ts,'k-o','linewidth',1), hold on
semilogx(bs,tsqb0(1,:),'k:*','linewidth',1)
semilogx(bs,tsqb(1,:),'k--x','linewidth',1)
lgd2 = legend('UBV','QB(P=0)','QB(P=1)'); 
lgd2.FontSize = 15; 
lgd2.Location = 'best'; 
ax = gca; 
ax.FontSize = 16; 
xlabel('Block size','fontsize',16)
ylabel('Time (s)')
ylim([0,30])
print('plots/blocktime3','-dpng')
