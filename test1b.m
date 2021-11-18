% Experiment 1b
% Testing the accuracy of the error indicator
%  using U(k+1) instead of U(k), to align better
%  with the results of Theorem 4.3
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

%% Run the experiment
for i = 1:4

sa = SA{i};
A = (Ua.*sa)*Va';

% Randomized methods
% UBV
[U,B,V,errUBV,defl] = randUBV_k(A,k+b,b); 

Uk = U(:,1:k); 
Bk = B(1:k,1:(k+b)); 
Vk = V(:,1:(k+b)); 

% accuracy of error indicator
fprintf("i = %d\n", i)
pred_err = errUBV(end-1); 
act_err = norm(A-Uk*Bk*Vk','fro'); 
fprintf("Deflations: %d\n", defl); 
X = U'*U - eye(size(U,2)); 
loc_err = 0; 
for j = 0:k/b
    xj = X(j*b+1:j*b+b,j*b+1:j*b+b); 
    loc_err = max(loc_err,norm(xj));
    if j < k/b
        xj = X(j*b+1:j*b+b,j*b+b+1:j*b+2*b);
        loc_err = max(loc_err,norm(xj)); 
    end
end
fprintf("Total Loss of Orthogonality in U: %.4e\n", norm(X)); 
fprintf("Local Loss of Orthogonality in U: %.4e\n", loc_err);
fprintf("Squared predicted error: %.4e\n", pred_err^2); 
fprintf("Absolute squared difference: %.4e\n", abs(pred_err^2-act_err^2)); 
fprintf("Theoretical bound: %.4e\n", 4*loc_err*norm(A,'fro')^2); 

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