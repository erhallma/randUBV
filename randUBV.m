function [U,B,V,E] = randUBV(A, relerr, b)
% [U,B,V,E] = randUBV(A, relerr, b)
% The fixed-precision randUBV algorithm.
% It produces a factorization UBV of A that satisfies
%     ||A-UBV||_F <= ||A||_F* relerr.
% b is the block size. 
% E : approximation error estimate
% ---------------------------------
% Code for Algorithm 4.1 
% ---------------------------------


    [m,n] = size(A); 
    maxiter = floor(min(m,n)/(3*b)); 
 
    E = norm(A,'fro')^2;
    threshold = relerr^2*E; 
    deflTol = 1e-12*sqrt(norm(A,1)*norm(A,Inf));

    Omega = randn(n,b); 
    [Vk,~] = qr(Omega,0);
    Uk = zeros(m,0); 
    Lt = zeros(b,0);

    V = Vk;
    U = Uk; 
        
    for k = 1:maxiter
        ix = b*(k-1); 
        
        % Compute next U, R
        [Uk,R,du] = reinfQR(A*Vk-Uk*Lt',deflTol);
        if du > 0 
            [Uk,~] = qr(Ud - U*(U'*Ud), 0);
        end
        U = [U,Uk]; %#ok<AGROW>
        
        B(ix+1:ix+b,ix+1:ix+b) = R;
        E = E - norm(R,'fro')^2;
        
        % Compute next V, L
        Vk = A'*Uk - Vk*R';
        Vk = Vk - V*(V'*Vk);
        [Vk,Lt,dv] = reinfQR(Vk,deflTol);
        if dv > 0
           [Vk,~] = qr(Vk - V*(V'*Vk),0); 
        end
        V = [V,Vk]; %#ok<AGROW>
        
        B(ix+1:ix+b,ix+b+1:ix+2*b) = Lt'; 
        E = E - norm(Lt,'fro')^2;  
        
        if E < threshold
            break
        end
    end
end 


% -------------------
% Auxiliary functions
% -------------------

function [Q,R,d] = reinfQR(X,tau)
    % Computes QR with deflation tolerance tau
    % and reinflates Q with random vectors
    % d = number of deflations
    
    [Q,R] = qr(X,0); 
    if nargin < 2, tau = (1e-10)*norm(R); end
    
    H = @(u,x) x - u*(u'*x); 
    HT = @(u,x) x - (x*u)*u'; 
    
    d = 0; 
    [m,b] = size(X); 
    
    for j = 1:b
        i = j-d; 
        
        [u,normx] = house_gen(R(i:j,j)); 
        if normx < tau 
            R(i:j,j) = 0; 
            d = d + 1; 
        elseif d >= 1
            R(i,j) = -normx*sign(u(1)); 
            R(i+1:j,j) = 0;
            R(i:j,j+1:b) = H(u,R(i:j,j+1:b)); 
            Q(:,i:j) = HT(u,Q(:,i:j)); 
        end   
    end
    
    if d > 0    % reinflation, if needed
        i = b-d; 
        Omg = randn(m,d); 
        Qi  = orth(Omg - Q(:,1:i)*(Q(:,1:i)'*Omg)); 
        Q(:,i+1:b) = Qi; 
    end
end

function [u,nu] = house_gen(x)
    % Generate Householder reflection H = I-uu'
    % Adapted from code by Cleve Moler (2016)
    
    sig = @(u) sign(u) + (u==0);
    
    nu = norm(x);
    if nu ~= 0
        u = x/nu;
        u(1) = u(1) + sig(u(1));
        u = u/sqrt(abs(u(1)));
    else
        u = x;
        u(1) = sqrt(2);
    end
end
