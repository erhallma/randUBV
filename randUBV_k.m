function [U,B,V,errs,errU,errUloc,errV,errEst,defl] = randUBV_k(A, k, b)
% [U,B,V] = randUBV_k(A, k, b)
% The fixed-rank randUBV algorithm.
% It produces a factorization UBV of A
% k is the approximation rank
% b is the block size, usually a factor of k. 
% OUTPUTS: 
%          U,B,V: the approximation factors
% -----------------OPTIONAL OUTPUTS
%          errs : the errors ||A-UBV'||_F
%          errU : ||I-U'*U||_2
%          errUloc : max ||U_{i} - U{i+1}||_2
%          errV : ||I-V'*V||_2
%          errEst: the error approximations, ||A||^2 - ||B||^2
%          defl : the number of upper+lower deflations

    [m,n] = size(A); 
    maxiter = ceil(k/b); 
 
    E = norm(A,'fro')^2;
    deflTol = 1e-12*sqrt(norm(A,1)*norm(A,Inf)); 
    
    Omega = randn(n,b); 
    [Vk,~] = qr(Omega,0);
    Uk = zeros(m,0); 
    Lt = zeros(b,0); 

    V = Vk;
    U = Uk; 
        
    if nargout > 3
        errs      = []; 
        errEst    = []; 
        errU      = []; 
        errV      = norm(V'*V - eye(b), 'Inf'); 
        defl      = 0; 
    end
    
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
        
        % ------- optional computations
        
        if nargout > 3
            errs(end+1) = norm(A-U*B*V','fro'); %#ok<AGROW>
            errEst(end+1) = E; %#ok<AGROW>
            errU(end+1) = norm(U'*U - eye(b*k)); %#ok<AGROW>
            errV(end+1) = norm(V'*V - eye(b*(k+1))); %#ok<AGROW> 
            defl        = defl + du + dv; 
            if k == 1
                ek         = 0; 
                errUloc(1) = 0; 
            else
                Uold = U(:,ix-b+1:ix); 
                ek = max(ek, norm(Uold'*Uk)); 
                errUloc(end+1) = ek; %#ok<AGROW>
            end
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
