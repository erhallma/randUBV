function [U,B,V,errs,defl] = randUBV_k(A, k, b)
% [U,B,V] = randUBV_k(A, k, b)
% The fixed-rank randUBV algorithm.
% It produces a factorization UBV of A
% k is the approximation rank
% b is the block size, usually a factor of k. 
% OUTPUTS: 
%          U,B,V: the approximation factors
% -----------------OPTIONAL OUTPUTS
%          errs : the error approximations
%          defl : the number of deflations in U and/or V
% ---------------------------
% First commit: December 2020
% Last update : February 2021
% ---------------------------

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
    
    r = 0; % row indexing for B
    c = 0; % column indexing for B
    
    if nargout > 3
        errs = []; 
        defl = 0; 
    end
        
    for j = 1:maxiter
        
        % Compute next U, R
        [Uk,R,p] = qr(A*Vk-Uk*Lt',0); 
        dr = find(abs(diag(R))>=deflTol, 1, 'last'); 
        if isempty(dr)
            dr = 0; 
        end
        pt = zeros(1,b); pt(p) = 1:b; 
        R  = R(1:dr,pt);
        Uk = Uk(:,1:dr); 
        U  = [U,Uk]; %#ok<AGROW>
        
        % Update B and error indicator
        B(r+1:r+dr,c+1:c+b) = R;
        c = c + b; 
        E = E - norm(R,'fro')^2;
        
        % Compute next V, L
        Vk = A'*Uk - Vk*R';
        Vk = Vk - V*(V'*Vk); % full reorthogonalization
        [Vk,Lt,p] = qr(Vk,0); 
        pt = zeros(size(p)); pt(p) = 1:dr; 
        dc = find(abs(diag(Lt))>=deflTol, 1, 'last'); 
        if isempty(dc)
            dc = 0;
        end
        Lt = Lt(1:dc,pt); 
        Vk = Vk(:,1:dc); 
        V  = [V,Vk]; %#ok<AGROW>
        
        % Augmentation 
        if dc < b
            Vaug = randn(n,b-dc); 
            Vaug = qr(Vaug - V*(V'*Vaug),0); 
            V = [V,Vaug]; %#ok<AGROW>
            Vk = [Vk,Vaug]; %#ok<AGROW>
        end
        
        % Update B and error indicator
        B(r+1:r+dr,c+1:c+dc) = Lt'; 
        r = r + dr; 
        E = E - norm(Lt,'fro')^2;  
        
        % ------- optional computations
        if nargout > 3
            errs(end+1) = sqrt(E); %#ok<AGROW>
            defl = defl + (b-dc); 
        end
        
    end 
    
end 
