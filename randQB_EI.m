function [Q,B,errf2,err2] = randQB_EI(A, tol, b, P, maxit)
    % Block bidiagonalization of a matrix A 
    
    [m,n] = size(A); 
    
    normA2 = norm(A,'fro')^2; 
    normE2 = normA2; 
    errf2(1) = normA2;
    if nargout >= 4
        err2(1) = svds(A,1,'largest','Tolerance',1e-4);
    end
    
    if nargin < 2, tol = 0.1; end
    if nargin < 3, b = max(4, ceil(min(m,n)/10)); end
    if nargin < 4, P = 0; end
    if nargin < 5, maxit = floor(min(m,n)/b); end
    
    Q = zeros(m,0); 
    BT = zeros(n,0); % Store B^t for column-major ordering. 
     
    
    for k = 1:maxit
        omega = randn(n,b); 
        for i = 1:P
            [G,~] = qr(A*omega - Q*(BT'*omega),0); 
            [omega,~] = qr(A'*G- BT*(Q'*G),0); 
        end
        [Qk,~] = qr(A*omega-Q*(BT'*omega),0);
        [Qk,~] = qr(Qk - Q*(Q'*Qk),0); 
        
        
        Bk = Qk'*A; 
        Q = [Q, Qk]; 
        BT = [BT, Bk']; 
        
        normE2 = normE2 - norm(Bk,'fro')^2;
        
        errf2(k+1) = normE2; 
        if nargout >= 4
            err2(k+1) = svds(A-Q*BT',...
                        1,'largest','Tolerance',1e-4);
        end
    
        if normE2 <= tol^2*normA2
            break
        end 
    end
    
    B = BT'; 
 
end