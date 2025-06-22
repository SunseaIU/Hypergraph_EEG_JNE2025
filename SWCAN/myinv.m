function [A_inv] = myinv(A, tol)
    if nargin<2
        tol = 1e-15;
    end
    [U, S, V] = svd(A);
    
    % 计算 S 的伪逆
    S_pinv = zeros(size(S'));
    for i = 1:min(size(S))
        if S(i,i) > tol % 只对非零奇异值取倒数
            S_pinv(i,i) = 1 / S(i,i);
        end
    end
    
    % 计算 A 的伪逆
    A_inv = V * S_pinv * U';
end

