function [x, v, rec] = SimplexQP_ALM(A, b, mu, beta, pic, x)
% solve:
% min     x'Ax - b'x
% s.t.    x'1 = 1, x >= 0
% paras:
% mu    - mu > 0
% beta  - 1 < beta < 2
% 

NITER = 100;
THRESHOLD = 1e-8;

val = 0;

if nargin < 6
    x = zeros(size(b));
end
v = ones(size(x));

% TODO: consider the initialization of multipliers
lambda = ones(size(x));
cnt = 0;

if pic
    rec = [];
end

for iter = 1:NITER
    x = EProjSimplex_new(v - 1/mu*(lambda + 0.5*A*v-b));
    v = x + 1/mu*(lambda - 0.5*A'*x);
    lambda = lambda + mu*(x - v);
    mu = beta*mu;
    
    val_old = val;
    val = 0.5*x'*A*x - b'*x;
    
    if pic
        rec = [rec val];
    end
    
    if abs(val - val_old) < THRESHOLD
        if cnt >= 5
            break;
        else
            cnt = cnt + 1;
        end
    else
        cnt = 0;
    end
end
obj = 0.5*x'*A*x - b'*x;
obj = roundn(obj,-6);
% fprintf('Using SimplexQP, relax gap: %.5f, iter: %d, mu: %.3f, Obj: %.6f\n',  norm(x-v), iter, mu, obj);
% if pic
%     plot(rec);
% end

end