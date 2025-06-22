H=[1,1,0;1,1,0;1,0,0;1,1,0;0,0,1;1,0,1];
U=diag([1,1,1,1,2,2]);
W=diag([2,2,1]);
F=randn(6,3);
F = exp(F) ./ sum(exp(F), 2);
Dv = diag(sum(H * W, 2));
De = diag(sum(H' * U, 2));
Dv1=diag(diag(Dv).^(-1/2));
De1=inv(De);
oDe=diag(sum(H,1));
oDe1=inv(oDe);

l = 3;
m = 3;
n = 6;
c = 3;
M = zeros(n-l,c);
for i = l+1:n
    for j = l+1:n
        for k = 1:m
            sijk = (W(k,k)*U(i,i)*H(i,k)*U(j,j)*H(j,k))/De(k,k);
            if i == j
                sijk = 0;
            end
            M(i-l,:) = M(i-l,:) + sijk*F(j,:);
        end
    end
%     F(i,:) = EProjSimplex_new(m);
end
Lt=U(n-l+1:n,n-l+1:n)*H(n-l+1:n,:)*W*pinv(De)*H(n-l+1:n,:)'*U(n-l+1:n,n-l+1:n);
Lt=Lt-diag(diag(Lt));
M1=Lt*F(l+1:n,:);
% S = (W .* U) * H .* (U * H) ./ De;
% 
% % 将对角线元素设置为0，因为j不等于i
% for i = 1:n
%     S(i,i) = 0;
% end
% 
% % 计算m向量
% M = sum(S, 2) * F;

% tr1=0;
% otr1=0;
% 
% L=U-Dv1*U*H*W*De1*H'*U*Dv1;
% L=U*Dv-U*H*W*De1*H'*U;
% tr=trace(F'*L*F);
% 
% oL=eye(6)-Dv1*H*W*oDe1*H'*Dv1;
% oL=Dv-H*W*oDe1*H';
% otr=trace(F'*oL*F);
% 
% s=zeros(6,6,3);
% os=zeros(6,6,3);
% for i=1:6
%     for j=1:6
%         for k=1:3
%             s(i,j,k)=(W(k,k)*U(i,i)*H(i,k)*U(j,j)*H(j,k))/De(k,k);
%             os(i,j,k)=(W(k,k)*H(i,k)*H(j,k))/oDe(k,k);
%             fi=F(i,:);
%             fj=F(j,:);
%             %fi=F(:,i);
%             %fj=F(:,j);
%             tr1=tr1+(s(i,j,k)*(norm(fi-fj,2)^2));
%             otr1=otr1+(os(i,j,k)*(norm(fi-fj,2)^2));
%         end
%     end
% end
% fprintf('%f %f %f\n',tr1,tr,tr1/tr)
% fprintf('%f %f %f\n',otr1,otr,otr1/otr)