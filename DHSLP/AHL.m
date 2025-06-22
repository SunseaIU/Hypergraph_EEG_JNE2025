function [Max_acc,Bestalpha,Bestbeta,BestF_U,BestM,BestH,BestW,Y_predict,BestIter] = AHL(X,X_l,Y_l,X_u,Y_u)
% =========================================================================
% SPGO implements the Algorithm 1
%
% Input:
% X: n*d is the all data, which include source data and target data.
% X_l: l*d is the labeled data feature
% Y_l: l*c is the labeled data ground truth
% X_u: u*d is the unlabeled data feature
% Y_u: u*c is the unlabeled data ground truth
%
% Output:
% Max_acc: is the accuracy of emotion recognition by use VAHL model.
% Bestalpha: is the value of parameter alpha at maximum accuracy.
% Bestbeta: is the value of parameter beta at maximum accuracy.
% Bestgamma: is the value of parameter gamma at maximum accuracy.
% BestF_U: is the prediction accuracy for each class
% BestS: is the adjacency matrix of adaptive graph
% BestW: is the projection matrix, which can be adjusted by set param 'm'
% Y_predict: is the prediction of unlabeled data
% BestIter: is the value of Iter at maximum accuracy.
%
% =========================================================================

%% Prepare data

%去除X零空间
[U1, S, V] = svd(X);
% 设定阈值
threshold = 1e-10;
% 保留大于阈值的奇异值
S(S < threshold) = 0;
% 重新构造矩阵
X = U1 * S * V';

% matrix parameters
[n,d] = size(X);
[l,d] = size(X_l);
[l,c] = size(Y_l);
[u,d] = size(X_u);

% D_X = getDmatrix(X);
%% SPGO model
% set the parameter selection range to {2^-10, 2^-8,…,2^10}
% alphanums = [-10,-8,-6,-4,-2,0,2,4,6,8,10];
% betanums = [-10,-8,-6,-4,-2,0,2,4,6,8,10];
% gammanums = [-10,-8,-6,-4,-2,0,2,4,6,8,10];
alphanums = [-4];
betanums = [-4];
% gammanums = [2];
% set the feature dimension after projection
% Default is 50, it can be set 10,20,30,50,100,200
r = 45;
% Default MaxIteration is 50;
MaxIteration = 50;
epsilon = 1e-5;
Max_acc = 0;
current_acc = 0;

    for alpha_index = 1:length(alphanums)
        for beta_index = 1:length(betanums)
            alpha = 2^alphanums(alpha_index);
            beta = 2^betanums(beta_index);
            % fprintf('alpha: %.4f  beta: %.4f  gamma: %.4f，the best acc: %.4f \n',alpha,beta,gamma,Max_acc);

            % step2: Init
            F_u = ones(u,c)/c;
            F = [Y_l;F_u];
            H = createH_AHL(X,0.4);
            W = eye(n)/n;
            Dv = diag(sum(H * W, 2));
            De = diag(sum(H, 1));
            Dv1 = diag(diag(Dv).^(-1/2));
            De1 = diag(diag(De).^(-1));
            L = eye(n) - Dv1*H*W*De1*H'*Dv1;
            M = rand(d,r);
%             M = ones(d,r);
%             for i = 1:size(M, 2)
%                 M(:, i) = M(:, i) / norm(M(:, i));
%             end
%             M = eye(r);
%             M = [M;zeros(d-r,r)];

            fmbhs = [];
%             mbhs = trace(F'*L*F) + alpha*trace(M'*X'*L*X*M) + beta*trace(W'*W);
%             fmbhs = [fmbhs,mbhs];
            %% Loop Iteration
            for iter = 1:MaxIteration

                % step4: Update F
                % 待改进
%                 S1 = H(l+1:n,:)*W*De1*H(l+1:n,:)';
%                 S1 = S1-diag(diag(S1));
%                 m = S1*F(l+1:n,:);
%                 for i = l+1:n
%                     F(i,:) = EProjSimplex_new(m(i-l,:));
%                 end
%                 F(l+1:n,:) = -pinv(L(l+1:n,l+1:n))*L(l+1:n,1:l)*Y_l;
%                 % F = (1/lambda * L+eye(n))*Y1;
%                 mbhs = trace(F'*L*F) + alpha*trace(M'*X'*L*X*M) + beta*trace(W'*W);
%                 fmbhs = [fmbhs,mbhs];

                % step5: Update M
                solve_M = pinv(X'*X)*alpha*(X'*L*X);
                [eig_V,eig_E] = eig(solve_M);
                eig_V = (eig_V);
                eig_E = (eig_E);
                [~,index] = sort(diag(eig_E));
                V_sort = eig_V(:,index);
                M = V_sort(:,1:r);
%                 mbhs = trace(F'*L*F) + alpha*trace(M'*X'*L*X*M) + beta*trace(W'*W);
%                 fmbhs = [fmbhs,mbhs];
                
                %Update F
                F(l+1:n,:) = -pinv(L(l+1:n,l+1:n))*L(l+1:n,1:l)*Y_l;
                % F = (1/lambda * L+eye(n))*Y1;
%                 mbhs = trace(F'*L*F) + alpha*trace(M'*X'*L*X*M) + beta*trace(W'*W);
%                 fmbhs = [fmbhs,mbhs];

                % step6: Update H
                X1 = X*M;
                H = createH_AHL(X1,0.4);
                Dv = diag(sum(H * W, 2));
                De = diag(sum(H, 1));
                De1 = diag(diag(De).^(-1));
                Dv1 = diag(diag(Dv).^(-1/2));
%                 mbhs = trace(F'*L*F) + alpha*trace(M'*X'*L*X*M) + beta*trace(W'*W);
%                 fmbhs = [fmbhs,mbhs];

                % step7: Update W
%                 A = pinv(De)*H'*U*Dv1*(F*F'+alpha*X*M*M'*X')*Dv1*U*H;
                %             A = pinv(De)*H'*U*Dv1*(F*F'+alpha*X*X')*Dv1*U*H;

                A = De1*H'*Dv1*X*(M*M')*X'*Dv1*H;
                B = De1*H'*Dv1*(F*F')*Dv1*H;
                a = diag(A);
%                 a = a/sum(a);
                b = diag(B);
%                 b = b/sum(b);
                w = zeros(n,1);
                for i = 1:n
                    w(i) = (1/(2*beta))*(b(i)+alpha*a(i));
                end
                w = w/sum(w);
%                 w = EProjSimplex_new((1/(2*beta))*(b+alpha*a));
%                 w = w + 1e-15;
                W = diag(w);

                Dv = diag(sum(H * W, 2));
                Dv1 = diag(diag(Dv).^(-1/2));
                De = diag(sum(H, 1));
                De1 = diag(diag(De).^(-1));
                L = eye(n) - Dv1*H*W*De1*H'*Dv1;
                mbhs = trace(F'*L*F) + alpha*trace(M'*X'*L*X*M) + beta*trace(W'*W);
                fmbhs = [fmbhs,mbhs];


                % calculate the accuracy
                F_U = F(l+1:n,:);
                [~,Max_index] = max(F_U,[],2);
                acc = length(find(Max_index==Y_u))/u;

                if(acc > current_acc)
                    current_acc = acc;
                end

                if(acc > Max_acc)
                    Max_acc = acc;
                    Bestalpha = alpha;
                    Bestbeta = beta;
                    BestF_U = F_U;
                    BestM = M;
                    BestH = H;
                    BestW = W;
                    Y_predict = Max_index;
                    BestIter = iter;
                end
%                 mbhs = trace(F'*L*F) + alpha*trace(M'*X'*L*X*M) + beta*trace(W'*W);
%                 fmbhs = [fmbhs,mbhs];
                fprintf('目标函数值： %.6f ，准确度： %.6f \n',mbhs,acc);
            end
            fprintf('alpha: %.4f  beta: %.4f ，current acc: %.4f，the best acc: %.4f \n',alpha,beta,current_acc,Max_acc);
            current_acc = 0;

        end
    end
    plot(1:50,fmbhs(1:50),'b-o');
end

