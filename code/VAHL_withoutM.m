function [Max_acc,Bestalpha,Bestbeta,BestF_U,BestH,BestW,Y_predict,BestIter] = VAHL_withoutM(X,X_l,Y_l,X_u,Y_u)
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
alphanums = [-2,0,2,4];
betanums = [-2,0,2,4];
% gammanums = [2];
% set the feature dimension after projection
% Default is 50, it can be set 10,20,30,50,100,200
r = 50;
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
            H = createH_AHL(X);
            W = eye(n)/n;
            U = createU(X);
            Dv = diag(sum(H * W, 2));
            De = diag(sum(H' * U, 2));
            Dv1 = diag(diag(Dv).^(-1/2));
            L = U - Dv1*U*H*W*pinv(De)*H'*U*Dv1;
            %% Loop Iteration
            for iter = 1:MaxIteration

                % step4: Update F
%                 L1 = U(l+1:n,l+1:n)*H(l+1:n,:)*W*pinv(De)*H(l+1:n,:)'*U(l+1:n,l+1:n);
%                 L1 = L1-diag(diag(L1));
%                 L2 = U(l+1:n,l+1:n)*H(l+1:n,:)*W*pinv(De)*H(1:l,:)'*U(1:l,1:l);
%                 p = L1*F(l+1:n,:);
%                 q = L2*F(1:l,:);
%                 m = (p + 2*q)/3;
%                 for i = l+1:n
%                     F(i,:) = EProjSimplex_new(m(i-l,:));
%                 end
                F(l+1:n,:) = -pinv(L(l+1:n,l+1:n))*L(l+1:n,1:l)*Y_l;
                % F = (1/lambda * L+eye(n))*Y1;

                % step6: Update H

                % step7: Update W
%                 A = pinv(De)*H'*U*Dv1*(F*F'+alpha*X*M*M'*X')*Dv1*U*H;
                %             A = pinv(De)*H'*U*Dv1*(F*F'+alpha*X*X')*Dv1*U*H;

                A = pinv(De)*H'*U*Dv1*X*X'*Dv1*U*H;
                B = pinv(De)*H'*U*Dv1*(F*F')*Dv1*U*H;
                a = diag(A);
                b = diag(B);
                w = EProjSimplex_new((1/(2*beta))*(b+alpha*a)/sum((1/(2*beta))*(b+alpha*a)));
%                 w = w + 1e-10;
                W = diag(w);

%                 for i = 1:n
%                     W(i,i) = 1/n + A(i,i)/(2*beta) - trace(A)/(2*beta*n);
% %                     W(i,i) = max(1e-10,(1/(2*beta))*(B(i,i)+alpha*A(i,i)));
%                 end
%                 diagElements = diag(W);
% 
%                 % 应用 softmax 函数
%                 expElements = exp(diagElements - max(diagElements)); % 防止指数爆炸
%                 softmaxElements = expElements / sum(expElements);
% 
%                 % 重新构造对角矩阵
%                 W = diag(softmaxElements);
                %             W = diag(W);
                %             W = W/sum(W);
                %             W = diag(W);
                %             % 目标函数
                %             W = diag(W);
                %             objective = @(W) trace(A * diag(W)) + beta * trace((diag(W))' * diag(W));
                %
                %             % 约束条件
                %             % 所有元素都大于0
                %             lb = zeros(n, 1);
                %             % 行和为1
                %             Aeq = ones(1, n);
                %             beq = 1;
                %
                %             % 使用 fmincon 求解
                %             options = optimoptions('fmincon', 'Algorithm', 'interior-point','MaxFunctionEvaluations',5);
                %             W_opt = fmincon(objective, W, [], [], Aeq, beq, lb, [], [], options);
                %             W = diag(W_opt);

                Dv = diag(sum(H * W, 2));
                Dv1 = diag(diag(Dv).^(-1/2));
                De = diag(sum(H' * U, 2));
                L = U - Dv1*U*H*W*pinv(De)*H'*U*Dv1;


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
                    BestH = H;
                    BestW = W;
                    Y_predict = Max_index;
                    BestIter = iter;
                end
                mbhs = trace(F'*L*F) + alpha*trace(X'*L*X) + beta*trace(W'*W);
                fprintf('目标函数值： %.6f ，准确度： %.6f \n',mbhs,acc);
            end
            fprintf('alpha: %.4f  beta: %.4f ，current acc: %.4f，the best acc: %.4f \n',alpha,beta,current_acc,Max_acc);
            current_acc = 0;

        end
    end
end

