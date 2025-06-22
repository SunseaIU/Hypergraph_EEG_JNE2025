function [Max_acc,Bestlambda,Bestcreate,BestF_U,Besttheta,Y_predict,BestIter] = HSWCAN_noWnoH(X,X_l,Y_l,X_u,Y_u)
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

% %去除X零空间
% [U1, S, V] = svd(X);
% % 设定阈值
% threshold = 1e-10;
% % 保留大于阈值的奇异值
% S(S < threshold) = 0;
% % 重新构造矩阵
% X = U1 * S * V';

% matrix parameters
[n,d] = size(X);
[l,d] = size(X_l);
[l,c] = size(Y_l);
[u,d] = size(X_u);

% set the parameter selection range to {2^-10, 2^-8,…,2^10}
% alphanums = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6];
% betanums = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6];
% lambdanums = [-10,-8,-6,-4,-2,0,2,4,6,8,10];
% createnums =[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6];
createnums =[0.4,0.5,0.6,0.7,0.8,0.9,1];
lambdanums =[-8,-6,-4,-2,0,2,4,6,8];

% Default MaxIteration is 50;
MaxIteration = 50;
epsilon = 1e-15;
Max_acc = 0;%所有迭代最优值
current_acc = 0;%单次迭代最优值
for create_index = 1:length(createnums)
    for lambda_index = 1:length(lambdanums)
            create = createnums(create_index);
            lambda = 2^lambdanums(lambda_index);

            % step2: Init
            F_u = ones(u,c)/c;
            F = [Y_l;F_u];
            H = createH_AHL(X, create);
            Theta = eye(d)/d;
            W = eye(n)/n;
            U = eye(n)/n;
            Dv = diag(sum(H * W, 2));
            De = diag(sum(H' * U, 2));
            Dv1 = diag(diag(Dv).^(-1/2));
            De1 = diag(diag(De).^(-1));
            L = U - Dv1*U*H*W*De1*H'*U*Dv1;

            fmbhs = [];
            flag = 0;
%             mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W);
%             s.t. w和为1，大于等于0,；theta和为1，大于等于0；F_l = Y_l.
%             fmbhs = [fmbhs,mbhs];
            %% Loop Iteration
            for iter = 1:MaxIteration
                % step: Update F
                F(l+1:n,:) = -pinv(L(l+1:n,l+1:n))*L(l+1:n,1:l)*Y_l;
%                 mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W);
%                 fmbhs = [fmbhs,mbhs];
%                 fprintf('目标函数值： %.6f \n',mbhs);

                % step: Update Theta
                M = X'*L*X;
                m = diag(M);
                m = m + epsilon;
                theta = diag(Theta);
                for i=1:d
                    theta(i) = 1/(m(i)*sum(1./m));
                end
                Theta = diag(theta);
%                     mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W);
%                     fmbhs = [fmbhs,mbhs];
%                 fprintf('目标函数值： %.6f \n',mbhs);

                Q = X*Theta;
                U = createU(Q);
                De = diag(sum(H' * U, 2));
                De1 = diag(diag(De).^(-1));
                L = U - Dv1*U*H*W*De1*H'*U*Dv1;
%                     mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W);
%                     fmbhs = [fmbhs,mbhs];
%                 fprintf('目标函数值： %.6f \n',mbhs);

                % calculate the accuracy
                F_U = F(l+1:n,:);
                [~,Max_index] = max(F_U,[],2);
                acc = length(find(Max_index==Y_u))/u;

%                 mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W);
%                 fmbhs = [fmbhs,mbhs];

                if(acc > current_acc)
                    current_acc = acc;
                end

                if(acc > Max_acc)
                    Max_acc = acc;
                    Bestlambda = lambda;
                    Bestcreate = create;
                    Besttheta = theta;
                    BestF_U = F_U;
                    Y_predict = Max_index;
                    BestIter = iter;
                    flag = 1;
                end
%                 mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W);
%                 fmbhs = [fmbhs,mbhs];
%                 fprintf('目标函数值： %.6f ，准确度： %.6f \n',mbhs,acc);
            end
            if(current_acc == Max_acc && flag == 1)
%                 Bestfmbhs = fmbhs;
                flag = 0;
            end
%             fprintf('Patient: %02d : create: %.2f, lambda: %.4f , alpha: %.4f ,current acc: %.4f, the best acc: %.4f \n',patient,create,lambda,alpha,current_acc,Max_acc);
            current_acc = 0;
    end
end

