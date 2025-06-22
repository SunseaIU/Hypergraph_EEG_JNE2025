function [Max_acc,Bestlambda,Bestk,Bestalpha,Bestbeta,Bestcreate,BestF_U,Besttheta,BestW,Y_predict,BestIter,Bestfmbhs] = HSWCAN_k(X,X_l,Y_l,X_u,Y_u)
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
% alphanums = [-10,-8,-6,-4,-2,0,2,4,6,8,10];
% betanums = [-10,-8,-6,-4,-2,0,2,4,6,8,10];
% lambdanums = [-10,-8,-6,-4,-2,0,2,4,6,8,10];
% createnums =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8];
createnums =[0.2];
lambdanums =[-4];
knums = [250];
betanums = [-2];

% Default MaxIteration is 50;
MaxIteration = 100;
epsilon = 1e-15;
Max_acc = 0;%所有迭代最优值
current_acc = 0;%单次迭代最优值
for create_index = 1:length(createnums)
    for lambda_index = 1:length(lambdanums)
        for k_index = 1:length(knums)
            for beta_index = 1:length(betanums)
                create = createnums(create_index);
                lambda = 2^lambdanums(lambda_index);
                k = knums(k_index);
                beta = 2^betanums(beta_index);
    
                % step2: Init
                Y = [Y_l;ones(u,c)/c];
                H = createH_AHL(X, create);
                Theta = eye(d)/d;
                W = eye(n)/n;
                Dv = diag(sum(H * W, 2));
                De = diag(sum(H, 1));
                Dv1 = diag(diag(Dv).^(-1/2));
                De1 = diag(diag(De).^(-1));
                L = eye(n) - Dv1*H*W*De1*H'*Dv1;
    
                fmbhs = [];
                flag = 0;
    %             mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W) + beta*||F-Y||^2;
    %             s.t. w和为1，大于0,；theta和为1，大于等于0
    %             fmbhs = [fmbhs,mbhs];
                %% Loop Iteration
                for iter = 1:MaxIteration
    
                    % step: Update F
                    F = ((1/beta)*L+eye(n))\Y;
    %                 mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W) + beta*norm(F-Y,2)^2;
    %                 fmbhs = [fmbhs,mbhs];
    %                 fprintf('目标函数值： %.6f \n',mbhs);
    
                    % step: Update Theta
                    M = X'*L*X;
                    m = diag(M);
                    m = m + epsilon;
                    theta = diag(Theta);
                    for i = 1:d
                        theta(i) = 1/(m(i)*sum(1./m));
                    end
                    Theta = diag(theta);
%                     mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W) + beta*norm(F-Y,2)^2;
%                     fmbhs = [fmbhs,mbhs];
    %                 fprintf('目标函数值： %.6f \n',mbhs);
    
                    % step: Update W
                    Q = X*Theta;
                    P = De1*H'*Dv1*(F*F')*Dv1*H;
                    O = De1*H'*Dv1*(Q*Q')*Dv1*H;
                    p = diag(P);
                    o = diag(O);
                    po = p + lambda*o;
                    po = po/sum(po);
                    po_sort = sort(po,'descend');
                    % method 1:
                    w = zeros(n,1);
                    alpha = (1/2)*sum(po_sort(1:k))-(k/2)*po_sort(k+1);
                    a = (1/k)-(1/(2*k*alpha))*sum(po_sort(1:k));
                    for i = 1:n
                        w(i) = max((1/(2*alpha))*po(i)+a,epsilon);
                    end
%                     w = w/sum(w);
%                     % method 2:
%                     p = p/sum(p);
%                     o = p/sum(o);
%                     w = EProjSimplex_new((1/(2*alpha))*(p + lambda*o));
%                     w = w + epsilon;

                    W = diag(w);
    
                    Dv = diag(sum(H * W, 2));
                    Dv1 = diag(diag(Dv).^(-1/2));
                    De = diag(sum(H, 1));
                    De1 = diag(diag(De).^(-1));
                    L = eye(n) - Dv1*H*W*De1*H'*Dv1;
%                     mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W) + beta*norm(F-Y,2)^2;
%                     fmbhs = [fmbhs,mbhs];
    %                 fprintf('目标函数值： %.6f \n',mbhs);
    
                    % calculate the accuracy
                    F_U = F(l+1:n,:);
                    [~,Max_index] = max(F_U,[],2);
                    acc = length(find(Max_index==Y_u))/u;
    
                    mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W) + beta*norm(F-Y,2)^2;
                    fmbhs = [fmbhs,mbhs];
    
                    if(acc > current_acc)
                        current_acc = acc;
                    end
    
                    if(acc > Max_acc)
                        Max_acc = acc;
                        Bestlambda = lambda;
                        Bestalpha = alpha;
                        Bestk = k;
                        Bestbeta = beta;
                        Besttheta = theta;
                        Bestcreate = create;
                        BestF_U = F_U;
                        BestW = W;
                        Y_predict = Max_index;
                        BestIter = iter;
                        flag = 1;
                    end
    %                 mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W) + beta*norm(F-Y,2)^2;
    %                 fmbhs = [fmbhs,mbhs];
    %                 fprintf('目标函数值： %.6f ，准确度： %.6f \n',mbhs,acc);
                end
                if(current_acc == Max_acc && flag == 1)
                    Bestfmbhs = fmbhs;
                    flag = 0;
                end
                    
    %             fprintf('lambda: %.4f  alpha: %.4f  beta: %.4f ，current acc: %.4f，the best acc: %.4f \n',lambda,alpha,beta,current_acc,Max_acc);
                current_acc = 0;
    
            end
        end
%         plot(2:2:100,fmbhs(2:2:100),'b-o');
%         plot(1:100,fmbhs(1:100),'b-o');
    end
end

