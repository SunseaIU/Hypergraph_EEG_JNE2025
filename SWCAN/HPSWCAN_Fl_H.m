function [Max_acc,Bestr,Bestlambda,Bestalpha,Bestbeta,Besteta,Bestcreate,BestF_U,BestT,Besttheta,BestW,Y_predict,BestIter,Bestfmbhs] = HPSWCAN(X,X_l,Y_l,X_u,Y_u)
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
% etanums = [-10,-8,-6,-4,-2,0,2,4,6,8,10];
% createnums =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8];
% rnums = [20,30,40,45,50,60,70];
createnums =[0.4];
lambdanums =[-4];
alphanums = [-4];
betanums = [1];
etanums = [1];
rnums = [50];

% Default MaxIteration is 50;
MaxIteration = 100;
epsilon = 1e-15;
Max_acc = 0;%所有迭代最优值
current_acc = 0;%单次迭代最优值
for r_index = 1:length(rnums)
    for create_index = 1:length(createnums)
        for lambda_index = 1:length(lambdanums)
            for alpha_index = 1:length(alphanums)
                for beta_index = 1:length(betanums)
                    for  eta_index = 1:length(etanums)
                        create = createnums(create_index);
                        lambda = 2^lambdanums(lambda_index);
                        alpha = 2^alphanums(alpha_index);
                        beta = 2^betanums(beta_index);
                        eta = 2^etanums(eta_index);
                        r = rnums(r_index);

                        % step: Init
                        Y = [Y_l;ones(u,c)/c];
                        H = createH_AHL(X, create);
                        Theta = eye(d)/d;
                        W = eye(n)/n;
                        T = rand(d,r);
                        Dv = diag(sum(H * W, 2));
                        De = diag(sum(H, 1));
                        Dv1 = diag(diag(Dv).^(-1/2));
                        De1 = diag(diag(De).^(-1));
                        L = eye(n) - Dv1*H*W*De1*H'*Dv1;

                        fmbhs = [];
                        flag = 0;
%                         mbhs = trace(F'*L*F) + lambda*trace(Theta'*X'*L*X*Theta) + alpha*trace(W'*W) + beta*||F-Y||^2 - eta*trace(T'*Theta*T);
%                         s.t. w和为1，大于0；theta和为1，大于等于0；T'T = I.
%                         fmbhs = [fmbhs,mbhs];
                        %% Loop Iteration
                        for iter = 1:MaxIteration
                            %%

                            % step: Update F
                            F = ((1/beta)*L+eye(n))\Y;
%                             mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W) + beta*norm(F-Y,2)^2 - eta*trace(T'*Theta*T);
%                             fmbhs = [fmbhs,mbhs];
%                             fprintf('目标函数值： %.6f \n',mbhs);
                            %%

                            % step: Update T
                            solve_T = lambda*Theta*X'*L*X*Theta - eta*Theta;
                            [eig_V,eig_E] = eig(solve_T);
                            [~,index] = sort(diag(eig_E));
                            V_sort = eig_V(:,index);
                            T = V_sort(:,1:r);
%                             mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W) + beta*norm(F-Y,2)^2 - eta*trace(T'*Theta*T);
%                             fmbhs = [fmbhs,mbhs];
%                             fprintf('目标函数值： %.6f \n',mbhs);
                            %%

                            % step: Update Theta
                            A = 2*lambda*(X'*L*X).*(T*T');
                            f = -eta*diag(T*T');
                            % ALM
                            [theta,~,~] = SimplexQP_ALM(A,f,1,1.1,1,ones(d,1)/d);
                            theta = theta + 1e-10;
                            Theta = diag(theta);
%                             mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W) + beta*norm(F-Y,2)^2 - eta*trace(T'*Theta*T);
%                             fmbhs = [fmbhs,mbhs];
%                             fprintf('目标函数值： %.6f \n',mbhs);
                            %%

                            % step: Update W
                            Q = X*Theta*T;
                            P = De1*H'*Dv1*(F*F')*Dv1*H;
                            O = De1*H'*Dv1*(Q*Q')*Dv1*H;
                            p = diag(P);
                            o = diag(O);
                            % method 1:
                            w = zeros(n,1);
                            for i = 1:n
                                w(i) = (1/(2*alpha))*(p(i)+lambda*o(i));
                            end
                            w = w/sum(w);
%                             % method 2:
%                             p = p/sum(p);
%                             o = p/sum(o);
%                             w = EProjSimplex_new((1/(2*alpha))*(p + lambda*o));
%                             w = w + epsilon;

                            W = diag(w);

                            Dv = diag(sum(H * W, 2));
                            Dv1 = diag(diag(Dv).^(-1/2));
                            De = diag(sum(H, 1));
                            De1 = diag(diag(De).^(-1));
                            L = eye(n) - Dv1*H*W*De1*H'*Dv1;
%                             mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W) + beta*norm(F-Y,2)^2 - eta*trace(T'*Theta*T);
%                             fmbhs = [fmbhs,mbhs];
%                             fprintf('目标函数值： %.6f \n',mbhs);
                            %%

                            % calculate the accuracy
                            F_U = F(l+1:n,:);
                            [~,Max_index] = max(F_U,[],2);
                            acc = length(find(Max_index==Y_u))/u;

                            mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W) + beta*norm(F-Y,2)^2 - eta*trace(T'*Theta*T);
                            fmbhs = [fmbhs,mbhs];

                            if(acc > current_acc)
                                current_acc = acc;
                            end

                            if(acc > Max_acc)
                                Max_acc = acc;
                                Bestr = r;
                                Bestlambda = lambda;
                                Bestalpha = alpha;
                                Bestbeta = beta;
                                Besteta = eta;
                                BestT = T;
                                Besttheta = theta;
                                Bestcreate = create;
                                BestF_U = F_U;
                                BestW = W;
                                Y_predict = Max_index;
                                BestIter = iter;
                                flag = 1;
                            end
%                             mbhs = trace(F'*L*F) + lambda*trace(Theta*X'*L*X*Theta) + alpha*trace(W'*W) + beta*norm(F-Y,2)^2;
%                             fmbhs = [fmbhs,mbhs];
%                             fprintf('目标函数值： %.6f ，准确度： %.6f \n',mbhs,acc);
                        end
                        if(current_acc == Max_acc && flag == 1)
                            Bestfmbhs = fmbhs;
                            flag = 0;
                        end

%                         fprintf('lambda: %.4f  alpha: %.4f  beta: %.4f ，current acc: %.4f，the best acc: %.4f \n',lambda,alpha,beta,current_acc,Max_acc);
                        current_acc = 0;

                    end
                end
            end
        end
    end
end

