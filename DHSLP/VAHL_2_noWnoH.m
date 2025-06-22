function [Max_acc,Bestr,Bestalpha,Bestcreate,BestF_U,BestM,Y_predict,BestIter] = VAHL_2_noWnoH(X,X_l,Y_l,X_u,Y_u)
% =========================================================================
% SPGO implements the Algorithm 1
% change M'XX'M = I to M'M = I;
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

% %去除X零空间
% [U1, S, V] = svd(X);
% % 设定阈值
% threshold = 1e-10;
% % 保留大于阈值的奇异值
% S(S < threshold) = 0;
% % 重新构造矩阵
% X = U1 * S * V';

[n,d] = size(X);
[l,d] = size(X_l);
[l,c] = size(Y_l);
[u,d] = size(X_u);

% D_X = getDmatrix(X);
%% SPGO model
% set the parameter selection range to {2^-10, 2^-8,…,2^10}
% alphanums = [-10,-8,-6,-4,-2,0,2,4,6,8,10];
% betanums = [-10,-8,-6,-4,-2,0,2,4,6,8,10];
% createnums =[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6];
alphanums = [-6,-4,-2,0,2,4,6];
% set the feature dimension after projection
% Default is 50, it can be set 10,20,30,50,100,200
% rnums = [10,20,30,45,50,60,70,80,90];
rnums = [10,15,20,25,30,35];
createnums = [0.5,0.6,0.7,0.8,0.9,1];
% Default MaxIteration is 50;
MaxIteration = 50;
% epsilon = 1e-5;
Max_acc = 0;
current_acc = 0;
for create_index = 1:length(createnums)
    for r_index = 1:length(rnums)
        for alpha_index = 1:length(alphanums)
            alpha = 2^alphanums(alpha_index);
            r = rnums(r_index);
            create = createnums(create_index);

            % step2: Init
            F_u = ones(u,c)/c;
            F = [Y_l;F_u];
            H = createH_AHL(X,create);
            W = eye(n)/n;
            U = createU(X);
            Dv = diag(sum(H * W, 2));
            De = diag(sum(H' * U, 2));
            Dv1 = diag(diag(Dv).^(-1/2));
            De1 = diag(diag(De).^(-1));
            L = U - Dv1*U*H*W*De1*H'*U*Dv1;

            fmbhs = [];
            flag = 0;

            %% Loop Iteration
            for iter = 1:MaxIteration

                % step4: Update F
                F(l+1:n,:) = -pinv(L(l+1:n,l+1:n))*L(l+1:n,1:l)*Y_l;
                %                     F(l+1:n,:) = -(L(l+1:n,l+1:n)+eye(u)*1e-15)\L(l+1:n,1:l)*Y_l;

                % step5: Update M
                solve_M = alpha*(X'*L*X);
                [eig_V,eig_E] = eig(solve_M);
                eig_V = real(eig_V);
                eig_E = real(eig_E);
                [~,index] = sort(diag(eig_E),'ascend');
                V_sort = eig_V(:,index);
                M = V_sort(:,1:r);

                X1 = X*M;
                U = createU(X1);
                De = diag(sum(H' * U, 2));
                De1 = diag(diag(De).^(-1));
                L = U - Dv1*U*H*W*De1*H'*U*Dv1;

                % calculate the accuracy
                F_U = F(l+1:n,:);
                [~,Max_index] = max(F_U,[],2);
                acc = length(find(Max_index==Y_u))/u;

                if(acc > current_acc)
                    current_acc = acc;
                end

                if(acc > Max_acc)
                    Max_acc = acc;
                    Bestr = r;
                    Bestalpha = alpha;
                    Bestcreate = create;
                    BestF_U = F_U;
                    BestM = M;
                    Y_predict = Max_index;
                    BestIter = iter;
                    flag = 1;
                end
                % mbhs = trace(F'*L*F) + alpha*trace(M'*X'*L*X*M) % + beta*trace(W'*W);
                % fmbhs = [fmbhs,mbhs];
                % fprintf('目标函数值： %.6f ，准确度： %.6f \n',mbhs,acc);
            end
            if(current_acc == Max_acc && flag == 1)
                % Bestfmbhs = fmbhs;
                flag = 0;
            end
            % fprintf('create: %.2f , r: %d, alpha: %.4f , beta: %.4f ,current acc: %.4f, the best acc: %.4f \n',create,r,alpha,beta,current_acc,Max_acc);
            current_acc = 0;

        end
    end
end
end

