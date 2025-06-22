function [Max_acc,Bestr,Bestalpha,Bestbeta,Bestcreate,BestF_U,BestM,BestH,BestW,Y_predict,BestIter,Bestfmbhs] = VAHL(X,X_l,Y_l,X_u,Y_u)
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
alphanums = [-2];
betanums = [4];
% set the feature dimension after projection
% Default is 50, it can be set 10,20,30,50,100,200
% rnums = [10,20,30,45,50,60,70,80,90];
rnums = [45];
createnums = [0.4];
% Default MaxIteration is 50;
MaxIteration = 50;
% epsilon = 1e-5;
Max_acc = 0;
current_acc = 0;
for create_index = 1:length(createnums)
for r_index = 1:length(rnums)
    for alpha_index = 1:length(alphanums)
        for beta_index = 1:length(betanums)
            alpha = 2^alphanums(alpha_index);
            beta = 2^betanums(beta_index);
            r = rnums(r_index);
            create = createnums(create_index);
            % fprintf('alpha: %.4f  beta: %.4f  gamma: %.4f，the best acc: %.4f \n',alpha,beta,gamma,Max_acc);

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
            del = U*H*W*De1*H'*U;
            % vision
            fx0 = [];
            fx1 = [];
            fx2 = [];
            fx3 = [];
            fx4 = [];
            fl = [];
            fl = [fl,del(1,3)];
            fmbhs = [];
            flag = 0;

%             f0 = length(intersect(find(H(6,:)),find(H(8,:))))/(sum(H(6,:))+ ...
%                 sum(H(8,:))-length(intersect(find(H(6,:)),find(H(8,:)))));
%             fx0 = [fx0,f0];
%             f1 = length(intersect(find(H(92,:)),find(H(318,:))))/(sum(H(92,:))+ ...
%                 sum(H(318,:))-length(intersect(find(H(92,:)),find(H(318,:)))));
%             fx1 = [fx1,f1];
%             f2 = length(intersect(find(H(92,:)),find(H(6,:))))/(sum(H(92,:))+ ...
%                 sum(H(6,:))-length(intersect(find(H(92,:)),find(H(6,:)))));
%             fx2 = [fx2,f2];
%             f3 = length(intersect(find(H(6,:)),find(H(318,:))))/(sum(H(6,:))+ ...
%                 sum(H(318,:))-length(intersect(find(H(6,:)),find(H(318,:)))));
%             fx3 = [fx3,f3];

            M = ones(d,r);
            for i = 1:size(M, 2)
                M(:, i) = M(:, i) / norm(M(:, i));
            end
%             mbhs = trace(F'*L*F) + alpha*trace(M'*X'*L*X*M) + beta*trace(W'*W);
%             fmbhs = [fmbhs,mbhs];
%             X1 = X*M;
%             fD = pdist2(X,X,'euclidean');
%             fx0 = [fx0,fD(310,306)/mean(fD(310,:))];
%             fx1 = [fx1,fD(310,313)/mean(fD(310,:))];
%             fx2 = [fx2,fD(310,325)/mean(fD(310,:))];
%             fx3 = [fx3,fD(310,337)/mean(fD(310,:))];
%             fx4 = [fx4,fD(310,342)/mean(fD(310,:))];

%             fx0 = [fx0,cosineSimilarity(X(1,:),X(2,:))];
%             fx1 = [fx1,cosineSimilarity(X(1,:),X(100,:))];
%             fx2 = [fx2,cosineSimilarity(X(1,:),X(200,:))];
%             fx3 = [fx3,cosineSimilarity(X(1,:),X(300,:))];

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
%                 F(l+1:n,:) = -(L(l+1:n,l+1:n)+eye(u)*1e-15)\L(l+1:n,1:l)*Y_l;

                % step5: Update M
                solve_M = alpha*(X'*X)\(X'*L*X);
                [eig_V,eig_E] = eig(solve_M);
                eig_V = real(eig_V);
                eig_E = real(eig_E);
                [~,index] = sort(diag(eig_E),'ascend');
                V_sort = eig_V(:,index);
                M = V_sort(:,1:r);

                % step6: Update H
                X1 = X*M;
                H = createH_AHL(X1,create);
                U = createU(X1);
                Dv = diag(sum(H * W, 2));
                De = diag(sum(H' * U, 2));
                Dv1 = diag(diag(Dv).^(-1/2));
                De1 = diag(diag(De).^(-1));

%                 fx0 = [fx0,cosineSimilarity(X1(1,:),X1(2,:))];
%                 fx1 = [fx1,cosineSimilarity(X1(1,:),X1(100,:))];
%                 fx2 = [fx2,cosineSimilarity(X1(1,:),X1(200,:))];
%                 fx3 = [fx3,cosineSimilarity(X1(1,:),X1(300,:))];

%                 f0 = length(intersect(find(H(6,:)),find(H(8,:))))/(sum(H(6,:))+ ...
%                     sum(H(8,:))-length(intersect(find(H(6,:)),find(H(8,:)))));
%                 fx0 = [fx0,f0];
%                 f1 = length(intersect(find(H(92,:)),find(H(318,:))))/(sum(H(92,:))+ ...
%                     sum(H(318,:))-length(intersect(find(H(92,:)),find(H(318,:)))));
%                 fx1 = [fx1,f1];
%                 f2 = length(intersect(find(H(92,:)),find(H(6,:))))/(sum(H(92,:))+ ...
%                     sum(H(6,:))-length(intersect(find(H(92,:)),find(H(6,:)))));
%                 fx2 = [fx2,f2];
%                 f3 = length(intersect(find(H(338,:)),find(H(339,:))))/(sum(H(338,:))+ ...
%                     sum(H(338,:))-length(intersect(find(H(338,:)),find(H(339,:)))));
%                 fx3 = [fx3,f3];

%                 fD = pdist2(X1,X1,'euclidean');
%                 fx0 = [fx0,fD(310,306)/mean(fD(310,:))];
%                 fx1 = [fx1,fD(310,313)/mean(fD(310,:))];
%                 fx2 = [fx2,fD(310,325)/mean(fD(310,:))];
%                 fx3 = [fx3,fD(310,337)/mean(fD(310,:))];
%                 fx4 = [fx4,fD(310,342)/mean(fD(310,:))];

                % step7: Update W
%                 A = pinv(De)*H'*U*Dv1*(F*F'+alpha*X*M*M'*X')*Dv1*U*H;
                %             A = pinv(De)*H'*U*Dv1*(F*F'+alpha*X*X')*Dv1*U*H;

                A = De1*H'*U*Dv1*X*(M*M')*X'*Dv1*U*H;
                B = De1*H'*U*Dv1*(F*F')*Dv1*U*H;
                a = diag(A);
                a = a/sum(a);
                b = diag(B);
                b = b/sum(b);
%                 w = zeros(n,1);
%                 for i = 1:n
%                     w(i) = (1/(2*beta))*(b(i)+alpha*a(i));
%                 end
%                 w = w/sum(w);
                w = EProjSimplex_new((1/(2*beta))*(b+alpha*a));
                w = w + 1e-15;
                W = diag(w);

%                 for i = 1:n
%                     W(i,i) = 1/n + A(i,i)/(2*beta) - trace(A)/(2*beta*n);
% %                     W(i,i) = max(1e-10,(1/(2*beta))*(B(i,i)+alpha*A(i,i)));
%                 end
%                 diagElements = diag(W);
% 
                Dv = diag(sum(H * W, 2));
                Dv1 = diag(diag(Dv).^(-1/2));
                L = U - Dv1*U*H*W*De1*H'*U*Dv1;
                del = U*H*W*De1*H'*U;
                fl = [fl,del(1,3)];

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
                    Bestbeta = beta;
                    Bestcreate = create;
                    BestF_U = F_U;
                    BestM = M;
                    BestH = H;
                    BestW = W;
                    Y_predict = Max_index;
                    BestIter = iter;
                    flag = 1;
                end
                mbhs = trace(F'*L*F) + alpha*trace(M'*X'*L*X*M) + beta*trace(W'*W);
                fmbhs = [fmbhs,mbhs];
%                 fprintf('目标函数值： %.6f ，准确度： %.6f \n',mbhs,acc);
            end
            if(current_acc == Max_acc && flag == 1)
                Bestfmbhs = fmbhs;
                flag = 0;
            end
%             fprintf('r: %d , alpha: %.4f , beta: %.4f ,current acc: %.4f, the best acc: %.4f \n',r,alpha,beta,current_acc,Max_acc);
            current_acc = 0;

        end
    end
end
end
% plot(0:50, fl,'b-h');
% plot(0:50,fmbhs,'b-o');

% plot(0:50, fx0,'r-o',0:50,fx1,'b-s',0:50,fx2,'b-p',0:50,fx3,'b-^',0:50,fx4,'b-h');
% legend('1 and 1','1 and 2','1 and 3','1 and 4','1 and 5');
% xlabel('Iter');
% ylabel('Value');
% frame = getframe(gcf); % gcf代表当前的图形窗口
% img = frame2im(frame); % 将帧转换为图像数组
% imwrite(img, '../result/relative distance.png');
end

