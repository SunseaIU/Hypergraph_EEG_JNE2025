clear ;close all;clc;

tic
t1 = clock;

% %全体被试
% % load data 
% train_path = '../../VAHL/sub1_session1_session2/data/s1.mat';
% load(train_path);
% X_src = fea;
% Y_src = label;
% clear fea label;
% test_path = '../../VAHL/sub1_session1_session2/data/s2.mat';
% load(test_path);
% X_tar = fea;
% Y_tar = label;
for i = 1
%单独被试
patient = i;
% load data 
train_path = '../../VAHL/sub1_session1_session2/data/s1.mat';
% train_path = '../data/iris1.mat';
load(train_path);
% X_src = fea;
% Y_src = label;
X_src = fea(300*(patient-1)+1:300*patient,:);
Y_src = label(300*(patient-1)+1:300*patient,:);
clear fea label;
test_path = '../../VAHL/sub1_session1_session2/data/s2.mat';
% test_path = '../data/iris2.mat';
load(test_path);
% X_tar = fea;
% Y_tar = label;
X_tar = fea(50*(patient-1)+1:50*patient,:);
Y_tar = label(50*(patient-1)+1:50*patient,:);

% process data
% You can select process method by change 'process_param' to 1,2,3,4
% Default select the first process method
process_param = 5;
[X,X_l,Y_l,X_u,Y_u] = process_data(X_src,Y_src,X_tar,Y_tar,process_param);

% main program
[Max_acc,Bestlambda,Bestalpha,Bestbeta,Bestcreate,BestF_U,Besttheta,BestW,Y_predict,BestIter,Bestfmbhs] = HSWCAN(X,X_l,Y_l,X_u,Y_u);

% save result
fprintf('Bestlambda: %.4f , Bestalpha: %.4f , Bestbeta: %.4f , Bestcreate: %.2f , the best acc: %.4f \n',Bestlambda,Bestalpha,Bestbeta,Bestcreate,Max_acc);
result_struct = struct('Bestlambda',Bestlambda,'Bestalpha',Bestalpha,'Bestbeta',Bestbeta,'Bestcreate',Bestcreate,'BestF_U',BestF_U,'Besttheta',Besttheta,'BestW',BestW,'Y_predict',Y_predict,'BestIter',BestIter,'function_value',Bestfmbhs);
save_path = ['../result/HSWCAN_create_patient',num2str(patient),'_process',num2str(process_param)];
figure;
% plot(1:length(Bestfmbhs),Bestfmbhs(1:end),'b-o');
% plot(1:2:length(Bestfmbhs),Bestfmbhs(1:2:end),'b-o');
plot(2:2:length(Bestfmbhs),Bestfmbhs(2:2:end),'b-o');
% save(save_path,'Max_acc','result_struct')
end
t2 = clock;
fprintf('start time:%d.%d.%d, %2d:%2d:%2d\n',t1(1),t1(2),t1(3),t1(4),t1(5),fix(t1(6)));
fprintf('end time:%d.%d.%d, %2d:%2d:%2d\n',t2(1),t2(2),t2(3),t2(4),t2(5),fix(t2(6)));
toc
% disp(['run time:',num2str(toc)]);
% nohup matlab -nodesktop -nosplash -nodisplay < main.m >log.txt 2>&1   &