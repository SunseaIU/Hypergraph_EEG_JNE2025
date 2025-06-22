clear ;close all;clc;

tic
t1 = clock;

patient = 7;
% load data 
train_path = '../data/s1_tp_all.mat';
% train_path = '../data/iris1.mat';
load(train_path);
X_src = fea(300*(patient-1)+1:300*patient,:);
Y_src = label(300*(patient-1)+1:300*patient,:);
clear fea label;
test_path = '../data/s2_tp_all.mat';
% test_path = '../data/iris2.mat';
load(test_path);
X_tar = fea(50*(patient-1)+1:50*patient,:);
Y_tar = label(50*(patient-1)+1:50*patient,:);

% train_path = '../data/s1.mat';
% % train_path = '../data/s1_session1.mat';
% load(train_path);
% fea = fea(1:300,:);
% label = label(1:300,:);
% totalrows = size(fea,1);
% rng(0);
% randomindices = randperm(totalrows);
% halfpoint = floor(totalrows/6);
% train_indices = randomindices(1:5*halfpoint);
% test_indices = randomindices(5*halfpoint+1:end);
% % train_indices = [1:halfpoint];
% % test_indices = [halfpoint+1:totalrows];
% X_src = fea(train_indices,:);
% Y_src = label(train_indices,:);
% X_tar = fea(test_indices,:);
% Y_tar = label(test_indices,:);

% process data
% You can select process method by change 'process_param' to 1,2,3,4
% Default select the first process method
process_param = 61;
[X,X_l,Y_l,X_u,Y_u] = process_data(X_src,Y_src,X_tar,Y_tar,process_param);

% main program
fprintf('\n');
[Max_acc,Bestr,Bestalpha,Bestbeta,Bestcreate,BestF_U,BestM,BestH,BestW,Y_predict,BestIter,Bestfmbhs] = DHSLP_visual(X,X_l,Y_l,X_u,Y_u);

% save result
fprintf('Bestr: %d , Bestalpha: %.4f , Bestbeta: %.4f , Bestcreate :%.2f , the best acc: %.4f \n',Bestr,Bestalpha,Bestbeta,Bestcreate,Max_acc);
result_struct = struct('Bestr',Bestr,'Bestalpha',Bestalpha,'Bestbeta',Bestbeta,'Bestcreate',Bestcreate,'BestF_U',BestF_U,'BestM',BestM,'BestH',BestH,'BestW',BestW,'Y_predict',Y_predict,'BestIter',BestIter,'function_value',Bestfmbhs);
save_path = ['../result/process_',num2str(process_param)];
% figure;
% plot(1:length(Bestfmbhs),Bestfmbhs(1:end),'b-o');
% save(save_path,'Max_acc','result_struct')

t2 = clock;
fprintf('start time:%d.%d.%d, %02d:%02d:%02d\n',t1(1),t1(2),t1(3),t1(4),t1(5),fix(t1(6)));
fprintf('end time:%d.%d.%d, %02d:%02d:%02d\n',t2(1),t2(2),t2(3),t2(4),t2(5),fix(t2(6)));
toc
% nohup matlab -nodesktop -nosplash -nodisplay < main.m >log.txt 2>&1 &
