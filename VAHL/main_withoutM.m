clear ;close all;clc;

% % load data 
% train_path = '../data/s1.mat';
% % train_path = '../data/iris1.mat';
% load(train_path);
% X_src = fea;
% Y_src = label;
% clear fea label;
% test_path = '../data/s2.mat';
% % test_path = '../data/iris2.mat';
% load(test_path);
% X_tar = fea;
% Y_tar = label;

train_path = '../data/s2.mat';
% train_path = '../data/s1_session1.mat';
load(train_path);
totalrows = size(fea,1);
rng(0);
randomindices = randperm(totalrows);
halfpoint = floor(totalrows/5);
train_indices = randomindices(1:4*halfpoint);
test_indices = randomindices(4*halfpoint+1:end);
% train_indices = [1:halfpoint];
% test_indices = [halfpoint+1:totalrows];
X_src = fea(train_indices,:);
Y_src = label(train_indices,:);
X_tar = fea(test_indices,:);
Y_tar = label(test_indices,:);

% process data
% You can select process method by change 'process_param' to 1,2,3,4
% Default select the first process method
process_param = 5;
[X,X_l,Y_l,X_u,Y_u] = process_data(X_src,Y_src,X_tar,Y_tar,process_param);

% main program
[Max_acc,Bestalpha,Bestbeta,BestF_U,BestH,BestW,Y_predict,BestIter] = VAHL(X,X_l,Y_l,X_u,Y_u);

% save result
fprintf('Bestalpha: %.4f , Bestbeta: %.4f , the best acc: %.4f \n',Bestalpha,Bestbeta,Max_acc);
result_struct = struct('Bestalpha',Bestalpha,'Bestbeta',Bestbeta,'BestF_U',BestF_U,'BestH',BestH,'BestW',BestW,'Y_predict',Y_predict,'BestIter',BestIter);
save_path = ['../result/process_',num2str(process_param)];
save(save_path,'Max_acc','result_struct')
% nohup matlab -nodesktop -nosplash -nodisplay < main.m >log.txt 2>&1   &
