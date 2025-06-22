clear;

patient = 1;
% load data 
train_path = '../data/s1.mat';
load(train_path);
XL = fea(300*(patient-1)+1:300*patient,:);
YL = label(300*(patient-1)+1:300*patient,:);
clear fea label;
test_path = '../data/s2.mat';
load(test_path);
XU = fea(50*(patient-1)+1:50*patient,:);
YU = label(50*(patient-1)+1:50*patient,:);

% train_path = '../data/s1.mat';
% load(train_path);
% fea = fea(1:300,:);
% label = label(1:300,:);
% totalrows = size(fea,1);
% rng(0);
% randomindices = randperm(totalrows);
% halfpoint = floor(totalrows/5);
% train_indices = randomindices(1:4*halfpoint);
% test_indices = randomindices(4*halfpoint+1:end);
% XL = fea(train_indices,:);
% YL = label(train_indices,:);
% XU = fea(test_indices,:);
% YU = label(test_indices,:);

[~,XL,YL,XU,YU] = process_data(XL,YL,XU,YU,5);
[~,YL] = max(YL,[],2);
% 使用fitcknn训练KNN模型
k = 1; % 选择K值
knnModel = fitcknn(XL, YL, 'NumNeighbors', k);

% 使用predict函数进行预测
predictions = predict(knnModel, XU);

% 显示预测结果

acc = length(find(predictions==YU))/size(YU,1);
disp(acc);