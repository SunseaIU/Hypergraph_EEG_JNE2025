clear ;close all;clc;

tic
t1 = clock;

accuracies = zeros(1,15);
for i = 1:15
    %单独被试
    patient = i;
    % load data 
    train_path = '../data/s1_tp_all.mat';
    % train_path = '../data/iris1.mat';
    load(train_path);
    % X_src = fea;
    % Y_src = label;
    X_src = fea(300*(patient-1)+1:300*patient,:);
    Y_src = label(300*(patient-1)+1:300*patient,:);
    clear fea label;
    test_path = '../data/s2_tp_all.mat';
    % test_path = '../data/iris2.mat';
    load(test_path);
    % X_tar = fea;
    % Y_tar = label;
    X_tar = fea(50*(patient-1)+1:50*patient,:);
    Y_tar = label(50*(patient-1)+1:50*patient,:);
    
    % process data
    % You can select process method by change 'process_param' to 1,2,3,4
    % Default select the first process method
    process_param = 61;
    [X,X_l,Y_l,X_u,Y_u,Y_index] = process_data(X_src,Y_src,X_tar,Y_tar,process_param);
    
    % main program
    [Max_acc,Bestr,Bestalpha,Bestbeta,Bestcreate,BestF_U,BestM,BestH,BestW,Y_predict,BestIter,AB] = DHSLP_ab(X,X_l,Y_l,X_u,Y_u);
    
    % save result
    result_struct = struct('Bestr',Bestr,'Bestalpha',Bestalpha,'Bestbeta',Bestbeta,'Bestcreate',Bestcreate,'BestF_U',BestF_U,'BestM',BestM,'BestH',BestH,'BestW',BestW,'Y_predict',Y_predict,'BestIter',BestIter,'AB',AB);
    save_path = ['../result/all channels/DHSLP_ab/DHSLP_ab_patient',num2str(patient),'_process',num2str(process_param)];
    save(save_path,'Max_acc','result_struct');

    accuracies(patient) = Max_acc;
    fprintf('Patient %02d: , acc: %.4f\n',patient, accuracies(patient));
end
fprintf('Overall mean acc: %.4f\n', mean(accuracies));
t2 = clock;
fprintf('start time:%d.%d.%d, %2d:%2d:%2d\n',t1(1),t1(2),t1(3),t1(4),t1(5),fix(t1(6)));
fprintf('end time:%d.%d.%d, %2d:%2d:%2d\n',t2(1),t2(2),t2(3),t2(4),t2(5),fix(t2(6)));
toc
% disp(['run time:',num2str(toc)]);
% nohup matlab -nodesktop -nosplash -nodisplay < main.m >log.txt 2>&1   &