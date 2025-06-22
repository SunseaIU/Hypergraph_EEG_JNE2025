clear ;close all;clc;

tic
t1 = clock;

session = '01';
accuracies1 = zeros(1,10);

for patient = 1:10

    % load data 
    patient_str = sprintf('%02d', patient);
    data_path = ['../data/session',session,'/S',patient_str,'-',session,'.mat'];
    load(data_path);

    cv = cvpartition(Y,'KFold',5);
    fold_accuracies = zeros(1, cv.NumTestSets);
    for fold = 1:cv.NumTestSets
        trainIdx = cv.training(fold);
        testIdx = cv.test(fold);
        X_src = X(trainIdx,:);
        Y_src = Y(trainIdx,:);
        X_tar = X(testIdx,:);
        Y_tar = Y(testIdx,:);
        
        % process data
        % You can select process method by change 'process_param' to 1,2,3,4
        % Default select the first process method
        process_param = 5;
        [X,X_l,Y_l,X_u,Y_u] = process_data(X_src,Y_src,X_tar,Y_tar,process_param);
        
        % main program
        [Max_acc,Bestlambda,Bestalpha,Bestcreate,BestF_U,Besttheta,BestH,BestW,Y_predict,BestIter] = HSWCAN_Fl_H(X,X_l,Y_l,X_u,Y_u);
        
        % save result
        fprintf('Patient %02d , Fold %d : Bestlambda: %.4f , Bestalpha: %.4f , Bestcreate: %.2f , the best acc: %.4f \n',patient,fold,Bestlambda,Bestalpha,Bestcreate,Max_acc);
        result_struct = struct('Bestlambda',Bestlambda,'Bestalpha',Bestalpha,'Bestcreate',Bestcreate,'BestF_U',BestF_U,'Besttheta',Besttheta,'BestH',BestH,'BestW',BestW,'Y_predict',Y_predict,'BestIter',BestIter);
        save_path = ['../result/session',session,'/DHSLF/DHSLF_patient',patient_str,'-',session,'process_',num2str(process_param),'_fold',num2str(fold)];
        save(save_path,'Max_acc','result_struct');
        fold_accuracies(fold) = Max_acc;
    end
    accuracies1(patient) = mean(fold_accuracies);
    fprintf('Patient %02d: mean acc: %.4f\n',patient, accuracies1(patient));
end
fprintf('Overall mean acc: %.4f\n', mean(accuracies1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
session = '02';
accuracies2 = zeros(1,10);

for patient = 1:10

    % load data 
    patient_str = sprintf('%02d', patient);
    data_path = ['../data/session',session,'/S',patient_str,'-',session,'.mat'];
    load(data_path);

    cv = cvpartition(Y,'KFold',5);
    fold_accuracies = zeros(1, cv.NumTestSets);
    for fold = 1:cv.NumTestSets
        trainIdx = cv.training(fold);
        testIdx = cv.test(fold);
        X_src = X(trainIdx,:);
        Y_src = Y(trainIdx,:);
        X_tar = X(testIdx,:);
        Y_tar = Y(testIdx,:);
        
        % process data
        % You can select process method by change 'process_param' to 1,2,3,4
        % Default select the first process method
        process_param = 5;
        [X,X_l,Y_l,X_u,Y_u] = process_data(X_src,Y_src,X_tar,Y_tar,process_param);
        
        % main program
        [Max_acc,Bestlambda,Bestalpha,Bestcreate,BestF_U,Besttheta,BestH,BestW,Y_predict,BestIter] = HSWCAN_Fl_H(X,X_l,Y_l,X_u,Y_u);
        
        % save result
        fprintf('Patient %02d , Fold %d : Bestlambda: %.4f , Bestalpha: %.4f , Bestcreate: %.2f , the best acc: %.4f \n',patient,fold,Bestlambda,Bestalpha,Bestcreate,Max_acc);
        result_struct = struct('Bestlambda',Bestlambda,'Bestalpha',Bestalpha,'Bestcreate',Bestcreate,'BestF_U',BestF_U,'Besttheta',Besttheta,'BestH',BestH,'BestW',BestW,'Y_predict',Y_predict,'BestIter',BestIter);
        save_path = ['../result/session',session,'/DHSLF/DHSLF_patient',patient_str,'-',session,'process_',num2str(process_param),'_fold',num2str(fold)];
        save(save_path,'Max_acc','result_struct');
        fold_accuracies(fold) = Max_acc;
    end
    accuracies2(patient) = mean(fold_accuracies);
    fprintf('Patient %02d: mean acc: %.4f\n',patient, accuracies2(patient));
end
fprintf('Overall mean acc: %.4f\n', mean(accuracies2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t2 = clock;
fprintf('start time:%d.%d.%d, %2d:%2d:%2d\n',t1(1),t1(2),t1(3),t1(4),t1(5),fix(t1(6)));
fprintf('end time:%d.%d.%d, %2d:%2d:%2d\n',t2(1),t2(2),t2(3),t2(4),t2(5),fix(t2(6)));
toc
% disp(['run time:',num2str(toc)]);
% nohup matlab -nodesktop -nosplash -nodisplay < main.m >log.txt 2>&1   &