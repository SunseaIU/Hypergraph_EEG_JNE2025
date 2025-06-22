clear ;close all;clc;
acc15 = zeros(1,15);
alpha15 = zeros(1,15);
beta15 = zeros(1,15);
r15 = zeros(1,15);
for i = 1:15

    patient = i;

    train_path = '../data/s1.mat';

    load(train_path);
    X_src = fea(300*(patient-1)+1:300*patient,:);
    Y_src = label(300*(patient-1)+1:300*patient,:);
    clear fea label;
    test_path = '../data/s2.mat';

    load(test_path);
    X_tar = fea(50*(patient-1)+1:50*patient,:);
    Y_tar = label(50*(patient-1)+1:50*patient,:);


    process_param = 6;
    [X,X_l,Y_l,X_u,Y_u] = process_data(X_src,Y_src,X_tar,Y_tar,process_param);

    % main program
    [Max_acc,Bestr,Bestalpha,Bestbeta,BestF_U,BestM,BestH,BestW,Y_predict,BestIter] = VAHL(X,X_l,Y_l,X_u,Y_u);

    acc15(i) = Max_acc;
    alpha15(i) = Bestalpha;
    beta15(i) = Bestbeta;
    r15(i) = Bestr;
end
save result
x = 1:15;
y = acc15;
bar(x,y);
ylim([0,1]);
xlabel('SUBJECT');
ylabel('ACCURACY');
title('Accuracy of 15 Subjects');
