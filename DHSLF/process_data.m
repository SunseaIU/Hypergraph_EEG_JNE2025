function [X,X_l,Y_l,X_u,Y_u,Y_index] = process_data(X_src,Y_src,X_tar,Y_tar,process_param)
    switch process_param
        case 1
            [X,X_l,Y_l,X_u,Y_u,Y_index] = process_1(X_src,Y_src,X_tar,Y_tar);
        case 2
            [X,X_l,Y_l,X_u,Y_u,Y_index] = process_2(X_src,Y_src,X_tar,Y_tar);
        case 3
            [X,X_l,Y_l,X_u,Y_u,Y_index] = process_3(X_src,Y_src,X_tar,Y_tar);
        case 4
            [X,X_l,Y_l,X_u,Y_u,Y_index] = process_4(X_src,Y_src,X_tar,Y_tar);
        case 5
            [X,X_l,Y_l,X_u,Y_u,Y_index] = process_5(X_src,Y_src,X_tar,Y_tar);
        case 6
            [X,X_l,Y_l,X_u,Y_u,Y_index] = process_6(X_src,Y_src,X_tar,Y_tar);
        case 7
            [X,X_l,Y_l,X_u,Y_u,Y_index] = process_7(X_src,Y_src,X_tar,Y_tar);
        case 61
             [X,X_l,Y_l,X_u,Y_u,Y_index] = process_6_1(X_src,Y_src,X_tar,Y_tar);
        otherwise
            fprintf('The process_param is error!');
    end
end