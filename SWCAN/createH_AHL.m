function H = createH_AHL(X, create)
    distances = pdist2(X, X, 'euclidean');
   
    % 初始化超图关联矩阵H
    H = zeros(size(X, 1), size(X, 1));
    d_mean = zeros(size(X,1),1);

    for i = 1:size(X, 1)
        d_mean(i) = sum(distances(i,:))/(size(X,1) - 1);
        nearest_neighbors = (distances(i,:) <= create*d_mean(i));
        % 在关联矩阵H中标记最近邻
        H(nearest_neighbors, i) = 1;
    end
end

