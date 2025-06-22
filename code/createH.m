function H = createH(X)
    k = 50;
    distances = pdist2(X, X, 'euclidean');
    
    % 初始化超图关联矩阵H
    H = zeros(size(X, 1), size(X, 1));
    
    % 对于每个样本点，找到k个最近邻
    for i = 1:size(X, 1)
        [~, idx] = sort(distances(i, :), 'ascend');
        nearest_neighbors = idx(2:k+1); % 排除自身，取下一个起的k个最近邻
        
        % 在关联矩阵H中标记最近邻
        H(i,i) = 1;
        H(nearest_neighbors, i) = 1;
    end
    
end

