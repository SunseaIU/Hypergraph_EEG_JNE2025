function U = createU(X)
    % 计算距离矩阵
    distances = pdist2(X, X, 'euclidean');
    
    % 计算每个样本点到所有样本点的距离的均值
    d_mean = mean(distances, 2);
    
    % 计算所有样本点的d值的总和
    d_sum = sum(d_mean);
    
    % 计算每个样本点的权重
    weights = d_mean / d_sum;
    U = diag(weights);
end

