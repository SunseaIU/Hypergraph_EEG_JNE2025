clear;

train_path = '../data/s1_tp_all.mat';
load(train_path);
X = fea;
Y = label;

m = size(X,2);
pearson_corr = zeros(1,m);

for i = 1:m
    pearson_corr(i) = corr(X(:, i), Y, 'type', 'Pearson');
end
pearson_corr(isnan(pearson_corr)) = 0;
pearson_corr = abs(pearson_corr);
feature_importance = zeros(1,12);
for i = 1:12
    indices = i:12:(i + 12*63);
    feature_importance(i) = sum(pearson_corr(indices));
end
feature_importance = feature_importance/sum(feature_importance);
bar(1:12,feature_importance)
ylim([0 0.2])
for i = 1:12
    text(i, feature_importance(i), num2str(feature_importance(i),'%.3f'), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end
% [ranks, weights] = relieff(X, Y, 10);
% weights(isnan(weights)) = 0;
% weights = abs(weights);
% feature_importance = zeros(12,1);
% for i = 1:12
%     indices = i:12:(i + 12*44);
%     feature_importance(i) = sum(weights(indices));
% end