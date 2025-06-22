% 频段标签
frequencies = {'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'};

% 对应的精度值
precision = [26, 31, 25, 39, 71];

% 绘制柱状图
figure; % 创建一个新图形窗口
rgb_color = [0, 121/255, 195/255];
hBar = bar(1:5, precision, 'FaceColor', rgb_color, 'EdgeColor', 'k'); % 绘制柱状图，设置柱子颜色

% 设置横坐标标签
set(gca, 'XTick', 1:5, 'XTickLabel', frequencies); % 设置横坐标刻度位置和标签

% 添加标题和坐标轴标签
xlabel('Frequency', 'FontSize', 12);
ylabel('Accuracy(%)', 'FontSize', 12);

% 添加网格线以便更清晰地查看数值
grid on;

% for i = 1:length(precision)
%     text(hBar.XData(i), precision(i), num2str(precision(i)), ...
%         'HorizontalAlignment', 'center', ...
%         'VerticalAlignment', 'bottom', ...
%         'FontSize', 10, ...
%         'Color', 'k');
% end