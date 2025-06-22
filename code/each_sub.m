% 定义4组精度数据，每组对应15个被试
% 每行代表一组精度数据
DHSLF2020 = [0.76 	0.62 	0.76 	0.78 	0.80 	0.66 	0.80 	0.78 	0.76 	0.66 	0.72 	0.58 	0.74 	0.54 	0.70 
]; % 示例数据1，替换为实际数据
DHSLP2020 = [0.84 	0.70 	0.86 	0.90 	0.88 	0.68 	0.90 	0.84 	0.68 	0.76 	0.78 	0.68 	0.80 	0.64 	0.82 
]; % 示例数据2，替换为实际数据
DHSLFlky1 = [0.6125 	0.6250 	0.7750 	0.6625 	0.6000 	0.6375 	0.6375 	0.6375 	0.6375 	0.6500 
];
DHSLFlky2 = [0.6000 	0.5875 	0.6625 	0.6500 	0.6250 	0.6500 	0.6125 	0.6375 	0.5875 	0.6250 
];
DHSLFlky3 = [0.7250 	0.6250 	0.7000 	0.5625 	0.6125 	0.6439 	0.6000 	0.7750 	0.6375 	0.5875 
];
DHSLFlky = (DHSLFlky1 + DHSLFlky2 + DHSLFlky3)/3;

DHSLPlky1 = [0.6500 	0.6500 	0.8750 	0.6250 	0.6500 	0.6500 	0.6375 	0.6500 	0.6000 	0.6125 
];
DHSLPlky2 = [0.6250 	0.6500 	0.6875 	0.6500 	0.6375 	0.6750 	0.6500 	0.5875 	0.6250 	0.6875 
];
DHSLPlky3 = [0.8750 	0.6500 	0.6667 	0.6375 	0.6625 	0.7136 	0.6750 	0.7750 	0.6500 	0.6125 
];
DHSLPlky = (DHSLPlky1 + DHSLPlky2 + DHSLPlky3)/3;


% 定义被试编号
subjects = 1:15;
subjectLabels = arrayfun(@num2str, subjects, 'UniformOutput', false); % 将被试编号转换为字符串数组

% 创建柱状图
figure; % 创建一个新图形窗口
set(gcf, 'Position', [100, 100, 650, 400]); % 设置图形窗口大小为800x600像素

barWidth = 0.4; % 设置柱子的宽度
bar1 = bar(subjects - barWidth/2, DHSLF2020, barWidth, 'FaceColor', [60/255 190/255 254/255]); % 第一组数据
hold on; % 保持当前图形，以便添加第二组数据
bar2 = bar(subjects + barWidth/2, DHSLP2020, barWidth, 'FaceColor', [95/255 238/255 149/255]); % 第二组数据
hold off; % 释放图形

% 设置横坐标标签
set(gca, 'XTick', subjects, 'XTickLabel', subjectLabels); % 设置横坐标刻度位置和标签

% 添加图例
legend([bar1, bar2], 'DHSLF', 'DHSLP');

% 添加标题和坐标轴标签
xlabel('Subject ID');
ylabel('Accuracy(%)');
ylim([0 1]);

% 添加网格线以便更清晰地查看数值
grid on;

subjects = 1:10;
subjectLabels = arrayfun(@num2str, subjects, 'UniformOutput', false); % 将被试编号转换为字符串数组
% 创建柱状图
figure; % 创建一个新图形窗口
set(gcf, 'Position', [200, 200, 650, 400]); % 设置图形窗口大小为800x600像素

barWidth = 0.4; % 设置柱子的宽度
bar1 = bar(subjects - barWidth/2, DHSLFlky, barWidth, 'FaceColor', [60/255 190/255 254/255]); % 第一组数据
hold on; % 保持当前图形，以便添加第二组数据
bar2 = bar(subjects + barWidth/2, DHSLPlky, barWidth, 'FaceColor', [95/255 238/255 149/255]); % 第二组数据
hold off; % 释放图形

% 设置横坐标标签
set(gca, 'XTick', subjects, 'XTickLabel', subjectLabels); % 设置横坐标刻度位置和标签

% 添加图例
legend([bar1, bar2], 'DHSLF', 'DHSLP');

% 添加标题和坐标轴标签
xlabel('Subject ID');
ylabel('Accuracy(%)');
ylim([0 1]);

% 添加网格线以便更清晰地查看数值
grid on;