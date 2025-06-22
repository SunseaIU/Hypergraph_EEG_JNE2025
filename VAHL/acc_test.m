clc;clear;
% 假设数据
%%%%%%%%%%%%% 2020 %%%%%%%%%%%%%%%%
% DHSLF_model_accuracy = [0.76 	0.62 	0.76 	0.78 	0.80 	0.66 	0.80 	0.78 	0.76 	0.66 	0.72 	0.58 	0.74 	0.54 	0.70 
% ];
% DHSLP_model_accuracy = [0.84 	0.70 	0.86 	0.90 	0.88 	0.68 	0.90 	0.84 	0.68 	0.76 	0.78 	0.68 	0.80 	0.64 	0.82 
% ];
% other_model_accuracy = [0.64 	0.54 	0.74 	0.70 	0.84 	0.52 	0.78 	0.68 	0.56 	0.56 	0.70 	0.46 	0.78 	0.48 	0.66 
% ];

%%%%%%%%%%%%% UNL %%%%%%%%%%%%%%%%
DHSLF1_model_accuracy = [0.6125 	0.6250 	0.7750 	0.6625 	0.6000 	0.6375 	0.6375 	0.6375 	0.6375 	0.6500 
];
DHSLF2_model_accuracy = [0.6000 	0.5875 	0.6625 	0.6500 	0.6250 	0.6500 	0.6125 	0.6375 	0.5875 	0.6250 
];
DHSLF3_model_accuracy = [0.7250 	0.6250 	0.7000 	0.5625 	0.6125 	0.6439 	0.6000 	0.7750 	0.6375 	0.5875 
];
DHSLF_model_accuracy = [DHSLF1_model_accuracy   DHSLF2_model_accuracy   DHSLF3_model_accuracy];
DHSLP1_model_accuracy = [0.6500 	0.6500 	0.8750 	0.6250 	0.6500 	0.6500 	0.6375 	0.6500 	0.6000 	0.6125 
];
DHSLP2_model_accuracy = [0.6250 	0.6500 	0.6875 	0.6500 	0.6375 	0.6750 	0.6500 	0.5875 	0.6250 	0.6875 
];
DHSLP3_model_accuracy = [0.8750 	0.6500 	0.6667 	0.6375 	0.6625 	0.7136 	0.6750 	0.7750 	0.6500 	0.6125 
];
DHSLP_model_accuracy = [DHSLP1_model_accuracy   DHSLP2_model_accuracy   DHSLP3_model_accuracy];
other1_model_accuracy = [0.4875 	0.5750 	0.9500 	0.6250 	0.4250 	0.6500 	0.6750 	0.6000 	0.6125 	0.6750 
];
other2_model_accuracy = [0.6375 	0.6750 	0.5625 	0.5000 	0.5375 	0.5000 	0.5375 	0.6250 	0.5875 	0.6750 
];
other3_model_accuracy = [0.8250 	0.7375 	0.5833 	0.4250 	0.5625 	0.7515 	0.7515 	0.7750 	0.6000 	0.6000 
];
other_model_accuracy = [other1_model_accuracy   other2_model_accuracy   other3_model_accuracy];

% DHSLF_model_accuracy = [0.6475 0.6238 0.6469];
% DHSLP_model_accuracy = [0.6600 0.6475 0.6918];
% other_model_accuracy = [0.6275 0.5837 0.6397];

% 配对t检验
[h, p, ci, stats] = ttest(DHSLF_model_accuracy, other_model_accuracy);

% 输出结果
fprintf('DHSLF:t检验结果: h=%d, p=%.20f, ci=[%.4f, %.4f], stats.tstat=%.4f, stats.df=%d\n', ...
    h, p, ci(1), ci(2), stats.tstat, stats.df);

% 配对t检验
[h, p, ci, stats] = ttest(DHSLP_model_accuracy, other_model_accuracy);

% 输出结果
fprintf('DHSLP:t检验结果: h=%d, p=%.20f, ci=[%.4f, %.4f], stats.tstat=%.4f, stats.df=%d\n', ...
    h, p, ci(1), ci(2), stats.tstat, stats.df);

% Wilcoxon符号秩检验
[p, h, stats] = signrank(DHSLF_model_accuracy, other_model_accuracy);

% 输出结果
fprintf('DHSLF:Wilcoxon符号秩检验结果: h=%d, p=%.10f, stats.wstat=%.4f\n', ...
    h, p, stats.signedrank);

% Wilcoxon符号秩检验
[p, h, stats] = signrank(DHSLP_model_accuracy, other_model_accuracy);

% 输出结果
fprintf('DHSLP:Wilcoxon符号秩检验结果: h=%d, p=%.10f, stats.wstat=%.4f\n', ...
    h, p, stats.signedrank);