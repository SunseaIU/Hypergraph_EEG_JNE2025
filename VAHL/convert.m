function YL = convert(Y)
%UNTITLED 此处显示有关此函数的摘要
%   输入n*c，输出n*1
YL=zeros(size(Y,1),1);
for i = 1:size(Y,1)
    index = find(Y(i,:));
    YL(i,1)=index;
end
end
