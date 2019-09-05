tic;
clear all
close all
format compact 
format long
%% 1.数据加载
fprintf(1,'加载数据 \n');
load('drivFace600');%其中1-173为1类，174-343为2类 344-510为3类 511-600为4类，各选择20%作为测试集
%第一类173组
[i1 i2]=sort(rand(173,1)); 
train(1:139,:)=input(i2(1:139),:);     train_label(1:139,1)=output(i2(1:139),1);
test(1:34,:)=input(i2(140:173),:);     test_label(1:34,1)=output(i2(140:173),1);
%第二类有170组
[i1 i2]=sort(rand(170,1));
train(140:275,:)=input(173+i2(1:136),:);    train_label(140:275,1)=output(173+i2(1:136),1);
test(35:68,:)=input(173+i2(137:170),:);     test_label(35:68,1)=output(173+i2(137:170),1);
%第三类有167
[i1 i2]=sort(rand(167,1));
train(276:408,:)=input(343+i2(1:133),:);    train_label(276:408,1)=output(343+i2(1:133),1);
test(69:102,:)=input(343+i2(134:167),:);     test_label(69:102,1)=output(343+i2(134:167),1);
%第4类有90
[i1 i2]=sort(rand(90,1));
train(409:480,:)=input(510+i2(1:72),:);    train_label(409:480,1)=output(510+i2(1:72),1);
test(103:120,:)=input(510+i2(73:90),:);     test_label(103:120,1)=output(510+i2(73:90),1); 
clear i1 i2 input output
%%打乱顺序
k=rand(480,1);[m n]=sort(k);
train=train(n(1:480),:);train_label=train_label(n(1:480),:);
k=rand(120,1);[m n]=sort(k);
test=test(n(1:120),:);test_label=test_label(n(1:120),:);
clear k m n

%no_dims = round(intrinsic_dim(train, 'MLE')); %round四舍五入
%disp(['MLE estimate of intrinsic dimensionality: ' num2str(no_dims)]);
numcases=48;%每块数据集的样本个数
numdims=size(train,2);%单个样本的大小
numbatches=10;  %%原则上每块的样本个数要大于分块数

% 训练数据
x=train;%将数据转换成DBN的数据格式
for i=1:numbatches
    train1=x((i-1)*numcases+1:i*numcases,:);
    batchdata(:,:,i)=train1;
end%将分好的10组数据都放在batchdata中

% rbm参数
maxepoch=20;%训练rbm的次数
numhid=500; numpen=200; numpen2=100;%dbn隐含层的节点数
disp('构建一个3层的置信网络');
clear i 
%% 2.训练RBM
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);%256-200
restart=1;
rbm;%使用cd-k训练rbm，注意此rbm的可视层不是二值的，而隐含层是二值的
vishid1=vishid;hidrecbiases=hidbiases;


fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);%200-100
batchdata=batchposhidprobs;%将第一个RBM的隐含层的输出作为第二个RBM 的输入
numhid=numpen;%将numpen的值赋给numhid，作为第二个rbm隐含层的节点数
restart=1;
rbm;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numpen,numpen2);%200-100

batchdata=batchposhidprobs;%显然，将第二哥RBM的输出作为第三个RBM的输入
numhid=numpen2;%第三个隐含层的节点数
restart=1;
rbm;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;


%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
w1=[vishid1; hidrecbiases]; 
w2=[hidpen; penrecbiases]; 
w3=[hidpen2; penrecbiases2];%%构建好的网络的参数
digitdata = [x ones(size(x,1),1)];
w1probs = 1./(1 + exp(-digitdata*w1));%
  w1probs = [w1probs  ones(size(x,1),1)];%
w2probs = 1./(1 + exp(-w1probs*w2));%
  w2probs = [w2probs ones(size(x,1),1)];%
w3probs = 1./(1 + exp(-w2probs*w3)); %

H_dbn = w3probs;  %%第三个rbm的实际输出值，也是elm的隐含层输出值H

%% 交叉验证
indices = crossvalind('Kfold',size(H_dbn,1),10);%对训练数据进行10折编码
%[Train, Test] = crossvalind('HoldOut', N, P) % 将原始数据随机分为两组,一组做为训练集,一组做为验证集
%[Train, Test] = crossvalind('LeaveMOut', N, M) %留M法交叉验证，默认M为1，留一法交叉验证
sum_accuracy = 0;
for i = 1:10
    %%
    cross_test = (indices == i); %每次循选取一个fold作为测试集
    cross_train = ~cross_test;   %取corss_test的补集作为训练集，即剩下9个fold
    %%
    P_train = H_dbn(cross_train,:)';
    P_test= H_dbn(cross_test,:)';
    T_train= train_label(cross_train,:)';
    T_test=train_label(cross_test,:)';
% 训练ELM
lamda=0.001;  %% 正则化系数在0.0007-0.00037之间时，一个一个试出来的
H1=P_train+1/lamda;% 加入regularization factor

T =T_train;            %训练集标签
T1=ind2vec(T);              %做分类需要先将T转换成向量索引
OutputWeight=pinv(H1') *T1'; 
Y=(H1' * OutputWeight)';

temp_Y=zeros(1,size(Y,2));
for n=1:size(Y,2)
    [max_Y,index]=max(Y(:,n));
    temp_Y(n)=index;
end
Y_train=temp_Y;
%Y_train=vec2ind(temp_Y1);
train_accuracy=sum(Y_train==T)/length(T);

H2=P_test+1/lamda;
T_cross=(H2' * OutputWeight)';                       %   TY: the actual output of the testing data
temp_Y=zeros(1,size(T_cross,2));
for n=1:size(T_cross,2)
    [max_Y,index]=max(T_cross(:,n));
    temp_Y(n)=index;
end
TY1=temp_Y;
% 加载输出
TV=T_test;
sum_accuracy=sum_accuracy+sum(TV==TY1) / length(TV);
end
per_accuracy_crossvalindation=sum_accuracy/10;
%========================================================
%===================交叉验证结束==========================