function [MRE]=dbn(numlink,week,day,argu,filename)
t1=clock;
% rbmFit里向顶层方向的*2-1
% week = 1; %1是工作日weekday，0是双休日weekend
% day = 0; %1是白天daytime，0是晚上nighttime
% numlink=151;
% daytimesize=96;

addpath RBM/
topActivateFunc=argu{1};
rbmbatchsize=argu{2};
rbmmaxepoch=argu{3};
bpmaxepoch=argu{4};
numperiod=argu{5};
layernum=argu{6};
hidenodenum=argu{7};

% topActivateFunc=@logistic;
% rbmbatchsize=50;
% rbmmaxepoch=50;
% bpmaxepoch=1000;
% numperiod=4;
% layernum=1;
% hidenodenum=100;
hidelayer=[];
for h=1:layernum
    hidelayer = [hidelayer,hidenodenum];
end
addpath data/
dataname=sprintf('new147k%d',numperiod);
load(dataname);

% PeMS取train，test数据
numtrain = 71*96; %训练的天数
numtest = 18*96; %测试的天数
traindata = data(1:numtrain,:);
trainlabels = labels(1:numtrain,:);
testdata = data(numtrain+1:numtrain+numtest,:);
testlabels = labels(numtrain+1:numtrain+numtest,:);

% 分weekday和weekend
weekflag=ones(size(data,1),1);
for i=1:size(data,1)
   if i<=68*96
       if mod(ceil(i/96)+1,7)==6||mod(ceil(i/96)+1,7)==0
           weekflag(i)=0;
       end
   else
       if i>68*96&&i<=258*96
           if mod(ceil(i/96)+2,7)==6||mod(ceil(i/96)+2,7)==0
              weekflag(i)=0;
           end
       else
           if mod(ceil(i/96)+3,7)==6||mod(ceil(i/96)+3,7)==0
              weekflag(i)=0;
           end
       end
   end
end
weektrain = weekflag(1:numtrain);
weektest = weekflag(numtrain+1:numtrain+numtest);

traindata=traindata(weektrain==week,:);
trainlabels=trainlabels(weektrain==week,:);
testdata=testdata(weektest==week,:);
testlabels=testlabels(weektest==week,:);

% 分白天晚上
daytime=zeros(96,1);
daytime(21:84)=1; %早上6点到晚上8点
traindaytime=repmat(daytime,size(traindata,1)/96,1);
testdaytime=repmat(daytime,size(testdata,1)/96,1);

traindata = traindata(traindaytime==day,:);
trainlabels = trainlabels(traindaytime==day,:);
testdata = testdata(testdaytime==day,:);
testlabels = testlabels(testdaytime==day,:);

numhide = size(hidelayer,2);
op1.verbose=true;
op1.maxepoch=rbmmaxepoch;
op1.batchsize=rbmbatchsize;
op2=op1;
op3=op1;
op4=op1;
op5=op1;

% 训练

models=dbnFit(traindata,hidelayer,trainlabels,topActivateFunc,...
op1,op2,op3,op4,op5); %ѵ��
[m,cost] = bpFine(models, traindata, trainlabels, bpmaxepoch, topActivateFunc);

% 测试
numtest = size(testdata,1);
a = runnet(testdata, m.W, m.b, numhide, topActivateFunc);
h = a{numhide+2};
dp=mapminmax('reverse',h,ps);
dr=mapminmax('reverse',testlabels,ps);
dr(dr==0)=1;
dp(dp<1)=5;
re=sum(abs(dp-dr)./dr)/numtest;
count=0; %测试结果比较差的路段个数
badlink=''; %测试结果比较差的路段
for i=1:numlink
    if re(i)>1
%         re(i)=0;
        s=sprintf('%d ',i);
        badlink=[badlink s];
        count=count+1;
    end
end
MRE = sum(re)/(numlink-count);
MAE = sum(sum(abs(dp-dr)))/(numlink*numtest);
RMSE = sqrt(sum(sum((dp-dr).^2))/(numlink*numtest));
t2=clock;
time=etime(t2,t1);
plot(1:numlink,re,'*r');
if isequal(topActivateFunc,@logistic)
    topActivateFunc=@sigmod;
end
log=sprintf('%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.2f\t%.2f\t%.2f\t%s',...
            func2str(topActivateFunc),rbmbatchsize,rbmmaxepoch,bpmaxepoch,numperiod,...
        layernum,hidenodenum,week,day,cost,MRE,MAE,RMSE,time,badlink);
fp = fopen(filename,'at'); 
fprintf(fp, '\n%s', log);
fclose(fp);