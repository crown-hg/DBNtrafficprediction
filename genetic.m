%% 遗传算法程序
agruname = sprintf('topFunc\trbsize\trbmaxe\tbpmax\tK\tL\tNH\tWEEK\tDAY\tCOST\tMRE\tMAE\tRMSE\tTIME\tBADLINK');
filename=sprintf('/home/crown/hg/testResult/%s_dbn_pemsd05_stationNew147_train71_test18.txt',datestr(date));
fp = fopen(filename,'wt'); 
fprintf(fp, '%s', agruname);
fclose(fp);
% n-- 种群规模
% ger-- 迭代次数
% pc--- 交叉概率
% pm-- 变异概率
% v-- 初始种群（规模为n）
% f-- 目标函数值
% fit-- 适应度向量
% vx-- 最优适应度值向量
% vmfit-- 平均适应度值向量
% tic;
week=0;
day=0;
numlink=147;

n=10;%种群规模
ger=100;%迭代次数
pc=0.65;%交叉概率
pm=0.05;%变异概率
% 以上为经验值，可以更改。
% 生成初始种群
v=init_population(n,15); %得到初始种群，10*15的0-1矩阵
[N,L]=size(v);           %得到初始规模行，列
%计算适应度
fit=countfit(v,week,day,numlink,filename);
str=sprintf('初始结果  最好%.4f  平均%.4f',min(-fit),mean(-fit));
fp = fopen(filename,'at'); 
fprintf(fp, '\n%s', str);
fclose(fp);
% 迭代前的初始化
vmfit=[];%平均适应度
vx=[]; %最优适应度
it=1; % 迭代计数器
% 开始进化
for it=1:ger %迭代次数 %100代
    %Reproduction(Bi-classist Selection)
    vtemp=roulette(v,fit);%复制算子,选中优秀的染色体
    %Crossover    
    v=crossover(vtemp,pc);%交叉算子 
    %Mutation变异算子
    M=rand(N,L)<=pm;%这里的作用找到比0.05小的分量
    %M(1,:)=zeros(1,L);
    v=v-2.*(v.*M)+M;%两个0-1矩阵相乘后M是1的地方V就不变，再乘以2. NICE!!确实好！！！把M中为1的位置上的地方的值变反
    %这里是点乘 %变异  
    %交叉变异后的结果 
    fit=countfit(v,week,day,numlink,filename);    %计算数值
    [sol,indb]=max(fit);% 每次迭代中最优目标函数值，包括位置
    v(1,:)=v(indb,:);   %用最大值代替
    fit_mean=mean(fit); % 每次迭代中目标函数值的平均值。mean求均值
    str=sprintf('第%d结果  最好%.4f  平均%.4f',it,min(-fit),mean(-fit));
    fp = fopen(filename,'at'); 
    fprintf(fp, '\n%s', str);
    fclose(fp); 
    vx=[vx sol];        %最优适应度值
    vmfit=[vmfit fit_mean];%适应度均值
end
runtime=toc;%记时结束

%% Decodify bitstrings
function fit=countfit(v,week,day,numlink,filename)
    x=decode(v);
    N=size(v,1); 
    fit=zeros(N,1);
    for i=1:N
        fit(i)=-dbn(numlink,week,day,x(i,:),filename); %这里是为了得到较小的mre，所以用负值做适应度
    end
end

function x=decode(v)
    N=size(v,1);
    x=cell(N,7);
    for i=1:N
        x(i,:)=decode_row(v(i,:));
    end
end
function r=decode_row(vr)
    r=cell(1,7);
    if vr(1)==0
        r{1}=@tanh;
    else
        r{1}=@logistic;
    end
    r{2}=r2argu(vr(2:3),100,100);
    r{3}=r2argu(vr(4:6),25,25);
    r{4}=r2argu(vr(7:8),250,250);
    r{5}=r2argu(vr(9:10),1,1);
    r{6}=r2argu(vr(11:12),1,1);
    r{7}=r2argu(vr(13:15),100,100);
end
function argu=r2argu(vr,begin,d)
    num=0;
    for i=1:size(vr,2)
        num=num*2+vr(i);
    end
    argu=begin;
    for j=1:num
        argu=argu+d;
    end
end
%%
%Crossover
function v=crossover(vtemp,pc)
[N,L]=size(vtemp);
C(:,1)=rand(N,1)<=pc;%选择被杂交的。<=pc就是1否则是0构成0-1向量
I=find(C(:,1)==1);%找分量等于1的元素，其下标构成向量。
for i=1:2:size(I)%两两配对所以以2为步长
    if i>=size(I)%奇数个处理 如果是奇数个，则最后一行不处理
        break;
    end
    site=fix(1+L*rand(1));%fix向零取整，L=22.%site属于1-22.随机确定交换点的位置
    temp=vtemp(I(i),:);%交换的暂存变量。T  记录要交叉的第一行基因
    vtemp(I(i),site:end)=vtemp(I(i+1),site:end);%交换后面的数值
    vtemp(I(i+1),site:end)=temp(:,site:end);%交换  temp没有被修改
end
v=vtemp;%复制返回
end        
%%
%Function init_population
function v=init_population(n1,s1)
v=round(rand(n1,s1));%rand产生随机数，%round四舍五入取整
end
%% 
function vtemp=roulette(v,fit)
N=size(v);  %N向量
fitmin=abs(min(fit));%最小值和绝对值
fit=fitmin+fit; %最小值加上步长，保证fit>=0.
%fit
S=sum(fit);%求向量的和
vtemp=zeros(size(v));
for i=1:N %一共要选N个，每个向量被选中的概率是fit(i)/sum(fit)
    SI=S*rand(1);%rand随机数。0-s之间的一个随机数
    for j=1:N
        if SI<=sum(fit(1:j))  %累加列值
            vtemp(i,:)=v(j,:);%选中此样本
            break
        end
    end
end
end

