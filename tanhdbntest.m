clear;
clc;
bpcost=@tanhbpcost;
dbnfit=@dbnFittanh;
run=@tanhrunnet;
tanh20161021=[];
numday=362;
daytimesize=96;
numlink=147;
addpath data/
load('pemsd05_2013_day363_link147.mat');
%获取当前时间，把时间和data写入log
for numperiod = 4:4
    [data,labels] = createPemsTraindata(daydata,daytimesize,numlink,numperiod);
    for layer = 1:3
        for hidenodenum = 100:100:500
        hidelayer=[];
            for h=1:layer
            hidelayer = [hidelayer,hidenodenum];
            end
            for week=0:1
                for day =0:1
                    [cost,time,MRE,MAE,RMSE]=tanhdbn(data,labels,numperiod,week,day,hidelayer,bpcost,dbnfit,run,ps);
                    r=[numperiod,layer,hidenodenum,week,day,cost,MRE,MAE,RMSE,time];
                    tanh20161021=[tanh20161021;r];
                end
            end
        end
    end
end