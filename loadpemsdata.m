% clear;
% numday:天数，10
% daytimesize：多少个15分钟，daytimesize
% numlink：多少条路，numlink
% conn = database('pems','root','root','com.mysql.jdbc.Driver','jdbc:mysql://localhost:3306/pems');
% numday = 363;
% numlink = 151;
% daytimesize = 96;
% d=cell(363*96*3*numlink,3);
% 
% for i=1:363
%     tic;
%     s1 = sprintf('SELECT Time,Station,Totalflow FROM d05_2013_month12 WHERE id >= (SELECT id FROM d05_2013_month12 LIMIT %d,1) LIMIT %d',(i-1)*daytimesize*numlink*3,daytimesize*numlink*3); 
%     cursf = fetch(exec(conn,s1));
%     d((i-1)*daytimesize*numlink*3+1:i*daytimesize*numlink*3,:) = cursf.Data;
%     fprintf('第%d组 ',i);
%     toc;
% end
stationid = d(1:147,2);
d = sortrows(d,[2 1]);
daydata =zeros(daytimesize*numlink*numday,3);
for i=1:daytimesize*numlink*numday
    daydata(i,1) = d{(i-1)*3+1,2};
    daydata(i,2) = mod(i,daytimesize);
    if mod(i,daytimesize)==0
        daydata(i,2) = daytimesize;
    end
    daydata(i,3) = d{(i-1)*3+1,3}+d{(i-1)*3+2,3}+d{(i-1)*3+3,3};
end
datarow=daydata(:,3)';
[datarow,ps]=mapminmax(datarow,0,1);
daydata(:,3)=datarow';
save data/pemsd05_2013_day363_link151 daydata ps stationid;