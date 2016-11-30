function [newmodel] = bp(eb,mc,eta,maxIter,lambda, model, data, labels)
%lambda = 0.00001;      weight decay parameter权重衰减参数
%eta =0.6;              学习率
%eb = 0.1 ;             误差容限
%maxIter = 2000;        最大迭代次数 
%mc = 0.8;              动量因子

numhide = size(model,1);
W=cell(1,numhide+1);
b=cell(1,numhide+1);
Wgradold=cell(1,numhide+1);
bgradold=cell(1,numhide+1);
for i=1:numhide
    W{i}=model{i}.W;
    b{i}=model{i}.b;
end
W{numhide+1}=model{i}.Wc';
b{numhide+1}=model{i}.cc;
for n=1:maxIter
    a = runnet(data,W,b,numhide);
    h=a{numhide+2};
    y=labels;
    squared_error=0.5*sum((h-y).^2,1);
    ws = 0;
    for i=1:numhide+1
        ws = sum(sum(W{i}.^2));
    end
    m=size(data,1);
    if n==1
        costold=0;
    else
        costold=cost;
    end
    cost=1/m*sum(squared_error)+lambda/2*(ws);
    fprintf('第%d次迭代，误差是%f\n',n,cost);
    if cost<0.001||abs(cost-costold)<0.0000001
        break;
    end
    %% 计算残差
    % 这几步比较简化，需要看着推导公式来看
    delta=cell(1,numhide+2);
    delta{numhide+2} = -(y-a{numhide+2}).*(1-a{numhide+2}.^2);
    for i=numhide+1:-1:2
        delta{i}=(delta{i+1}*W{i}').*a{i}.*(1-a{i});
    end
    Wgrad=cell(1,numhide+1);
    bgrad=cell(1,numhide+1);
    for i=1:numhide+1
        Wgrad{i}=1/m*(a{i}'*delta{i+1})+lambda*W{i};
        bgrad{i}=1/m*sum(delta{i+1},1);
    end
%    if n==1
        for i=1:numhide+1
            W{i} = W{i}-eta.*Wgrad{i};
            b{i} = b{i} - eta.*bgrad{i};
        end
%     else
%         for i=1:numhide+1
%             W{i} = W{i} - (1-mc).*eta.*Wgrad{i} - mc.*Wgradold{i};
%             b{i} = b{i} - (1-mc).*eta.*bgrad{i} - mc.*bgradold{i};
%         end
%     end
%     for i=1:numhide+1
%        Wgradold{i} = Wgrad{i};
%        bgradold{i} = bgrad{i};
%     end
end
newmodel.W=W;
newmodel.b=b;

