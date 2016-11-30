function [ model,cost ] = bpFine( m, data, labels, bpmaxepoch, topActivateFunc)
lambda = 0.0001;
visibleSize=size(data,2);
hiddenSize = size(m{1}.W,2);
numClasses = size(labels,2);
numhide = size(m,1);
theta=[];
b=[];
%sigma=m{1}.sigma;
for i=1:numhide
    theta = [theta;m{i}.W(:)];
    b=[b;m{i}.b(:)];
end
theta = [theta;m{i}.Wc(:)];
b = [b;m{i}.cc(:)];
theta = [theta;b];
addpath minFunc/
options.Method = 'lbfgs';
options.display = 'on';
options.maxIter = bpmaxepoch;
[opttheta, cost] = minFunc( @(p) bpcost( p ,...
                    numhide,numClasses ,visibleSize,hiddenSize,lambda, data, labels, topActivateFunc),...
                     theta, options);

[model.W, model.b]=thetatowb(opttheta, numhide, numClasses, visibleSize, hiddenSize);

end

