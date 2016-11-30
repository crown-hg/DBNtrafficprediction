x0=[1,1]';
% d0=[-1,-1]';
% alpha=armij(d0,x0 )
 y=newton(x0);    %精确搜索
 y=newtont(x0);         %wolfe搜索
 y=newtonamj(x0);          %armijo搜索
 y=deepest(x0);