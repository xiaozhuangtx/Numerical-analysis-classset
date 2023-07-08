clc,clear
% n=8;
% A=randn(n)
% b=randn(n,1);
A=[10,-1,-2;-1,10,-2;-1,-1,5];
b=[7.2;8.3;4.2];
iter_max=100;
tol=1e-10;

%%Jacobi method
toc
[x,~,~,error_his1]=Jacobi(A,b,iter_max,tol);
tic
disp(['最终解为：',num2str(x')])

%%SOR method
[x,~,~,error_his2]=SOR(A,b,iter_max,tol,[],1.05);
disp(['最终解为：',num2str(x')])

%%Gauss——Seidel
toc
[x,~,~,error_his3]=SOR(A,b,iter_max,tol,[],1);
tic 
disp(['最终解为：',num2str(x')])
plot(1:length(error_his1),error_his1,1:length(error_his2),error_his2,1:length(error_his3),error_his3)
legend('Jacobi method','SOR method','Gauss——Seidel')
hold off
%% 迭代速度：SOR（w=1.05）>Gauss Seidel > Jacobi
%%
function [x,Error,x_his,error_his]=Jacobi(A,b,iter_max,tol,x0)
% Jacobi 迭代方法求解线性方程组 Ax=b。
% 输入参数：
%   A: 系数矩阵
%   b: 右侧常数向量
%   iter_max: 最大迭代次数（可选，默认为100）
%   tol: 收敛容差（可选，默认为1e-8）
%   x0: 初始解向量（可选，默认为全零向量）
% 输出参数：
%   x: 求解得到的解向量
%   Error: 每次迭代的误差
%   x_his: 每次迭代得到的解向量历史记录
%   error_his: 每次迭代的误差历史记录
[~,n]=size(A);
D=zeros(n);
if isempty(iter_max)
    iter_max=100;
end
if isempty(tol)
    tol=1e-8;
end
if nargin==5
    x=x0;
else
    x=zeros(n,1);
end
x_his=x'.*ones(iter_max,n); %以行的形式存储解
error_his=zeros(iter_max,1);
iter_num=1;
for i=1:n
     D(i,i)=A(i,i);
end
while iter_num<iter_max
    temp=x;
    for k=1:n        
        x(k)=(-A(k,:)*temp+A(k,k)*temp(k)+b(k))/A(k,k);
    end
    x_his(iter_num,:)=x;
    Error=norm(A*x-b);
    fprintf('iter_num= %d Error=%.2e\n',iter_num,Error)
    error_his(iter_num)=Error;
    if Error<=tol
        fprintf('误差小于:%.2e\n退出循环',tol)
        error_his=error_his(1:iter_num);
        x_his=x_his(1:iter_num,:);
        break;        
    end
    iter_num=iter_num+1;    
end
if iter_num==iter_max
    fprintf('达到最大迭代次数，误差为：%.2e',error_his(end))
end
end



function [x,Error,x_his,error_his]=SOR(A,b,iter_max,tol,x0,w)
% SOR (Successive Over-Relaxation) 方法求解线性方程组 Ax=b。
% 输入参数：
%   A: 系数矩阵
%   b: 右侧常数向量
%   w: 松弛因子（可选，默认为1.3）
%   iter_max: 最大迭代次数（可选，默认为100）
%   tol: 收敛容差（可选，默认为1e-8）
%   x0: 初始解向量（可选，默认为全零向量）
% 输出参数：
%   x: 求解得到的解向量
%   Error: 每次迭代的误差
%   x_his: 每次迭代得到的解向量历史记录
%   error_his: 每次迭代的误差历史记录
%   w=1 为Gauss-Seidel
    
    [~,n]=size(A); 
    error_his=zeros(iter_max,1);
    iter_num=1;
    if nargin<3
        iter_max=100;
    end
    if nargin<4
        tol=1e-8;
    end
    if nargin<5
        x0=zeros(n,1);
        x=x0;
    end
    if nargin<6
        w=1.3;
         fprintf('松弛因子w=1.3\n')
    elseif nargin==6
        if w==1
            fprintf('w=1 <=> Gauss-Seidel method\n')
        end
    end
    if isempty(x0)
        x=zeros(n,1);
    end
    x_his=x'.*ones(iter_max,n); %以行的形式存储
    x_pre=x;
    x_hat=x;
    while iter_num<iter_max        
        for k=1:n        
            x_hat(k)=(-A(k,:)*x+A(k,k)*x(k)+b(k))/A(k,k);
            x(k)=(1-w)*x_pre(k)+w*x_hat(k);
        end
        x_his(iter_num,:)=x';
        Error=norm(A*x-b);
        fprintf('iter_num= %d Error=%.2e\n',iter_num,Error)
        error_his(iter_num)=Error;
        if Error<=tol
            fprintf('误差小于:%.2e\n退出循环',tol)
            error_his=error_his(1:iter_num);
            x_his=x_his(1:iter_num,:);
            break;        
        end
        x_pre=x;
        iter_num=iter_num+1;    
    end
    if iter_num==iter_max
        fprintf('达到最大迭代次数，误差为：%.2e',error_his(end))
    end
end
