%% 对称正定矩阵的三角分解或Cholesky 分解
clc,clear

%% 生成对称正定矩阵

%way1
n = 9;
A = randn(n);
A_sym = A * A';
A = A_sym + n * eye(n);
%------------------------------
% way2
% n = 6;
% A = randn(n);
% A_sym = 0.5 * (A + A');
% eigenvalues = eig(A_sym);
% while any(eigenvalues <= 0)
%     A = randn(n);
%     A = 0.5 * (A + A');
%     eigenvalues = eig(A);
% end
%-------------------------------
%%   生成对角占优矩阵
L=Cholesky(A);
B=L*L';
fprintf('cholesky分解误差：||L*L^T-A||=%f\n',norm(L'*L-A,2));% 默认为矩阵的二范数
L_=chol(A);
L_=L_';
B_=L_*L_';
fprintf('使用函数chol做分解误差：||L*L^T-A||=%f\n',norm(L'*L-A,2));% 默认为矩阵的二范数
fprintf('A-B=')
disp(A-B)
subplot(1,2,1)
spy(A),title(A); %sparsity pattern 结构
subplot(1,2,2)
spy(L),title(L);
b=randn(n,1);
x=ChoSolveLS(A,b)
fprintf('cholesky求解线性系统的误差：%f\n',norm(A*x-b,2))
disp('A*x-b=')
disp(A*x-b)

%% 判断test
A_=randn(10);
b=randn(10,1);
L_=Cholesky(A_);
x=ChoSolveLS(A_,b);

%% 函数chol
help chol

%% 求解线性系统
function x=ChoSolveLS(A,b)
L=Cholesky(A);
n=size(A,1);
x=zeros(n,1);
y=x;
if L==0
    fprintf('ERROR:矩阵A无法进行Cholesky分解')
    return
else
    %先求Ly=b
    U=L';
    y=ones(n,1)*b(1)/L(1,1);
    for i=2:n
        y(i)=(b(i)-L(i,1:i-1)*y(1:i-1))/L(i,i);
    end
    x=y/L(n,n);
    for i=n-1:-1:1
        x(i)=(y(i)-U(i,i+1:n)*x(i+1:n))/L(i,i);
    end
end
end
%% Cholesky 分解
function L=Cholesky(A)
[n,~]=size(A);
L=zeros(n);
if isPDM(A)==false
    fprintf('ERROR:该矩阵不是对称正定矩阵\n')
    return
else
    for i=1:n
        if i==1
            L(1,1)=sqrt(A(1,1));
            %L(1,1)=(A(1,1))^(1/2);
        else
            L(i,i)=sqrt(A(i,i)-sum(L(1:i-1,i).^2));
        end
        for j=i+1:n
            L(i,j)=(A(i,j)-sum((L(1:i-1,i).*L(1:i-1,j))))/L(i,i);
        end
    end
end
L=L';
end
%% 判断是否是对称正定矩阵
function flag=isPDM(X)
[n,m]=size(X);
flag=true;
if n~=m
    flag=false;
    fprintf('ERROR:该矩阵不是方阵\n')
    return
else
    eigenvalues = eig(X);
    if any(eigenvalues <= 0)
        flag=false;
        fprintf('ERROR:该矩阵不是正定矩阵\n')
    end
    %for i=1:n
    % if det(X(1:i,1:i)<0
    %     flag=false
    % end
    % end
    flag3=true;
    for i=1:n
        for j=1:n
            if X(i,j)~=X(j,i)
                flag3=false;
            end
        end
    end
    if flag3==false
        flag=flag3;
        fprintf('ERROR:该矩阵不是对称矩阵\n')
    end
end
end

