clc,clear
%% 二分法
func=@(z)z.^2-3*z+2-exp(z);
a=-rand;
b=-a+1;
bounds=[a,b];
tol=1/2*1e-8;
iter_max=100;
[x,~,~,error_his1]=Bisection(func,iter_max,tol,bounds);
fprintf('解为：%f',x)

%% 牛顿法
[x,~,~,error_his2]=Newtown(func,iter_max,tol);
fprintf('解为：%f',x)

%% 简化牛顿法 平行弦法
[x,~,~,error_his3]=Simple_Newtown(func,iter_max,tol);
fprintf('解为：%f',x)

%% result
%将误差上限设为1/2e-8时 ，二分法迭代 收敛
%在收敛容差设为1e-8时，牛顿法迭代   收敛，简化牛顿法迭代 收敛
%% visualize 
plot(1:length(error_his1),error_his1,'r-',1:length(error_his2),error_his2,'b-',1:length(error_his3),error_his3,'k-')
legend('Biesection','Newtown','Simple-Newtown')
hold off

function [x,error,x_his,error_his]=Bisection(func,iter_max,tol,bounds)
% Bisection 函数使用二分法来求解给定函数在指定区间上的根。
% 输入参数:
%   func: 待求解的函数句柄或符号表达式。
%   iter_max (可选): 最大迭代次数，默认为 100。
%   tol (可选): 误差容限，默认为 1e-8。
%   bounds: 包含上下界的一维数组，用于指定求解区间。
%
% 输出参数:
%   x: 近似根的值。
%   error: 近似根的误差。
%   x_his: 迭代过程中的根的历史记录。
%   error_his: 迭代过程中根的误差的历史记录。
    if isempty(iter_max)
        iter_max=100;
    end
    if isempty(tol)
        tol=1e-8;
    end
    a=bounds(1);b=bounds(2);
    fa=func(a);fb=func(b);
    iter_num=1;
    x_his=ones(iter_max,1);
    error_his=zeros(iter_max,1);
    while iter_num<iter_max
        x=(a+b)/2;
        x_his(iter_num)=x;
        fx=func(x);
        fprintf('iter_num=%d,a/fa=%.4f/%.4f,x/fx=%.4f/%.4f,b/fb=%.4f/%.4f', ...
            iter_num,a,fa,x,fx,b,fb)
        fprintf('\n')
        error=b-a;
        error_his(iter_num)=error;
        if error<tol
            fprintf('误差小于容差,退出循环,误差上限为：%.2e\n',error)
            error_his=error_his(1:iter_num);
            x_his=x_his(1:iter_num,:);
            break;
        end
        if fx*fa<0
            fb=fx;b=x;
        elseif fx*fb<0
            a=x;fa=fx;
        else 
            fprintf('func(a)*func(b)>0')
            break
        end
        if fa==0
            x=a;
            break  
        elseif fb==0
            x=b;
            break
        end
        iter_num=iter_num+1;
    end
    if iter_max==iter_num
        fprintf('达到最大迭代次数退出循环,误差上限为：%.2e\n',error)
    end
end

function [x,error,x_his,error_his]=Newtown(func,iter_max,tol,x0)
% Newtown 函数使用牛顿法来求解给定函数的根。 
% 输入参数:
%   func: 待求解的函数句柄或符号表达式。 传入的函数符号应为 z
%   iter_max (可选): 最大迭代次数，默认为 100。
%   tol (可选): 误差容限，默认为 1e-8。
%   x0 (可选): 初始值，默认为随机生成的大于 1 的数。
%
% 输出参数:
%   x: 近似根的值。
%   error: 近似根的误差。
%   x_his: 迭代过程中的根的历史记录。
%   error_his: 迭代过程中根的误差的历史记录。
   syms z
   f=sym(func);
   df=diff(f,z);
   df=matlabFunction(df);
   if isempty(iter_max)
        iter_max=100;
    end
    if isempty(tol)
        tol=1e-8;
    end
    if nargin<4
        x0=rand+1;
        x=x0;
        fprintf('未输入初始值,随机初始化x0=%f\n',x)
    else
        x=x0;
    end
    x_his=ones(iter_max,1);
    error_his=zeros(iter_max,1);
    iter_num=1;
    while iter_num<iter_max
        x0=x;
        x=x0-func(x)/df(x);
        x_his(iter_num)=x;
        error=norm(x-x0);
        %error=norm(func(x));
        error_his(iter_num)=error;
        fprintf('iter_num= %d Error=%.2e\n',iter_num,error)
        if error<tol
            fprintf('误差小于容差,退出循环,误差||x-x0||为：%.2e\n',error)
            fprintf('func(x)=%.2e\n',func(x))
            error_his=error_his(1:iter_num);
            x_his=x_his(1:iter_num,:);
            break;
        end
        iter_num=iter_num+1;
    end
    if iter_max==iter_num
        fprintf('达到最大迭代次数退出循环,误差||x-f(x)/df(x)||为：%.2e\n',error)
    end
end

function  [x,error,x_his,error_his]=Simple_Newtown(func,iter_max,tol,x0,df)
    % 输入参数：
    %   - func：非线性方程的函数句柄或字符串表示形式。函数 func 应接受一个标量输入 x，并返回该点的函数值。
    %   - iter_max：最大迭代次数（可选）。默认值为 100。
    %   - tol：收敛容差（可选）。默认值为 1e-8。
    %   - x0：初始点（可选）。默认值为一个随机数加 1。
    %   - df：函数 func 的导数（可选）。可以是函数句柄或字符串表示形式。如果未提供导数，将通过符号计算自动计算导数。
    %
    % 输出参数：
    %   - x：近似的根。
    %   - error：近似根与上一次迭代的差的范数。
    %   - x_his：每次迭代的根的历史记录。
    %   - error_his：每次迭代的误差的历史记录。
    %
    % 注意：
    %   - 当输入参数 iter_max 或 tol 为空时，使用默认值。
    %   - 如果未提供初始点 x0，则随机生成一个初始点。
    %   - 如果未提供导数 df，则使用符号计算自动计算导数。
   if isempty(iter_max)
        iter_max=100;
    end
    if isempty(tol)
        tol=1e-8;
    end
    if nargin<4
        x0=rand+1;
        x=x0;
        fprintf('未输入初始值,随机初始化x0=%f\n',x)
    else
        x=x0;
    end
    if nargin<5 
       syms z
       f=sym(func);
       df=diff(f,z);
       df=matlabFunction(df);
       df=df(x0);
    end
    x_his=ones(iter_max,1);
    error_his=zeros(iter_max,1);
    iter_num=1;
    while iter_num<iter_max
        x0=x;
        x=x0-func(x)/df;
        x_his(iter_num)=x;
        error=norm(x-x0);
        error_his(iter_num)=error;
        fprintf('iter_num= %d Error=%.2e\n',iter_num,error)
        if error<tol
            fprintf('误差小于容差,退出循环,误差||x-x0||为：%.2e\n',error)
            error_his=error_his(1:iter_num);
            x_his=x_his(1:iter_num,:);
            break;
        end
        iter_num=iter_num+1;
    end
    if iter_max==iter_num
        fprintf('达到最大迭代次数退出循环,误差||x-f(x)/df(x)||为：%.2e\n',error)
    end
end
