% --------------------------------------------------------------
%  Grey Wolf Optimizer (GWO) - MATLAB Demo
%  Bài toán mẫu: Minimize Sphere Function
%  github_link: https://github.com/LuongHuongGiang20236027/group14_gwo
% --------------------------------------------------------------

clear; clc; close all;

%% === HÀM MỤC TIÊU ===
% Sphere function: f(x) = sum(x.^2)
objFunc = @(x) sum(x.^2);

%% === THAM SỐ GWO ===
dim = 10;              % số chiều
lb = -10;              % biên dưới
ub = 10;               % biên trên
N = 30;                % số lượng sói
T = 100;               % số vòng lặp

%% === KHỞI TẠO BẦY SÓI ===
positions = lb + (ub - lb) * rand(N, dim);

alpha.pos = zeros(1, dim);  alpha.score = inf;
beta.pos  = zeros(1, dim);  beta.score  = inf;
delta.pos = zeros(1, dim);  delta.score = inf;

convergence = zeros(1, T);

%% === VÒNG LẶP GWO ===
for t = 1:T

    a = 2 * (1 - t / T);  % giảm tuyến tính từ 2 → 0

    % --- Đánh giá & cập nhật alpha, beta, delta ---
    for i = 1:N
        fitness = objFunc(positions(i,:));

        if fitness < alpha.score
            delta = beta;
            beta = alpha;
            alpha.pos = positions(i,:);
            alpha.score = fitness;

        elseif fitness < beta.score
            delta = beta;
            beta.pos = positions(i,:);
            beta.score = fitness;

        elseif fitness < delta.score
            delta.pos = positions(i,:);
            delta.score = fitness;
        end
    end

    % --- Cập nhật vị trí các sói ---
    for i = 1:N
        X = positions(i,:);

        % Alpha
        r1 = rand(1,dim); r2 = rand(1,dim);
        A1 = 2*a*r1 - a;
        C1 = 2*r2;
        X1 = alpha.pos - A1 .* abs(C1 .* alpha.pos - X);

        % Beta
        r1 = rand(1,dim); r2 = rand(1,dim);
        A2 = 2*a*r1 - a;
        C2 = 2*r2;
        X2 = beta.pos - A2 .* abs(C2 .* beta.pos - X);

        % Delta
        r1 = rand(1,dim); r2 = rand(1,dim);
        A3 = 2*a*r1 - a;
        C3 = 2*r2;
        X3 = delta.pos - A3 .* abs(C3 .* delta.pos - X);

        newX = (X1 + X2 + X3) / 3;

        % ràng buộc biên
        newX = max(newX, lb);
        newX = min(newX, ub);

        positions(i,:) = newX;
    end

    % Lưu nghiệm tốt nhất hiện tại
    convergence(t) = alpha.score;

    if mod(t,10)==0
        fprintf("Iter %d/%d -- Best = %.6f\n", t, T, alpha.score);
    end
end

%% === KẾT QUẢ ===
fprintf("\n===== KẾT QUẢ TỐT NHẤT =====\n");
disp("Best score:");
disp(alpha.score);
disp("Best position:");
disp(alpha.pos);

%% === VẼ ĐỒ THỊ HỘI TỤ ===
figure;
plot(convergence, 'LineWidth',2);
xlabel("Iteration");
ylabel("Best Fitness");
title("Grey Wolf Optimizer - Convergence Curve");
grid on;

