%
% one-vs-all VS. softmax-regression
%
clear;
clc;

% load data and set variables
load('ex3data1.mat') % loads the variables

K = 10;                     % number of classes
X = [ones(length(X),1) X];  % add intercept term
[m, n] = size(X);
all_theta = zeros(n, K);    %401x10
lambda = 0.1;
options = optimset('GradObj','on','MaxIter',100);

% one vs all
tic
for i = 1:K
    initial_theta = zeros(n, 1);
    [theta] = fmincg(@(t)(lrCostFunction(t, X, (y == i), lambda)),...
        initial_theta, options);
    all_theta(:,i) = theta;
end
toc

% prediction function
[max_h, predictions] = max(X*all_theta,[],2);

accuracy = sum(y==predictions)/m
error_rate = 1 - sum(y==predictions)/m

%K = 10;
%X = [ones(length(X),1) X];
[m, n] = size(X);

% softmax-regression
lambda = 0.1;
options = optimset('GradObj','on','MaxIter',200);

initial_theta = rand(n, K)*2*0.7 - 0.7;
tic
[theta] = fmincg(@(t)(softmaxCost(t, X, y, lambda)),...
    initial_theta(:), options);
toc
all_theta = reshape(theta, n, K);

% prediction function
z = X*all_theta;    % m x K (m x n * n x K)
h = exp(z);     % m x K
exp_sum = sum(h, 2);                % m x 1
h = bsxfun(@rdivide, h, exp_sum);   % f x u
[max_h2, predictions2] = max(h, [], 2);

accuracy = sum(y==predictions2)/m
error_rate2 = 1 - accuracy


%calculate F1-Score
f1_predictions = zeros(K, 1);
f1_predsami = zeros(K, 1);
f1_recall = zeros(K, 1);
f1_recallsami = zeros(K, 1);
f1_score = zeros(K, 1);
f1_scoresami = zeros(K, 1);


for i = 1:K
    
   f1_y = (y==i);
   f1_h = (predictions==i);
   
   y_0 = find(f1_y == 0); %4500x1
   h_0 = find(f1_h == 0);
   y_1 = find(f1_y == 1); %500x1
   h_1 = find(f1_h == 1);
   
   tp = sum(ismember(y_1, h_1));
   tn = sum(ismember(y_0, h_0));
   fp = sum(ismember(y_1, h_0));
   fn = sum(ismember(h_1, y_0));
   
   f1_predictions(i) = tp/(tp+fn);
   f1_recall(i) = tp/(tp+fp);
   f1_score(i) = 2*(f1_predictions(i)*f1_recall(i))/(f1_predictions(i)+f1_recall(i));
   %sami
   tp_sami = f1_h & f1_y;
   tn_sami = ~f1_h & ~f1_y;
   fp_sami = f1_h & ~f1_y;
   fn_sami = ~f1_h & f1_y;
   
   f1_predsami(i) = sum(tp_sami)/(sum(tp_sami)+sum(fp_sami));
   f1_recallsami(i) = sum(tp_sami)/(sum(tp_sami)+sum(fn_sami));
   f1_scoresami(i) = 2*(f1_predsami(i)*f1_recallsami(i))/(f1_predsami(i)+f1_recallsami(i));
end

fprintf('F1 score Softmax:\n')
f1_scoresami
fprintf('F1 sum score Softmax:\n')
sum(f1_scoresami)

%calculate F1-Score
f1_predictions = zeros(K, 1);
f1_predsami = zeros(K, 1);
f1_recall = zeros(K, 1);
f1_recallsami = zeros(K, 1);
f1_score = zeros(K, 1);
f1_scoresami = zeros(K, 1);


for i = 1:K
    
   f1_y = (y==i);
   f1_h = (predictions2==i);
   
   y_0 = find(f1_y == 0); %4500x1
   h_0 = find(f1_h == 0);
   y_1 = find(f1_y == 1); %500x1
   h_1 = find(f1_h == 1);
   
   tp = sum(ismember(y_1, h_1));
   tn = sum(ismember(y_0, h_0));
   fp = sum(ismember(y_1, h_0));
   fn = sum(ismember(h_1, y_0));
   
   f1_predictions(i) = tp/(tp+fn);
   f1_recall(i) = tp/(tp+fp);
   f1_score(i) = 2*(f1_predictions(i)*f1_recall(i))/(f1_predictions(i)+f1_recall(i));
   %sami
   tp_sami = f1_h & f1_y;
   tn_sami = ~f1_h & ~f1_y;
   fp_sami = f1_h & ~f1_y;
   fn_sami = ~f1_h & f1_y;
   
   f1_predsami(i) = sum(tp_sami)/(sum(tp_sami)+sum(fp_sami));
   f1_recallsami(i) = sum(tp_sami)/(sum(tp_sami)+sum(fn_sami));
   f1_scoresami(i) = 2*(f1_predsami(i)*f1_recallsami(i))/(f1_predsami(i)+f1_recallsami(i));
end

fprintf('F1 score Softmax:\n')
f1_scoresami
fprintf('F1 sum score Softmax:\n')
sum(f1_scoresami)
