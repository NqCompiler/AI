%
%
% Exercise 1 - Linear Regression with multiple variables
%
%

% load data into variables

data = load('ex1data2.txt');
X = data(:,1:end-1);
y = data(:,end);
m = length(y);

% normalize data
X = featureNormalize(X);
X = [ones(m,1) X];
n = size(X, 2);

% run gradient descent
init_theta = zeros(n, 1);
alpha = 0.01;
iterations = 7000;

J = computeCost(X, y, init_theta);
[theta, J_his] = gradientDescent(X, y, init_theta, alpha, iterations);
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

% show plot of J
plot([1:1:iterations], J_his, '-');

% run normal equation
theta = pinv(X'*X)*X'*y;
fprintf('Theta found by normal equation: ');
fprintf('%f %f \n', theta(1), theta(2));