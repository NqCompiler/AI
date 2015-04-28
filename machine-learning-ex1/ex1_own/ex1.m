%
%
% Exercise 1 coursera ML 
%
%

% TODO: keine For Loop Schleife verwenden in Surf Plot
%       Warum J_val Transpose?

%
% linear regression with one variable
%

% load data and save to variables
data = load('ex1data1.txt');
X = data(:, 1);
y = data(:, 2);
m = length(y);

% plot Data
plotData(X, y);

%
% run gradient descent
%

X = [ones(m,1) X]; % add intercept term
n = size(X, 2);
theta = zeros(n, 1);

% options for batch gradient descent
iterations = 1500;
alpha = 0.01;

computeCost(X, y, theta);
[theta, J_his] = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

% predict two examples
predict1 = [1, 3.5]*theta;
predict2 = [1, 7]*theta;
fprintf('Prediction 1 is %f and Prediction 2 is %f \n', predict1, predict2);

% plot linear regression
hold on; 
plot(X(:,2), X*theta, '-');
legend('Training data', 'Linear regression');

% plot J history
figure;
plot([1:1:30], J_his(1:30), '-');

% plot cost function three dimensional
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% Solution provided by Andrew Ng
% J_vals = zeros(length(theta0_vals), length(theta1_vals));
% for i = 1:length(theta0_vals)
%     for j = 1:length(theta0_vals)
%         temp_theta = [theta0_vals(i); theta1_vals(j)];
%         J_vals(i,j) = computeCost(X, y, temp_theta);
%     end
% end

%figure;
%surf(theta0_vals, theta1_vals, J_vals');

[Theta0 Theta1] = meshgrid(theta0_vals, theta1_vals);
J_vals = zeros(length(theta0_vals)*length(theta1_vals), 1);

for i = 1:numel(Theta0)
   J_vals(i) = computeCost(X, y, [Theta0(i); Theta1(i)]); 
end

Jvals = reshape(J_vals, length(theta0_vals), length(theta1_vals));
figure;
surf(theta0_vals, theta1_vals, Jvals);

% plot contour
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, Jvals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
