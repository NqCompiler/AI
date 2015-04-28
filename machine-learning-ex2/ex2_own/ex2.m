% load data and save into variables

data = load('ex2data1.txt');
X = data(:,1:end-1); y = data(:,end);
pos = find(y==1); neg = find(y==0);
plot(X(pos,1),X(pos,2),'bo',X(neg,1),X(neg,2),'rx','MarkerSize',7);
xlabel('Exam 1'); ylabel('Exam 2');

% map Featuring
% X = mapFeature(X(:, 1), X(:, 2));
X = [ones(length(y), 1) X];
[m, n] = size(X);

% run algorithm
initial_theta = zeros(n, 1);
lambda = 1;
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
    fminunc(@(t)(costFunction(t, X, y, lambda)), initial_theta, options);

cost

% plot decision Boundary
X1_int = linspace(min(X(:, 2)), max(X(:, 2)), 100);
X2_int = (-theta(1)-theta(2)*X1_int)./theta(3);

hold on;
plot(X1_int, X2_int, 'r-');
