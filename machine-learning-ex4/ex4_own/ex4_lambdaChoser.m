% always take the same random stream
rand('seed', 3);

% load variables

load('ex4data1.mat');
load('ex4weights.mat');

m = length(X);
X = [ones(m,1) X];
num_labels = max(y);
lambda = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24];

% split into X_train, X_cv, X_test
ind_train_end = ceil(0.6*m);
ind_cv_end = ceil(0.8*m);
perm = randperm(m);

X = X(perm,:);
y = y(perm,:);

X_train = X(1:ind_train_end,:);
y_train = y(1:ind_train_end,:);

X_cv = X(ind_train_end+1:ind_cv_end,:);
y_cv = y(ind_train_end+1:ind_cv_end,:);

X_test = X(ind_cv_end+1:end,:);
y_test = y(ind_cv_end+1:end,:);

a = eye(num_labels);
y_class = zeros(m,num_labels);
y_class(:,:) = a(y,:);

y_class_train = y_class(1:ind_train_end,:);
y_class_cv = y_class(ind_train_end+1:ind_cv_end,:);
y_class_test = y_class(ind_cv_end+1:end,:);

% initialize random thetas
init_theta1 = randInitializeWeights(25, 400); % 25 x 401
init_theta2 = randInitializeWeights(10, 25);  % 10 x 26

% optimize and get thetas
theta_un    = [init_theta1(:); init_theta2(:)];
options     = optimset('MaxIter',100);
J_train     = zeros(length(lambda),1);
J_cv        = zeros(length(lambda),1);
theta_hist  = zeros(length(theta_un),length(lambda));

tic
for i = 1:length(lambda)
    costFunction =  @(t)(nnCostFunction(t,X_train,y_train,y_class_train,lambda(i)));
    [theta, cost] = fmincg( costFunction, theta_un, options);
    
    J_train(i)  = nnCostFunction(theta,X_train,y_train,y_class_train,0);
    J_cv(i)     = nnCostFunction(theta,X_cv,y_cv,y_class_cv,0);
    theta_hist(:,i)= theta;
end
toc

plot(lambda,J_train,'b',lambda,J_cv,'r');

% take minimum of J_cv and set theta
[~, J_cv_min] = min(J_cv);
theta = theta_hist(:,J_cv_min);

% set variables for prediction
X = X_cv;
y = y_cv;

% prediction
t1 = reshape(theta(1:25 * 401), 25, 401);      % 25 x 401
t2 = reshape(theta(25 * 401 + 1:end), 10, 26); % 10 x 26
[m, n] = size(X);
a1 = X;
z2 = X*t1';                  % 5000 x 25
a2 = sigmoid(z2);            % 5000 x 25
a2 = [ones(m,1) a2];         % 5000 x 26
h = sigmoid(a2*t2');         % 5000 x 10
[dummy h_ind] = max(h,[],2); % 5000 x 1

prec = sum(y == h_ind)/length(y)

