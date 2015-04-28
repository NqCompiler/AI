% always take the same random stream
rand('seed', 3);

% load variables

load('ex4data1.mat');
load('ex4weights.mat');

m = length(X);
X = [ones(m,1) X];
num_labels = max(y);
lambda = 1;

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

% turn target variables into vector

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

% train network
tic
costFunction =  @(t)(nnCostCE(t, X_train, y_train, y_class_train, lambda));
[theta, cost] = fmincg(costFunction, theta_un, options);
toc

% set variables for prediction
X = X_test;
y = y_test;

% prediction
t1 = reshape(theta(1:25 * 401), 25, 401);      % 25 x 401
t2 = reshape(theta(25*401 + 1:end), 10, 26); % 10 x 26
[m, n] = size(X);
a1 = X;
z2 = X*t1';                  % 5000 x 25
a2 = sigmoid(z2);            % 5000 x 25
a2 = [ones(m,1) a2];         % 5000 x 26
z3 = (a2*t2');               % 5000 x 10
h_exp     = exp(z3);
h_exp_sum = sum(h_exp, 2);   % 5000 x 1
h_softmax = bsxfun(@rdivide, h_exp, h_exp_sum);

[~, h_ind] = max(h_softmax,[] , 2); % 5000 x 1

prec = sum(y == h_ind)/length(y)


% % vectorized
% clearvars a1 a2 a3 z2 z3 D1 D2
% 
% tic
% y_actual = zeros(length(y),num_labels);  % 5000 x 10
% one_indeces = sub2ind(size(y_actual),1:length(y),y');
% y_actual(one_indeces) = 1;
% y_actual = y_actual'; % 10 x 5000
% 
% Theta1 = init_theta1;
% Theta2 = init_theta2;
% 
% a1 = X';                       % 401 x 5000
% z2 = Theta1*a1;
% a2 = sigmoid(z2);              % 25 x 5000
% a2 = [ones(1,size(a2,2)); a2]; % 26 x 5000
% z3 = Theta2*a2;
% a3 = sigmoid(z3);              % 10 x 5000
% 
% delta3 = (a3-y_actual);                         % 10x5000
% z2 = [ones(1,size(z2,2)); z2];                  % 26 x 5000
% delta2 = (Theta2'*delta3).*sigmoidG(z2); % 26 x 5000
% delta2 = delta2(2:end,:);                       % 25x5000
% 
% D1 = delta2*a1';          % 25x401
% D2 = delta3*a2';          % 10x26
% toc
% 
% % % calc grads (because the grads where summed up for each example, we have to get the arithmetic grads) 
% % Theta1_grad = (1/m).*D1; % 25x401
% % Theta1_grad = Theta1_grad + (lambda/m).*Theta1_reg; % regularized
% % Theta2_grad = (1/m).*D2; % 10x26
% % Theta2_grad = Theta2_grad + (lambda/m).*Theta2_reg; % regularized
% 
% D_ = 1/m * [D1(:); D2(:)];

% 
% clearvars a1 a2 a3 z2 z3 d3 d2 D1 D2
% y_matrix = eye(num_labels);
% Theta1 = init_theta1;
% Theta2 = init_theta2;
% tic
% for t = 1:m
%     %set forward prop
%     a1(t,:) = X(t,:); % 1x401
%     z2(t,:) = a1(t,:)*Theta1'; % 1x25
%     a2(t,:) = [1 sigmoid(z2(t,:))]; % 1x26
%     z3(t,:) = a2(t,:)*Theta2'; % 1x10
%     a3(t,:) = sigmoid(z3(t,:)); % 1x10
%     
%     % set y-value
%     y_k = y_matrix(y(t),:); % 1x10
%     
%     % set deltas
%     d3(t,:) = a3(t,:)-y_k; % 1x10
%     d2(t,:) = d3(t,:)*Theta2(:,2:end).*sigmoidG(z2(t,:)); %  1x25
%     
% end
% toc
% %set capital deltas
% D1 = d2'*a1; % 25x401
% D2 = d3'*a2; % 10x26
% 
% D_ = 1/m*[D1(:); D2(:)];




% % gradient checking
% grad_approx = zeros(size(D));
% EPSILON     = 0.0001;
% theta_un    = [init_theta1(:); init_theta2(:)];
% 
% costFunction = @(p)(nnCostFunction(p,X,y,lambda));
% 
% for t = 1:20
%    t
%    theta_plus     = theta_un;
%    theta_plus(t)  = theta_plus(t) + EPSILON;
%    theta_minus    = theta_un;
%    theta_minus(t) = theta_minus(t) - EPSILON;
%    
%    J_plus  = costFunction(theta_plus);
%    J_minus = costFunction(theta_minus);
%    
%    grad_approx(t) = (J_plus - J_minus)/(2 * EPSILON);
% end
% 
% [D(1:20,:) D_(1:20,:) grad_approx(1:20,:)]




