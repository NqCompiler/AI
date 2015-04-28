clear;
clc;
load('ex8_movieParams.mat');
load('ex8_movies.mat');

% data

num_movies   = num_movies;
num_users    = num_users;
max_features = 100;
Y            = Y;           % 1682x943
[m, n]       = size(Y);
m_r          = sum(sum(R));

% generate Data

X = randinit( m, max_features, 1.5 );
Theta = randinit ( n, max_features, 2.5 );

% run collaborative filtering algorithm

start_features = 10;
step_size      = 10;
lambda         = 0.5;
options        = optimset('GradObj','on','MaxIter', 100);
J_hist         = zeros(1, 1);
dev_hist       = zeros(1, 1);

for i = start_features:step_size:max_features
    X_curr = X(:, 1:i);
    Theta_curr = Theta(:, 1:i);
    initial_parameters = [X_curr(:); Theta_curr(:)];   
    num_features = i;
    cost_function = @(t)(CostGradReco(t, Y, R, lambda, num_features));
    [theta J_curr] = fmincg(cost_function, initial_parameters, options);
    
    % deviation
    X_curr      = reshape(theta(1: m*num_features), m, num_features);
    Theta_curr  = reshape(theta(m*num_features+1: end), n, num_features);
    h           = X_curr*Theta_curr';
    dev         = 1/m_r*sum(sum((h-Y).*R));
    
    dev_hist(end+1) = abs(dev); 
    
    % save J_hist for i
    J_hist(end+1) = J_curr(length(J_curr));
end

plotyy(linspace(start_features, max_features, step_size), J_hist(2:end),linspace(start_features, max_features, step_size),dev_hist(2:end))
ylabel('J');
xlabel('# of features');

% initial_parameters = [X(:); Theta(:)];   
% [m,n] = size(Y);
% cost_function = @(t)(CostGradReco(t, Y, R, lambda, num_features));
% [theta J_hist] = fmincg(cost_function, initial_parameters, options);
% X      = reshape(theta(1: m*num_features), m, num_features);
% Theta  = reshape(theta(m*num_features+1: end), n, num_features); 

%plot(J_hist);

% for i = 1:10    
%     theta = fmincg(@(t)(CostGradReco(t, Y, R, lambda, num_features)), initial_parameters, options);
%     X_curr      = reshape(theta(1: m*num_features), m, num_features);
%     Theta_curr  = reshape(theta(m*num_features+1: end), n, num_features);
%     initial_parameters = [X_curr(:); Theta_curr(:)];
%     J_hist(i) = CostGradReco( initial_parameters, Y, R, lambda, num_features );
% end


