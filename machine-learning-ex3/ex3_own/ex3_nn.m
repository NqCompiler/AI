load('ex3weights.mat');
load('ex3data1.mat');

% forward propagation
X = [ones(length(X),1) X];
[m, n] = size(X);
% h = zeros(m,1);
tic
for i = 1:m
   a1 = X(i,:);
   z2 = Theta1*a1'; %25x401*401x1
   a2 = sigmoid(z2);
   a2 = [1; a2]; % 26x1
   z3 = Theta2*a2; %10x26*26x1
   h_temp = sigmoid(z3);
   [dummy h(i)] = max(h_temp);
end
h = h(:);

% prediction-rate
sum(h==y)/length(y)
toc
% fast forward vectorized
tic
a1 = X;
z2 = X*Theta1'; %5000x401*401x25;
a2 = sigmoid(z2); % 5000x25
a2 = [ones(m,1) a2]; % 5000x26
z3 = a2*Theta2'; %5000x10
[~, h] = max(z3,[],2);

% prediction-rate
sum(h==y)/length(y)
toc