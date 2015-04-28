clear;
clf;
% load('ex7data1.mat');
% [m, n] = size(X);
% 
% % mean normalization
% X = bsxfun(@minus, X, mean(X,1));
% X = bsxfun(@rdivide, X, std(X,1));
% 
% % plot data
% plot(X(:,1), X(:,2), 'ko');
% 
% % PCA
% Sigma = 1/m*X'*X;
% [U, S, V] = svd(Sigma);
% k = 1;
% U_reduced = U(: ,1:k);        % n x k;
% e = eye(n);
% S_cum_sum = cumsum(sum(eye(n).*S,1));
% retained_var = trace(S(1:k,1:k))/trace(S)
% % retained_var = ( S_cum_sum(k)/S_cum_sum(n) );
%  
% X_reduced = X*U_reduced;    % m x k (m x n * n x k);
% X_reconst = X_reduced*U_reduced'; % m x n
% 
% hold on;
% plot( X_reconst(:, 1), X_reconst(:, 2), 'ro')
% 
% drawLine = @(p1,p2) plot([p1(1) p2(1)], [p1(2) p2(2)],'--k', 'LineWidth', 1);
% for i = 1:size(X, 1)
%     drawLine(X(i,:), X_reconst(i,:));   
% end
% axis([-2.5 2.5 -2.5 2.5]);

load('ex7faces.mat');
displayData(X(1:100, :));

[m, n] = size(X);
X = bsxfun(@minus, X, mean(X,1));
X = bsxfun(@rdivide, X, std(X,1));

Sigma = 1/m*X'*X;
[U, S, V] = svd(Sigma);
k = 333;

U_reduced = U(: ,1:k);        % n x k;
e = eye(n);
retained_var = trace(S(1:k,1:k))/trace(S)
% S_cum_sum = cumsum(sum(eye(n).*S,1));
% retained_var = ( S_cum_sum(k)/S_cum_sum(n) );

X_reduced = X*U_reduced;    % m x k (m x n * n x 1);
X_reconst = X_reduced*U_reduced'; % m x n

figure;
displayData(X_reconst(1:100, :));



