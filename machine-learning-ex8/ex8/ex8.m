%clear;
clc;
load('ex8data1.mat');

plot(X(:, 1), X(:, 2), 'kx');

[m, n] = size(X);

mu = mean(X, 1);
sigma2 = 1/m * sum(bsxfun(@minus, X, mu).^2, 1);

x = linspace(min(X(:,1)), max(X(:,1)), 100);
y = linspace(min(X(:,2)), max(X(:,2)), 100);
[X_m Y_m] = meshgrid(0:.5:35);
% [X_m, Y_m] = meshgrid(x,y);

% univariate gaussian

% gauss_pdf = @(x) 1./(sqrt(2*pi*sigma2)).*exp(-((x - mu).^2)./(2*sigma2));
% z = zeros(numel(X_m),2);
% for i = 1:size(z,1)
%    z(i,:) =  gauss_pdf([X_m(i) Y_m(i)]);
% end
% %z = sum(z,2);
% hold on;
% z1 = reshape(z(:,1),size(X_m));
% contour(X_m,Y_m, z1);
% z2 = reshape(z(:,2),size(X_m));
% contour(X_m,Y_m, z2);

% multivariate gaussian

X_covar = bsxfun(@minus, X, mu);
covar = 1/m * (X_covar' * X_covar);
covar_det = det(covar);
covar_pinv = pinv(covar);
mgauss_pdf = @(x) 1 ./ ((2*pi) ^ (n/2) * covar_det ^ (1/2)) .* ...
     exp( - (1/2) * (x - mu) * covar_pinv * (x - mu)');
 
% 
% z = zeros(numel(X_m),1);
% for i = 1:size(z,1)
%     z(i) =  mgauss_pdf([X_m(i) Y_m(i)]);
% end
% z = reshape(z,size(X_m));
% hold on;
% contour(X_m, Y_m, z, 10.^(-20:3:0)');
% axis([0 30 0 30]);

% Selecting threshhold
[m_val, ~] = size(Xval); 
pval = zeros(m_val, 1);

for i = 1:m_val
    pval(i) = mgauss_pdf(Xval(i,:));
end

f1_y = yval;
stepsize = (max(pval) - min(pval)) / 1000;
F1 = 0;
epsilon = 0;
for epsilon_curr = min(pval):stepsize:max(pval)

   f1_h = (pval <= epsilon_curr);
    
   tp_sami = sum(f1_h & f1_y);
   fp_sami = sum(f1_h & ~f1_y);
   fn_sami = sum(~f1_h & f1_y);
   prec = tp_sami/(tp_sami+fp_sami);
   recall = tp_sami/(tp_sami+fn_sami);
   F1_curr = 2*prec*recall/(prec+recall);
   
   if F1_curr > F1
        F1 = F1_curr;
        epsilon = epsilon_curr;
   end
    
end

% high dimensional dataset
clear;
clc;
load('ex8data2.mat');

[m, n] = size(X);

mu = mean(X, 1);
sigma2 = 1/m * sum(bsxfun(@minus, X, mu).^2, 1);

% multivariate gaussian

X_covar = bsxfun(@minus, X, mu);
covar = 1/m * (X_covar' * X_covar);
covar_det = det(covar);
covar_pinv = pinv(covar);
mgauss_pdf = @(x) 1 ./ ((2*pi) ^ (n/2) * covar_det ^ (1/2)) .* ...
     exp( - (1/2) * (x - mu) * covar_pinv * (x - mu)');
 
% Selecting threshhold
[m_val, ~] = size(Xval); 
% pval = zeros(m_val, 1);
% 
% for i = 1:m_val
%     pval(i) = mgauss_pdf(Xval(i,:));
% end

pval = multivariateGaussian(Xval, mu, sigma2);

f1_y = yval;
stepsize = (max(pval) - min(pval)) / 1000;
F1 = 0;
epsilon = 0;
for epsilon_curr = min(pval):stepsize:max(pval)

   f1_h = (pval <= epsilon_curr);
    
   tp_sami = sum(f1_h & f1_y);
   fp_sami = sum(f1_h & ~f1_y);
   fn_sami = sum(~f1_h & f1_y);
   prec = tp_sami/(tp_sami+fp_sami);
   recall = tp_sami/(tp_sami+fn_sami);
   F1_curr = 2*prec*recall/(prec+recall);
   
   if F1_curr > F1
        F1 = F1_curr;
        epsilon = epsilon_curr;
   end
    
end

F1
epsilon

