function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
loop = length(C);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

predictions_error = zeros(loop*loop,1);
[C_m, sigma_m] = meshgrid(C, sigma); % loop x loop

for i = 1:numel(C_m)
       % train model
       model = svmTrain(X, y, C_m(i), @(x1, x2) gaussianKernel(x1, x2, sigma_m(i))); 
       % predict model
       predictions = svmPredict(model, Xval);
       predictions_error(i) = mean(double(predictions ~= yval));
end

[c, i] = min(predictions_error);
C = C_m(i);
sigma = sigma_m(i);


% =========================================================================

end
