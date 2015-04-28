function [error_train, error_val, loop] = learningCurve(X, y_class, Xval, yval, theta_un, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
step_size = (0.1*m); % caution: works not if division is rests

% You need to return these values correctly
loop        = m/step_size;
error_train = zeros(loop, 1);
error_val   = zeros(loop, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
options = optimset('MaxIter', 100);
nn_Forward = @(t, X_train, y_class_train)(nnForward(t, X_train, y_class_train));
index_c = 0;

for i = 0:step_size:m
    if (i == 0) i = 1; end;
    index_c = index_c + 1;
    costFunction =  @(t)(nnCostFunction(t, X(1:i,:), y_class(1:i,:), lambda));
    [output theta] = evalc('fmincg(costFunction, theta_un, options)'); % evalc for surpressing output
    %compute train error
    error_train(index_c) = nn_Forward( theta, X(1:i,:), y_class(1:i,:) );
    %compute cv error
    error_val(index_c) = nn_Forward( theta, Xval, yval );
end

% -------------------------------------------------------------

% =========================================================================

end
