function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = dataset3Params(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
% Minimum error 0.035000 with C = 0.333333 and sigma = 0.111111

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%
min_error = 100;
% cSigError = zeros(64, 3);
% idx = 0;
for i = -4:3
  for j = -4:3
    C_temp = 3^i; % Set values to train model
    sigma_temp = 3^j;
    model = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
    predictions = svmPredict(model, Xval); % Collect predictions with validation set
%   Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
    error = mean(double(predictions ~= yval));
%   visualizeBoundary(X, y, model);
%   pause;
    if error < min_error
      min_error = error;
      C = C_temp;
      sigma = sigma_temp;
%     fprintf('New minimum error %f with C = %f and sigma = %f', min_error, C, sigma);
    end
  end
end
%fprintf('Minimum error %f with C = %f and sigma = %f', min_error, C, sigma);


% =========================================================================

end
