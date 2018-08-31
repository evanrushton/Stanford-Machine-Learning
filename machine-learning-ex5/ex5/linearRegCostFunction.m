function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
J = (1 / (2*m)) * sum((X * theta - y) .* (X * theta - y));

for j = 1:n
  grad(j,1) = sum((X * theta - y) .* X(:,j));
end
grad = (1/m) * grad;

% Regularization
theta(1) = 0;
J_reg = (lambda / (2*m)) * sum(theta.^2);
grad_reg = (lambda / m) * theta;

J = J + J_reg;
grad = grad + grad_reg;






% =========================================================================

grad = grad(:);

end
