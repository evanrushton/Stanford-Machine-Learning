function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
distance = zeros(size(X,1), K);
for k =1:K
  distance(:,k) = sum(bsxfun(@minus, X, centroids(k,:)).^2, 2);
end
[dis, idx] = min(distance, [], 2);

%for i = 1:size(X,1)
%  min_dis = [100, -1];
%  for k = 1:K
%    dis = norm(X(i) - centroids(k,:)).^2;
%    if min_dis(1) > dis
%      min_dis = [dis, k];
%    end
%  end
%  idx(i, 1) = min_dis(2);
%end


% Couldn't figure a vectorized norm
% [dis(i,1), idx(i,1)] = min( norm( X(i) - centroids ) );
% =============================================================

end

