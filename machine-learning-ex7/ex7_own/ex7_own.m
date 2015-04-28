clear;
load('ex7data2.mat');
rand('seed',10);
[m, n] = size(X);
K = 10;

% normalize
mu = mean(X, 1);
sd = std(X);
X = bsxfun(@minus, X, mu);
X = bsxfun(@rdivide, X, sd);

outer_loop = 100;
J_hist         = zeros(K, outer_loop);
centroids_hist = zeros(outer_loop, n, K);
tic
% run for multiple K's
for k = 1:K
    centroids = zeros(k, n);
    % try different initial centroids
    for i = 1:outer_loop
        perm = randperm(m);
        centroids(1:k,:) = X(perm(1:k),:);
        counter = 0;
        % run as long as nothing changes
        while true
          counter = counter + 1;
          % find closest centroids
          c = fcc(X, centroids);
          % asign new centroids
          current_centroids = anc(X, c, k);
          % break while loop when no changes
          if (all(centroids == current_centroids)) centroids = current_centroids; break; end
          % break if while loop get stuck
          if (counter >= 1000) centroids = current_centroids;break; end
          centroids = current_centroids;
          % save centroids hist
          centroids_hist(i, :, k) = centroids;
        end
        J_hist(k,i) = cost(X,centroids,c);
    end
end
toc

clf;
J_mins = min(J_hist, [], 2);
plot([1:length(J_mins)],J_mins, 'ro-');

% create data set k=3 clustered with centroids
colors = ['rx', 'bx', 'kx'];
k = 3;
inner_loop = J_mins(k);
centroids = centroids_hist(inner_loop, :, k);
c = anc(X, centroids);
figure;
hold on;
for i = 1:k
    data_clustered = find(c==i);
    plot(X(data_clustered,1), X(data_clustered,2), colors(i))
    hold on;
end

plot(centroids(:, 1), centroids(:, 2), 'go');

