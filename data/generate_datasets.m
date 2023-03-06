% Set random seed
rng('default');
rng(1)

% Select discretization size and number of training and testing data
s = 421;
ntrain = 1000;
ntest = 200;

% Parameter of the covariance kernel
alpha = 2;
tau = 3;

% Coefficient
coef = ones(s,s);

% Create PDE operator
K = length(coef);
[X1,Y1] = meshgrid(1/(2*K):1/K:(2*K-1)/(2*K),1/(2*K):1/K:(2*K-1)/(2*K));
[X2,Y2] = meshgrid(0:1/(K-1):1,0:1/(K-1):1);
coef = interp2(X1,Y1,coef,X2,Y2,'spline');
d = cell(K-2,K-2);
[d{:}] = deal(sparse(zeros(K-2)));
for j=2:K-1
	d{j-1,j-1} = spdiags([[-(coef(2:K-2,j)+coef(3:K-1,j))/2;0],...
		(coef(1:K-2,j)+coef(2:K-1,j))/2 + (coef(3:K,j)+coef(2:K-1,j))/2 ...
		+ (coef(2:K-1,j-1)+coef(2:K-1,j))/2 + (coef(2:K-1,j+1)+coef(2:K-1,j))/2,...
		[0;-(coef(2:K-2,j)+coef(3:K-1,j))/2]],...
		-1:1,K-2,K-2);
	
	if j~=K-1
		d{j-1,j} = spdiags(-(coef(2:K-1,j)+coef(2:K-1,j+1))/2,0,K-2,K-2);
		d{j,j-1} = d{j-1,j};
	end
end
A = cell2mat(d)*(K-1)^2;

% Forcing array
F = zeros(ntrain+ntest, s, s);

% Generate random forcing functions
for i = 1:ntrain+ntest
    sprintf("Forcing terms: i = %d / %d", i, ntrain+ntest)
    
    f = GRF(alpha, tau, s);
    F(i,:,:) = f;
end

% Solve 2D laplacian PDE
U = solve_gwf(F, X1, Y1, X2, Y2, A);

% Save training data
force = F(1:ntrain,1:15:end,1:15:end);
sol = U(1:ntrain,1:15:end,1:15:end);
save('train.mat', 'force', 'sol')

% Save testing data
force = F(ntrain+1:end,1:15:end,1:15:end);
sol = U(ntrain+1:end,1:15:end,1:15:end);
save('test.mat', 'force', 'sol')
