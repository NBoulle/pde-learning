% Solve the equation

function U = solve_gwf(force, X1, Y1, X2, Y2, A)
    K = size(force,2);
    n = size(force,1);
    F = zeros((K-2)^2,n);
    for i = 1:n
        f = interp2(X1,Y1,reshape(force(i,:,:),K,K),X2,Y2,'spline');
        f = f(2:K-1,2:K-1);
        F(:,i) = f(:);
    end
    X = A \ F;
    U = zeros(n, K, K);
    for i = 1:n
        P = [zeros(1,K);[zeros(K-2,1),reshape(X(:,i),K-2,K-2),zeros(K-2,1)];zeros(1,K)];
	    P = interp2(X2,Y2,P,X1,Y1,'spline')';
        U(i,:,:) = P;
    end
end
