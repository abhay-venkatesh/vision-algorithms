function str = invhomog(H, K)
H = inv(K) * H * K;
[U,S,V] = svd(H); % Use power method since it's just 3 by 3..
% We assume that S(2,2) is 1, so we need to normalize.
if S(2,2) ~= 1,
    H = H / S(2,2);
    [U,S,V] = svd(H);
end
if det(V) < 0,
    V = -V; % Useful to check for the physically feasible solutions later.
end
s1 = S(1,1)^2;
s3 = S(3,3)^2;
v1 = V(:,1); v2 = V(:,2); v3 = V(:,3);
u1 = (sqrt(1-s3)*v1 + sqrt(s1-1)*v3) / sqrt(s1-s3);
u2 = (sqrt(1-s3)*v1 - sqrt(s1-1)*v3) / sqrt(s1-s3);
U1 = [v2 u1 cross(v2,u1)];
W1 = [H*v2 H*u1 cross(H*v2, H*u1)];
U2 = [v2 u2 cross(v2,u2)];
W2 = [H*v2 H*u2 cross(H*v2, H*u2)];
R1 = W1*U1';
R2 = W2*U2';
% Pick the two feasible planes
n = cross(v2, u1);
if n(3) > 0,
    sol(1).n = n;
    t = (H-R1)*n;
else
    sol(1).n = -n;
    t = -(H-R1)*n;
end
sol(1).T = [R1 t; 0 0 0 1];
n = cross(v2, u2);
if n(3) > 0,
    sol(2).n = n;
    t = (H-R2)*n;
else
    sol(2).n = -n;
    t = -(H-R2)*n;
end
sol(2).T = [R2 t; 0 0 0 1];
str = sol;
