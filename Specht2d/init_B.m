function B = init_B(u, Nv, Ne)
    B = zeros(3*Nv, 9*Nv+3*Ne);
    u1x = Nv+1: 2*Nv;
    u1y = 2*Nv+1: 3*Nv;
    u2x = 4*Nv+Ne+1: 5*Nv+Ne;
    u2y = 5*Nv+Ne+1: 6*Nv+Ne;
    u3x = 7*Nv+2*Ne+1: 8*Nv+2*Ne;
    u3y = 8*Nv+2*Ne+1: 9*Nv+2*Ne;
    B(1: Nv, [u1x, u2x, u3x]) = 2 * [diag(u(u1x)), diag(u(u2x)), diag(u(u3x))];
    B(Nv+1: 2*Nv, [u1x, u2x, u3x, u1y, u2y, u3y]) = [diag(u(u1y)), diag(u(u2y)), diag(u(u3y)), diag(u(u1x)), diag(u(u2x)), diag(u(u3x))];
    B(2*Nv+1: 3*Nv, [u1y, u2y, u3y]) = 2 * [diag(u(u1y)), diag(u(u2y)), diag(u(u3y))];
    B = sparse(B);
end