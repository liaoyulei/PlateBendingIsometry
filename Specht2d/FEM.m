function [S, F] = FEM(f, vertices, mesh, phi, phixx, phixy, phiyy, T)
wg = [0.050844906370207; 0.050844906370207; 0.050844906370207; 0.116786275726379; 0.116786275726379; 0.116786275726379; 0.082851075618374; 0.082851075618374; 0.082851075618374; 0.082851075618374; 0.082851075618374; 0.082851075618374];
ld = [
    0.873821971016996, 0.063089014491502, 0.063089014491502;
    0.063089014491502, 0.873821971016996, 0.063089014491502;
    0.063089014491502, 0.063089014491502, 0.873821971016996;
    0.501426509658179, 0.249286745170910, 0.249286745170910;
    0.249286745170910, 0.501426509658179, 0.249286745170910;
    0.249286745170910, 0.249286745170910, 0.501426509658179;
    0.636502499121399, 0.310352451033785, 0.053145049844816;
    0.636502499121399, 0.053145049844816, 0.310352451033785;
    0.310352451033785, 0.636502499121399, 0.053145049844816
    0.310352451033785, 0.053145049844816, 0.636502499121399;
    0.053145049844816, 0.636502499121399, 0.310352451033785
    0.053145049844816, 0.310352451033785, 0.636502499121399;
];
Nv = size(vertices, 1);
Nt = size(mesh, 1);
a = @(uxx, uxy, uyy, vxx, vxy, vyy) uxx * vxx + 2 * uxy * vxy + uyy * vyy;
S = zeros(Nv);
for k = 1: Nt
    K = zeros(size(mesh, 2));
    v1 = vertices(mesh(k, 1), :);
    v2 = vertices(mesh(k, 2), :);
    v3 = vertices(mesh(k, 3), :);       
    sgn1 = sign(mesh(k, 3) - mesh(k, 2));
    sgn2 = sign(mesh(k, 1) - mesh(k, 3));
    sgn3 = sign(mesh(k, 2) - mesh(k, 1));
    jacobi = T(v1(1), v2(1), v3(1), v1(2), v2(2), v3(2));
    for l = 1: size(wg, 1)
        vxx = phixx(ld(l, 1), ld(l, 2), ld(l, 3), sgn1, sgn2, sgn3, v1(1), v2(1), v3(1), v1(2), v2(2), v3(2));
        vxy = phixy(ld(l, 1), ld(l, 2), ld(l, 3), sgn1, sgn2, sgn3, v1(1), v2(1), v3(1), v1(2), v2(2), v3(2));
        vyy = phiyy(ld(l, 1), ld(l, 2), ld(l, 3), sgn1, sgn2, sgn3, v1(1), v2(1), v3(1), v1(2), v2(2), v3(2));
        for j = 1: size(mesh, 2)
            for i = 1: size(mesh, 2)
                K(i, j) = K(i, j) + wg(l) * jacobi * a(vxx(j), vxy(j), vyy(j), vxx(i), vxy(i), vyy(i));
            end
        end
    end
    S(mesh(k, :), mesh(k, :)) = S(mesh(k, :), mesh(k, :)) + K;
end
S = sparse([S, zeros(Nv), zeros(Nv); zeros(Nv), S, zeros(Nv); zeros(Nv), zeros(Nv), S]);
F = zeros(Nv, 1);
for k = 1: Nt
    v1 = vertices(mesh(k, 1), :);
    v2 = vertices(mesh(k, 2), :);
    v3 = vertices(mesh(k, 3), :);
    sgn1 = sign(mesh(k, 3) - mesh(k, 2));
    sgn2 = sign(mesh(k, 1) - mesh(k, 3));
    sgn3 = sign(mesh(k, 2) - mesh(k, 1));
    jacobi = T(v1(1), v2(1), v3(1), v1(2), v2(2), v3(2));
    for l = 1: size(wg, 1)
        v = phi(ld(l, 1), ld(l, 2), ld(l, 3), sgn1, sgn2, sgn3, v1(1), v2(1), v3(1), v1(2), v2(2), v3(2));
        F(mesh(k, :)) = F(mesh(k, :)) + wg(l) * jacobi * v * f(3);
    end
end
F = [zeros(2*Nv, 1); F];
end