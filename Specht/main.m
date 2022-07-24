clear;
f = [0; 0; 2.5e-2];
red = 2;
dt = 2^(-red);
[vertices, mesh] = init_mesh(2^red);
[phi, phix, phiy, phixx, phixy, phiyy, T] = init_fespace;
Nv = size(vertices, 1) / 3;
u = zeros(12*Nv, 1);
% for i = 1: Nv
%     if vertices(i, 1) < -1.4
%         u(i) = vertices(i, 1) + 1.4;
%     elseif vertices(i, 1) < 0
%         u(6*Nv+i) = vertices(i, 1) + 1.4;
%     elseif vertices(i, 1) < 1.4
%         u(6*Nv+i) = -vertices(i, 1) + 1.4;
%     else
%         u(i) = vertices(i, 1) - 1.4;       
%     end
%     u(3*Nv+i) = vertices(i, 2); 
% end
u(1: Nv) = vertices(1: Nv, 1);
u(3*Nv+1: 4*Nv) = vertices(1: Nv, 2);
u(Nv+1: 2*Nv) = 1;
u(5*Nv+1: 6*Nv) = 1;
[S, F] = FEM(f, vertices, mesh, phi, phixx, phixy, phiyy, T);
corr = 1;
not_bdr = prod(vertices, 2);%4 - vertices(:, 1).^2;
free = [not_bdr ~= 0; not_bdr ~= 0; not_bdr ~= 0; not_bdr ~= 0];
c = zeros(12*Nv, 1);
trisurf(mesh(:, 1: 3), u(1: Nv), u(3*Nv+1: 4*Nv), u(6*Nv+1: 7*Nv));
step = 0;
while corr > 1e-6
   step = step + 1;
   [B, T, L] = init_B(u); 
   A = [S, B'; B, zeros(3*Nv)];
   b = [F - S * u(1: 9*Nv); zeros(3*Nv, 1)];% - L;
   c(free) = A(free, free) \ b(free);
   du = c(1: 9*Nv);
   u = u + dt * c;
   corr = du' * S * du;
   energy = 1/2 * u(1: 9*Nv)' * S * u(1: 9*Nv) - F' * u(1: 9*Nv);
   trisurf(mesh(:, 1: 3), u(1: Nv), u(3*Nv+1: 4*Nv), u(6*Nv+1: 7*Nv));
   u1x = u(Nv+1: 2*Nv);
   u1y = u(2*Nv+1: 3*Nv);
   u2x = u(4*Nv+1: 5*Nv);
   u2y = u(5*Nv+1: 6*Nv);
   u3x = u(7*Nv+1: 8*Nv);
   u3y = u(8*Nv+1: 9*Nv);
   error = sum(abs([u1x.^2+u2x.^2+u3x.^2-1; u1x.*u1y+u2x.*u2y+u3x.*u3y; u1y.^2+u2y.^2+u3y.^2-1]));
end



