function [vertices, mesh] = init_mesh(n)
gm = [3; 4; 0; 4; 4; 0; 0; 0; 4; 4];
%gm = [3; 4; -2; 2; 2; -2; 0; 0; 1; 1];
model = createpde(1);
geometryFromEdges(model, decsg(gm));
mesh = generateMesh(model, 'Hmax', 1/n, 'GeometricOrder', 'linear');
Nv = size(mesh.Nodes, 2);
[p, ~, t] = meshToPet(mesh);

% p = zeros(2, (n+1)^2);
% t = zeros(3, n^2);
% idx = @(i, j) i*(n+1)+j+1;
% for i = 0: n
%     for j = 0: n
%         p(:, idx(i, j)) = [i/n, j/n];
%     end
% end
% for i = 1: n
%     for j = 1: n
%         t(:, 2*((i-1)*n+j)-1: 2*((i-1)*n+j)) = idx([i-1, i, i; i-1, i, i-1], [j-1, j-1, j; j-1, j, j])';
%     end
% end
% Nv = size(p, 2);

Nt = size(t, 2);
vertices = zeros(3*Nv, 2);
mesh = zeros(Nt, 9);
vertices(1: Nv, :) = p';
vertices(Nv+1: 2*Nv, :) = p';
vertices(2*Nv+1: 3*Nv, :) = p';
mesh(:, 1: 3) = t(1: 3, :)';
mesh(:, 4: 6) = t(1: 3, :)' + Nv;
mesh(:, 7: 9) = t(1: 3, :)' + 2 * Nv;
end