function [vertices, mesh, Nv, Ne] = init_mesh(n)
gm = [3; 4; 0; 4; 4; 0; 0; 0; 4; 4];
model = createpde(1);
geometryFromEdges(model, decsg(gm));
mesh = generateMesh(model, 'Hmax', 1/n, 'GeometricOrder', 'linear');
Nv = size(mesh.Nodes, 2);
mesh = generateMesh(model, 'Hmax', 1/n);
[p, e, t] = meshToPet(mesh);
Ne = size(p, 2) - Nv;
Nt = size(t, 2);
vertices = zeros(3*Nv+Ne, 2);
mesh = zeros(Nt, 12);
vertices(1: Nv, :) = p(:, 1: Nv)';
mesh(:, 1: 3) = t(1: 3, :)';
vertices(Nv+1: 2*Nv, :) = p(:, 1: Nv)';
mesh(:, 4: 6) = t(1: 3, :)' + Nv;
vertices(2*Nv+1: 3*Nv+Ne, :) = p';
mesh(:, 7: 12) = t([1: 3, 5, 6, 4], :)' + 2 * Nv;
end