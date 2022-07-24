%The enery for bilayer problem
syms x y real
alpha = 2.5;
length = 5;
width = 2;
u = [sin((x + length)*alpha)/alpha-5, y, (1 - cos((x + length)*alpha))/alpha];
ux = diff(u, x);
uy = diff(u, y);
n = cross(ux, uy);
iso = simplify([ux*ux', ux*uy'; uy*ux', uy*uy']);
uxx = diff(ux, x);
uxy = diff(ux, y);
uyy = diff(uy, y);
H = [n*uxx', n*uxy'; n*uxy', n*uyy'];
loss = simplify(sum((H - alpha*eye(2)).^2, "all"))/2;
energy = loss * 4 * length * width;