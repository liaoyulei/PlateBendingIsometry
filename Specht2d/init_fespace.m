function  [phi, phix, phiy, phixx, phixy, phiyy, T] = init_fespace
syms lambda1 lambda2 lambda3 x1 x2 x3 y1 y2 y3 sgn1 sgn2 sgn3
xi1 = x2 - x3;
xi2 = x3 - x1;
xi3 = x1 - x2;
eta1 = y2 - y3;
eta2 = y3 - y1;
eta3 = y1 - y2;
inner11 = eta1^2 + xi1^2;
inner12 = eta1 * eta2 + xi1 * xi2;
inner13 = eta1 * eta3 + xi1 * xi3;
inner22 = eta2^2 + xi2^2;
inner23 = eta2 * eta3 + xi2 * xi3;
inner33 = eta3^2 + xi3^2;
T2 = det([1, x1, y1; 1, x2, y2; 1, x3, y3]);
bK = lambda1 * lambda2 * lambda3;
phi7 =  bK * ((-24/3 + 10)*(5*(lambda1*lambda2 + lambda2*lambda3 + lambda3*lambda1) - 1) - 30 * lambda2 * lambda3);
phi8 =  bK * ((-24/3 + 10)*(5*(lambda1*lambda2 + lambda2*lambda3 + lambda3*lambda1) - 1) - 30 * lambda1 * lambda3);
phi9 =  bK * ((-24/3 + 10)*(5*(lambda1*lambda2 + lambda2*lambda3 + lambda3*lambda1) - 1) - 30 * lambda1 * lambda2);
phi = [
    lambda1 * lambda1 * (3-2*lambda1) +  inner12/inner22*phi8 + inner13/inner33*phi9;
    lambda2 * lambda2 * (3-2*lambda2) + inner12/inner11*phi7 + inner23/inner33*phi9;
    lambda3 * lambda3 * (3-2*lambda3) + inner13/inner11*phi7 + inner23/inner22*phi8;
    lambda1 * lambda1 * (-xi3*lambda2+xi2*lambda3) + xi1 / 3 * (phi8 - phi9);
    lambda2 * lambda2 * (-xi1*lambda3+xi3*lambda1) + xi2 / 3 * (phi9 - phi7);
    lambda3 * lambda3 * (-xi2*lambda1+xi1*lambda2) + xi3 / 3 * (phi7 - phi8);
    lambda1 * lambda1 * (-eta3*lambda2+eta2*lambda3) + eta1 / 3 * (phi8 - phi9);
    lambda2 * lambda2 * (-eta1*lambda3+eta3*lambda1) + eta2 / 3 * (phi9 - phi7);
    lambda3 * lambda3 * (-eta2*lambda1+eta1*lambda2) + eta3 / 3 * (phi7 - phi8);
    sgn1 * phi7 * T2 / sqrt(xi1^2 + eta1^2);
    sgn2 * phi8 * T2 / sqrt(xi2^2 + eta2^2);
    sgn3 * phi9 * T2 / sqrt(xi3^2 + eta3^2);
];
lambda1x = eta1 / T2;
lambda2x = eta2 / T2;
lambda3x = eta3 / T2;
lambda1y = -xi1 / T2;
lambda2y = -xi2 / T2;
lambda3y = -xi3 / T2;
phix = lambda1x * diff(phi, lambda1) + lambda2x * diff(phi, lambda2) + lambda3x * diff(phi, lambda3);
phiy = lambda1y * diff(phi, lambda1) + lambda2y * diff(phi, lambda2) + lambda3y * diff(phi, lambda3);
phixx = lambda1x * diff(phix, lambda1) + lambda2x * diff(phix, lambda2) + lambda3x * diff(phix, lambda3);
phixy = lambda1y * diff(phix, lambda1) + lambda2y * diff(phix, lambda2) + lambda3y * diff(phix, lambda3);
phiyy = lambda1y * diff(phiy, lambda1) + lambda2y * diff(phiy, lambda2) + lambda3y * diff(phiy, lambda3);
phi = matlabFunction(phi);
phix = matlabFunction(phix);
phiy = matlabFunction(phiy);
phixx = matlabFunction(phixx);
phixy = matlabFunction(phixy);
phiyy = matlabFunction(phiyy);
T = matlabFunction(T2 / 2);
end