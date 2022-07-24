# Plate Bending Isometry
The work is for the large bending problem with isometry [1] and the bilayer plate bending problem with isometry [2].

* `bilayer.m` verifies the analytic solution for the bilayer problem.

## Finite element method
The finite element method is quite standard. The isometry constraint is imposed on the element nodes directly.

* `Specht` uses the standard Specht element with the 1-order convergence rate for 4-order biharmonic problem. 

* `Specht2d` uses the 2-order Specht element [3] with the 2-order convergence rate for 4-order biharmonic problem.

### Something left
* It is much easier to work on a small domain, i.e. for the bilayer problem in the domain $(0,1)^2$ with $\alpha=25$.

* At least for the bilayer problem, the analytic solution is known. We can calculate the relative error.

* We observe from the figures for the large bending problem and the numerical results for the bilayer problem that the isometry constraint does not maintain well.

* We have serious reservations about dependency for initial guess, especially for the bilayer problem. 

* Model reduction for both the physical model and numerical method, e.g. reduced basis method.

## Deep learning method
The motivation is that we observe that the finite element method can not maintain the isometry constraint well, and rely on the choice of initial guess. Deep learning method can deal with the isometry constraint in a more direct and effective method, and the initialization is random.

We implement the ResNet with Python 3.10.5 and PyTorch 1.12.0+cpu.

* `NNforPossion` solves the Poisson equation in the domain $(0,1)^2$. The boundary is imposed by the penalty term in the loss function.

* `NNforBending` solves the plate bending isometry in the domain $(0,1)^2$. We find that it is difficult to maintain the boundary constraint and the isometry constraint together. So we use the ansatz

	$$
		u(x,y)=x^2y^2f(x)+[x,y,0]^T,
	$$

	where $f$ is represented by a neural network, and $u$ satisfies the boundary constraint strictly.

* `NNforBilayer` solves the bilayer bending isometry in the domain $(-5,5)\times (-2,2)$. For convenience, we restore the domain information in the neural network. Similar to before, we use the ansatz

	$$
		u(x,y)=(x+5)^2f(x)+[x,y,0]^T.
	$$

	The experience shows that it needs 100w iteration. The function $[x,y,0]^T$ here may play a similar role as the initial guess in the finite element method.

* `NN2forBilayer` changes the ansatz to

	$$
		u(x,y)=(x+5)^2f(x)+g(x)
	$$

	where $f$ is a neural network approximating the interior, and $g$ is a neural network approximating the boundary. The experience shows that the boundary constraint should be included in the total loss, which is not needed in usual. This method trapped in local optimum after 80w iteration while it is flexible because we do not need to guess $g$.

### Something left
* The structure of the network, the choice of the loss function, and the penalty factor may influence the result.

* We observe that the procedure in `NNforBilayer` is similar to that in `Specht`, while that in `NN2forBilayer` is not, which may confirm the relationship between gradient flow and machine learning. 

* I do not know whether the methods for time-dependent problems may work for this problem and whether the methods with the domain decomposition may solve this problem for a large domain.

## References
[1] S. Bartels, *Finite element approximation of large bending isometries*, Numer. Math., **124** (2013), no. 3, 415-440.

[2] S. Bartels, and C. Palus, *Stable gradient flow discretizations for simulating bilayer plate bending with isometry and obstacle constraints*, IMA J. Numer. Anal., **42** (2022), no. 3, 1903-1928.

[3] H. L. Li, and P. B. Ming, and Z. C. Shi, *The Quadratic Specht Triangle*, J. Comput. Math., **38** (2020), no. 1, 103-124.