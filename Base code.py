import numpy as np
import matplotlib.pyplot as plt

#Physical parameters and mesh reolution
Pr=0.7
Ra=1e4

Lx=1
Ly=1
X=60
Y=60
dx=Lx/X
dy=Ly/Y
alpha=0.7
tolerance=1e-8

#Initializing velocity and pressure
u=np.zeros((X+2,Y+2))
v=np.zeros((X+2,Y+2))
theta=np.zeros((X+2,Y+2))
p=np.zeros((X+2,Y+2))

#Initialzing temperature 
x_centers = np.linspace(dx/2, Lx-dx/2, X)
for j in range(1, Y+1):
    for i in range(1, X+1):
        theta[i,j] = 1 - x_centers[i-1]

# Initialize arrays
ue=np.zeros_like(u)
uw=np.zeros_like(u)
vn=np.zeros_like(v)
vs=np.zeros_like(v)
aP=np.zeros_like(ue)
aE=np.zeros_like(ue)
aW=np.zeros_like(ue)
aN=np.zeros_like(ue)
aS=np.zeros_like(ue)
up=np.zeros_like(u)
vp=np.zeros_like(u)
ae=np.zeros((X+2,Y+2))
aw=np.zeros((X+2,Y+2))
an=np.zeros((X+2,Y+2))
asouth=np.zeros((X+2,Y+2))
bE=np.zeros_like(ae)
bW=np.zeros_like(ae)
bS=np.zeros_like(ae)
bN=np.zeros_like(ae)
bP=np.zeros_like(ae)
eE=np.zeros_like(ue)
eW=np.zeros_like(ue)
eN=np.zeros_like(ue)
eS=np.zeros_like(ue)
eP=np.zeros_like(ue)


def update_ghost(u,v,p,theta):
    # This functions calculates boundary conditions
    for j in range(1,Y+1):
        u[0,j]=-u[1,j]      # Left wall
        u[X+1,j]=-u[X,j]    # Right wall
        v[0,j]=-v[1,j]      # Left wall
        v[X+1,j]=-v[X,j]    # Right wall
        p[0,j]=p[1,j]       # Neumann for pressure
        p[X+1,j]=p[X,j]     # Neumann for pressure
        theta[0,j]=2-theta[1,j]    # Left wall: theta=1
        theta[X+1,j]=-theta[X,j]   # Right wall: theta=0

    for i in range(1,X+1):
        u[i,0]=-u[i,1]      # Bottom wall
        u[i,Y+1]=-u[i,Y]    # Top wall
        v[i,0]=-v[i,1]      # Bottom wall
        v[i,Y+1]=-v[i,Y]    # Top wall
        theta[i,0]=theta[i,1]      # Adiabatic bottom
        theta[i,Y+1]=theta[i,Y]    # Adiabatic top
        # Hydrostatic pressure correction
        p[i,0]=p[i,1]-dy*Ra*Pr*theta[i,1]
        p[i,Y+1]=p[i,Y]+dy*Ra*Pr*theta[i,Y]
    
    return(u,v,p,theta)

def velocity_at_boundary(u,v):
    # This function calculates velocities at each face between the cells
    for j in range(1,Y+1):
        for i in range(1,X+1):
            ue[i,j]=(u[i,j]+u[i+1,j])/2
            uw[i,j]=(u[i-1,j]+u[i,j])/2
            vn[i,j]=(v[i,j+1]+v[i,j])/2
            vs[i,j]=(v[i,j-1]+v[i,j])/2
    return(ue,uw,vn,vs)

def momentum_coefficients(ue,uw,vn,vs):
    # Momentum coefficients are calculated using the velocity at faces
    for j in range(Y+2):
        for i in range(X+2):
            aP[i,j]=((ue[i,j]-uw[i,j])/(2*dx)+(vn[i,j]-vs[i,j])/(2*dy)+2*Pr/dx**2+2*Pr/dy**2)
            aE[i,j]=ue[i,j]/(2*dx)-Pr/dx**2
            aW[i,j]=-uw[i,j]/(2*dx)-Pr/dx**2
            aN[i,j]=vn[i,j]/(2*dy)-Pr/dy**2
            aS[i,j]=-vs[i,j]/(2*dy)-Pr/dy**2 
    return(aP,aE,aW,aN,aS)

def Solve_momentum_equation(u,v,p,theta,aP,aW,aE,aN,aS):
    # This function uses one iteration to solve the momentum equations
    u_new = u.copy()
    v_new = v.copy()
    # Gauss-seidel method has been implemented
    for j in range(1,Y+1):
        for i in range(1,X+1):       
            u_new[i,j] = (1/aP[i,j])*(-(p[i+1,j]-p[i-1,j])/(2*dx) 
                                  - aN[i,j]*u_new[i,j+1] - aE[i,j]*u_new[i+1,j] 
                                  - aW[i,j]*u_new[i-1,j] - aS[i,j]*u_new[i,j-1])

            v_new[i,j] = (1/aP[i,j])*(-(p[i,j+1]-p[i,j-1])/(2*dy) + Ra*Pr*theta[i,j]
                                  - aN[i,j]*v_new[i,j+1] - aE[i,j]*v_new[i+1,j] 
                                  - aW[i,j]*v_new[i-1,j] - aS[i,j]*v_new[i,j-1])
            
    return (u_new, v_new)

def Residual_coefficients(aP):
    #In this function residual coefficients are caculated
    for j in range(1,Y+1):
        for i in range(1,X+1):
            ae[i,j]=2/(1/aP[i,j]+1/aP[i+1,j])
            aw[i,j]=2/(1/aP[i,j]+1/aP[i-1,j])
            an[i,j]=2/(1/aP[i,j]+1/aP[i,j+1])
            asouth[i,j]=2/(1/aP[i,j]+1/aP[i,j-1])
    return(ae,aw,an,asouth)

def Residual_computation(u,v,p,aP,ae,aw,an,asouth):
    """
    This function implements Rhie-cho interpolation to solve the decoupling of 
    pressure and velocity. it is only used for interior cells.
    """
    Residual=np.zeros_like(u)
    for j in range(1,Y+1):
        for i in range(1,X+1):
            if i==1:
                uw[i,j]=(u[i-1,j]+u[i,j])/2
            else:
                uw[i,j]=0.5*(u[i,j]+u[i-1,j]+(p[i+1,j]-p[i-1,j])/(2*dx*aP[i,j])+(p[i,j]-p[i-2,j])/(2*dx*aP[i-1,j]))-(p[i,j]-p[i-1,j])/(dx*aw[i,j])
            if j==1:
                vs[i,j]=(v[i,j-1]+v[i,j])/2
            else:
                vs[i,j]=0.5*(v[i,j]+v[i,j-1]+(p[i,j+1]-p[i,j-1])/(2*dy*aP[i,j])+(p[i,j]-p[i,j-2])/(2*dy*aP[i,j-1]))-(p[i,j]-p[i,j-1])/(dy*asouth[i,j])
            if i==X:
                ue[i,j]=(u[i,j]+u[i+1,j])/2
            else:
                ue[i,j]=0.5*(u[i,j]+u[i+1,j]+(p[i+1,j]-p[i-1,j])/(2*dx*aP[i,j])+(p[i+2,j]-p[i,j])/(2*dx*aP[i+1,j]))-(p[i+1,j]-p[i,j])/(dx*ae[i,j])
            if j==Y:
                vn[i,j]=(v[i,j+1]+v[i,j])/2
            else:
                vn[i,j]=0.5*(v[i,j]+v[i,j+1]+(p[i,j+1]-p[i,j-1])/(2*dy*aP[i,j])+(p[i,j+2]-p[i,j])/(2*dy*aP[i,j+1]))-(p[i,j+1]-p[i,j])/(dy*an[i,j])
    for j in range(1,Y+1):
        for i in range(1,X+1):        
            Residual[i,j]=-(ue[i,j]-uw[i,j])/dx-(vn[i,j]-vs[i,j])/dy
    # Compute L2 norm safely
    residual_internal = Residual[1:X+1,1:Y+1]
    L2_norm = np.sqrt(np.sum(residual_internal**2))/(X*Y)
    return(Residual,L2_norm)
        
def pressure_coefficient(ae,aw,an,asouth):
    # This function calculates the pressure coefficients
    for j in range(1,Y+1):
        for i in range(1,X+1):
            bE[i,j]=-1/(ae[i,j]*np.power(dx,2))
            bW[i,j]=-1/(aw[i,j]*np.power(dx,2))
            bN[i,j]=-1/(an[i,j]*np.power(dy,2))
            bS[i,j]=-1/(asouth[i,j]*np.power(dy,2))
    
    # Boundary conditions
    for i in range(1,X+1):
        bN[i,Y] = 0
        bS[i,1] = 0
    for j in range(1,Y+1):
        bE[X,j] = 0
        bW[1,j] = 0
    for j in range(1,Y+1):
        for i in range(1,X+1):
            bP[i,j]=-(bE[i,j]+bW[i,j]+bN[i,j]+bS[i,j])
    return(bE,bW,bN,bS,bP)
    
def solve_pressure_equation(bP,bE,bW,bN,bS,Residual):
    #This function solves the pressure correction equation using gause-seidel 
    # An inner-iter was selected for the number of iterations used 
    inner_iter = 30
    p_prime_new=np.zeros_like(bP)
    for key in range(inner_iter):
        p_prime_temp = p_prime_new.copy()
        
        for j in range(1,Y+1):
            for i in range(1,X+1):
                    p_prime_new[i,j] = (-bE[i,j]*p_prime_temp[i+1,j] 
                                       - bW[i,j]*p_prime_new[i-1,j] 
                                       - bN[i,j]*p_prime_temp[i,j+1] 
                                       - bS[i,j]*p_prime_new[i,j-1] 
                                       + Residual[i,j])/bP[i,j]
    #boundary condition for p_prime 
    for j in range(1,Y+1):
        p_prime_new[0,j] = p_prime_new[1,j]
        p_prime_new[X+1,j] = p_prime_new[X,j]
    for i in range(1,X+1):
        p_prime_new[i,0] = p_prime_new[i,1]
        p_prime_new[i,Y+1] = p_prime_new[i,Y]

    return p_prime_new
    
def update_variables(u,v,p,p_prime,aP):
    # This function updates the variables using pressure-corection eqa
    for j in range(1,Y+1):
        for i in range(1,X+1):
            u[i,j] = u[i,j] - (p_prime[i+1,j]-p_prime[i-1,j])/(2*aP[i,j]*dx)
            v[i,j] = v[i,j] - (p_prime[i,j+1]-p_prime[i,j-1])/(2*dy*aP[i,j])
            
            p[i,j] += alpha*p_prime[i,j]
    
    # Reference pressure is selected at the bottom left corner
    p_ref = p[1,1]
    for j in range(1,Y+1):
        for i in range(1,X+1):
            p[i,j] = p[i,j] - p_ref
    
    return(u,v,p)

def energy_coefficients(ue,uw,vn,vs):
    # This function calculates the energy coefficients
    for j in range(1,Y+1):
        for i in range(1,X+1):
            eE[i,j] = ue[i,j]/(2*dx) - 1/dx**2
            eW[i,j] = -uw[i,j]/(2*dx) - 1/dx**2
            eN[i,j] = vn[i,j]/(2*dy) - 1/dy**2
            eS[i,j] = -vs[i,j]/(2*dy) - 1/dy**2
            eP[i,j] = (ue[i,j]-uw[i,j])/(2*dx) + (vn[i,j]-vs[i,j])/(2*dy) + 2/dx**2 + 2/dy**2   
    return(eE,eW,eN,eS,eP)

def compute_energy_equation(theta,eE,eW,eN,eS,eP):
    #Gauss-seidel method has been implemented in one iteration to solve energy eqa
    theta_new = theta.copy()
    for j in range(1,Y+1):
        for i in range(1,X+1):
            theta_temp = (-eE[i,j]*theta_new[i+1,j] - eN[i,j]*theta_new[i,j+1] 
                        - eW[i,j]*theta_new[i-1,j] - eS[i,j]*theta_new[i,j-1])/eP[i,j]

            theta_new[i,j] = theta_temp 
    
    return(theta_new)

# Main iteration loop
iter = 0
max_iterations = 50000

print("Starting simulation...")
while iter < max_iterations:
    u,v,p,theta=update_ghost(u,v,p,theta)
    ue,uw,vn,vs=velocity_at_boundary(u,v)
    aP,aE,aW,aN,aS=momentum_coefficients(ue,uw,vn,vs)
    ae,aw,an,asouth=Residual_coefficients(aP)
    u,v=Solve_momentum_equation(u,v,p,theta,aP,aW,aE,aN,aS)
    ue,uw,vn,vs=velocity_at_boundary(u,v)
    aP,aE,aW,aN,aS=momentum_coefficients(ue,uw,vn,vs)
    ae,aw,an,asouth=Residual_coefficients(aP)
    Residual,L2_norm=Residual_computation(u,v,p,aP,ae,aw,an,asouth)
    bE,bW,bN,bS,bP=pressure_coefficient(ae,aw,an,asouth)
    p_prime=solve_pressure_equation(bP,bE,bW,bN,bS,Residual)
    u,v,p_new=update_variables(u,v,p,p_prime,aP)
    ue,uw,vn,vs=velocity_at_boundary(u,v)
    eE,eW,eN,eS,eP=energy_coefficients(ue,uw,vn,vs)
    theta=compute_energy_equation(theta,eE,eW,eN,eS,eP)
    iter += 1
    
    if iter % 100 == 0:
        print(f"Iteration {iter}, L2_norm: {L2_norm:.2e}")
        print(f"  Max u: {np.max(np.abs(u[1:X+1,1:Y+1])):.2e}")
        print(f"  Max v: {np.max(np.abs(v[1:X+1,1:Y+1])):.2e}")
        print(f"  theta: {(theta[15,1]+theta[16,1])/2:.2e}")
    
    if L2_norm < tolerance:
        print(f"Converged in {iter} iterations, L2_norm: {L2_norm:.2e}")
        break
    
    # Check for divergence
    if np.isnan(L2_norm) or np.isinf(L2_norm) or L2_norm > 1e10:
        print(f"Simulation diverged at iteration {iter}")
        break

# Final update and results
u,v,p,theta = update_ghost(u,v,p,theta)
#print(f"Temperature at the bottom wall is: {(theta[30,1]+theta[31,1])/2}")
print("Simulation complete.")

# Plotting the temperature contours
fig, ax = plt.subplots(figsize=(6, 12))
x_coords = np.linspace(0, Lx, X)
y_coords = np.linspace(0, Ly, Y)
X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
contour_levels = 20

# Get the internal part of theta for plotting
theta_internal = theta[1:X + 1, 1:Y + 1].T

contour = ax.contourf(X_grid, Y_grid, theta_internal, levels=contour_levels, cmap='coolwarm')
fig.colorbar(contour, ax=ax, label='Temperature')
ax.set_title(f'Temperature Contours ($Ra={Ra:.0f}$, Aspect Ratio 1x2)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()