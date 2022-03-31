from pylab import *
from casadi import *
from casadi.tools import *  # for dotdraw
#%matplotlib inline

x = SX.sym("x")  # scalar symbolic primitives
y = SX.sym("y")

z = x*sin(x+y)   # common mathematical operators


print(z)


J = jacobian(z,x)
print(J)

print(x*y/x-y)

H = hessian(z,x)
print(H)

f = Function("f",[x,y],[z])

print(f)

print(f(1.2,3.4))

print(f(1.2,x+y))

file = f.generate("f.c")
#print(file("f.c").read())


A = SX.sym("A",3,3)
B = SX.sym("B",3)
print(A)

print(solve(A,B))

print(trace(A))

print(mtimes(A,B))

print(norm_fro(A))

print(A[2,:])

print(A.shape, z.shape)

I = SX.eye(3)
print(I)

Ak = kron(I,A)
print(Ak)

Ak.sparsity().spy()

A.sparsity().spy()

z.sparsity().spy()


t = SX.sym("t")    # time
u = SX.sym("u")    # control
p = SX.sym("p")
q = SX.sym("q")
c = SX.sym("c")
x = vertcat(p,q,c) # state

ode = vertcat((1 - q**2)*p - q + u, p, p**2+q**2+u**2)
print(ode, ode.shape)

J = jacobian(ode,x)
print(J)


f = Function("f",[t,u,x],[ode])

ffwd = f.forward(1)

fadj = f.reverse(1)

# side-by-side printing
print('{:*^24} || {:*^28} || {:*^28}'.format("f","ffwd","fadj"))
def short(f):
    import re
    return re.sub(r", a\.k\.a\. \"(\w+)\"",r". \1",str(f).replace(", No description available","").replace("Input ","").replace("Output ",""))
for l in zip(short(f).split("\n"),short(ffwd).split("\n"),short(fadj).split("\n")):
    print('{:<24} || {:<28} || {:<28}'.format(*l))

print(I)

for i in range(3):
    print(ffwd(t,u,x, ode, 0,0,I[:,i]))

print(J)

for i in range(3):
    print(fadj(t,u,x, ode, I[:,i])[2])



f = {'x':x,'t':t,'p':u,'ode':ode}

tf = 10.0
N = 20
dt = tf/N


Phi = integrator("Phi","cvodes",f,{"tf":dt})
x0 = DM([0,1,0])
print(Phi(x0=x0))


x = x0
xs = [x]

for i in range(N):
    x = Phi(x0=x)["xf"]
    xs.append(x)


plot(horzcat(*xs).T)
legend(["p","q","c"])




n = 3

A = SX.sym("A",n,n)
B = SX.sym("B",n,n)
C = mtimes(A,B)
print(C)


A = MX.sym("A",n,n)
B = MX.sym("B",n,n)
C = mtimes(A,B)
print(C)


C = solve(A,B)
print(C)


X0 = MX.sym("x",3)

XF = Phi(x0=X0)["xf"]
print(XF)

expr = sin(XF)+X0


F = Function("F",[X0],[  expr  ])
print(F)

print(F(x0))


J = F.jacobian()

print(J(x0))

