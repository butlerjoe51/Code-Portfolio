println("Hello World!") #similar to Python's print("Hello World!")

#How to assign variables
g = 9.81
typeof(g)

age = 24
typeof(age)

days = 365
days_float = convert(AbstractFloat, days)
typeof(days_float)

print(days_float)

Lab = "I am a mad scientist!"
typeof(Lab)

no_error = """I can do this "without" any errors!"""

name = "Joseph"
age = 24
vehicle = "electric scooter"

println("Hello, my name is $name.")
println("I am $age years old and I use a/an $vehicle") #$ sign is the same as print("..my name is" + name) in Python

s1 = "This is a "
s2 = "combined string"

string(s1,s2)

s1*s2

hi = "hi"
hi_1000 = repeat(hi,1000) #Repeat Operator

hi_1000_alt = hi^1000 #expotentation method

a = 3
b = 4
c = string(a, "+", b) 
d = "7"

c + d

#While Loops
n = 0
while n < 10 #Whitespace like Python, but no : in statement
    n += 1
    println(n)
end
n

#For Loops 
for n in 1:10 #Equivalent to "for n in range(10):" in Python
    println(n)
end

#Loops and Arrays
m, n = 5, 5
A = fill(0, (m, n)) #Similar to np.zeros((5,5)) in Python

for j in 1:n
    for i in 1:m
        A[i, j] = i + j
    end
end
A

#Loops and Arrays - More syntatic sugar
B = fill(0, (m, n))

for j in 1:n, i in 1:m
    B[i, j] = i + j
end
B

#The Proper Julia way for this matrix - array comprehension
C = [i + j for i in 1:m, j in 1:n]

n = 0
squares = []
for n in 1:1000
    println(n^2)
end

squares_arr = [n^2 for n in 1:1000]

N = 5

#If statements
if (N % 3 == 0) && (N % 5 == 0) # `&&` means "AND"; % computes the remainder after division
    println("FizzBuzz")
elseif N % 3 == 0 #elseif = elif on Python
    println("Fizz")
elseif N % 5 == 0
    println("Buzz")
else
    println(N)
end #Reminder: In Julia, you have to tell the loop when it ends for syntax reasons
    

#Standard notation
function sayhi(name)
    println("Hi $name, it's great to see you!")
end

sayhi("Edward")

function f(x)
    x^3
end

f(7)

#Abridged notation
sayhi2(name) = println("Hi $name, it's great to see you!")
sayhi2("David")

f2(x) = x^3
f2(3)

#Alternate version - anonmyous functions
sayhi3 = name -> println("Hi $name, it's great to see you!")
sayhi3("Alex")

f3 = x -> x^3
f3(3)

sayhi(84048120) #Sayhi works on integers

A = rand(3, 3)
A

f3(A) #A works on matrices

using Plots

#Data to plot - Weather data
globaltemperatures = [14.4, 14.5, 14.8, 15.2, 15.5, 15.8]
numpirates = [45000, 20000, 15000, 5000, 400, 17];

gr() #Use gr backend

#Scatter plot
plot(numpirates, globaltemperatures, label="line")  
scatter!(numpirates, globaltemperatures, label="points") #Labels do the job that plt.legend() does on Python
xlabel!("Number of Pirates [Approximate]")
ylabel!("Global Temperature (C)")
title!("Influence of pirate population on global warming") 
#Because it is mutating, you can add to plot instead of repeating plt.show() like you do in Python
xflip!() #Flip for readability

#Redo with Unicode backend
unicodeplots()

plot(numpirates, globaltemperatures, label="line")  
scatter!(numpirates, globaltemperatures, label="points") 
xlabel!("Number of Pirates [Approximate]")
ylabel!("Global Temperature (C)")
title!("Influence of pirate population on global warming")

gr()

x = -10:10
x_squared = [x^2 for x in -10:10]
plot(x,x_squared, label="x^2")

p1 = plot(x, x)
p2 = plot(x, x.^2)
p3 = plot(x, x.^3)
p4 = plot(x, x.^4)
plot(p1, p2, p3, p4, layout = (2, 2), legend = false)

f(x) = x.^2 #x-square function
f(10)

f([1,2,3,4])

#We can tell Julia what type we want the inputs to be
foo(x::String, y::String) = println("My inputs x and y are both strings!")

foo("card","games")

#sum(a)

a = rand(10^7) # 1D vector of random numbers, uniform on [0,1

sum(a)

@time sum(a)

using PyCall #Allows us to run Python code on Julia

pysum = pybuiltin("sum")
pysum(a)

using BenchmarkTools  

py_list_bench = @benchmark pysum(a)

julia_bench = @benchmark sum(a)

using Conda
Conda.add("numpy") #Numpy on Julia

numpy_sum = pyimport("numpy")["sum"] #imports numpy.sum()
py_numpy_bench = @benchmark numpy_sum(a)

#Python hand-written

py"""
def py_sum(A):
    s = 0.0
    for a in A:
        s += a
    return s
"""

sum_py = py"py_sum"

py_hand = @benchmark sum_py(a)


