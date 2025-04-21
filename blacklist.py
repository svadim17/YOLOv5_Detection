def sol(a, b):
    return a+b, a-b

print(type(sol(4, 6)))
c = [(1, 2)]
a, b = zip(*c)
print(a, b)