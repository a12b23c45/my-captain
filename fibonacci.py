def fibonacci_series(n):
    fib = [0, 1]
    while len(fib) < n:
        fib.append(fib[-1] +  fib[-2])
    return fib

#testing
n=10
print(fibonacci_series(n))