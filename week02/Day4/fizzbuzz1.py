n = int(raw_input("Enter number: "))
def fizzbuzz(n):
    for x in range(1,n+1):
        if x % 15 == 0:
            print 'fizzbuzz'
        elif x % 3 == 0:
            print 'fizz'
        elif x % 5 == 0:
            print 'buzz'
        else:
            print x
fizzbuzz(n)
