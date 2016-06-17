n = int(raw_input("please input a number n to fizzbuzz:"))



def fizzbuzz(n):
    for (i) in range(1,(n+1)):
        if (i%3 == 0 and i%5==0):
            print 'fizzbuzz'
        elif (i%3 == 0):
            print 'fizz'
        elif (i%5 == 0):
            print 'buzz'
        else:
            print i

fizzbuzz(n)