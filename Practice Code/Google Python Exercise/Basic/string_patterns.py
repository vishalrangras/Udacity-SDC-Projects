def main():
    n = int(input())
    a = list(range(n))
    b = list(range(n))
    b.reverse()

    print("Pattern 1")
    for x in a:
        for y in b:
            if y<=x:
                print ("*",end="")
        print()
    print()
    
    print("Pattern 2")
    for x in a:
        for y in b:
            if x>=y:
                print("*",end="")
            else:
                print(" ",end="")
        print()
    print()

    print("Pattern 3")
    for x in a:
        for y in b:
            if x>y:
                print(" ",end="")
            else:
                print("*",end="")
        print()

    print("Pattern 4")
    for x in a:
        for y in a:
            if y>=x:
                print("*",end="")
            else:
                print(" ",end="")
        print()
    print()

    print("Pattern 5")
    k = 2*n - 2
    for i in range(0,n):
        for j in range(0,k):
            print(end=" ")

        k = k -1

        for j in range(0,i+1):
            print("* ",end="")
        print("\r")
        
    

if __name__ == '__main__':
    main()
