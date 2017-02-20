import sys

# Gather our code in a main() function
def main():
    f = open("sample.txt",'w')
    print("This text goes in sample text file.",file=f)
    f.write("This line also goes into sample text file itself but as a new line.")
    f.close()

    f = open("sample.txt",'r')
    for line in f:
        print("Line:",line,end="")
    f.close()
    f = open("sample.txt",'r')
    lines = f.readlines()
    print(lines)
    f.close()
    f = open("sample.txt",'r')
    file_content = f.read()
    print(file_content)
    f.close()

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()

