

def title(show=''):
    def printStar(func):
        def f():
            print(show, '*************************************')
            return func()

        return f

    return printStar

@title('sub')
def sub():
    return 2 - 1


@title('add')
def add():
    return 1 + 1



print(add.__name__)
print(add())

print(sub())