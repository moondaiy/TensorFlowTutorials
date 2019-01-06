

def create():

    pos = [0,0]
    x_g = 0
    y_g = 0

    def run(x,y):

        # nonlocal pos

        pos[0] = x + pos[0]
        pos[1] = y + pos[1]

        return pos

    return run


people = create()

print(people(1,1))
print(people(1,1))
print(people(1,1))