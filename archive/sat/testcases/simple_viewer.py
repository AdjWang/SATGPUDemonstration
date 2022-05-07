import sys
from matplotlib import pyplot as plt

def get_xy(vertices):
    return list(zip(*vertices))

if __name__ == '__main__':
    input_file = sys.argv[1]

    vertex_list = []
    with open(input_file, 'r') as f:
        # total number of polygons
        n = int(f.readline())
        print(n)
        for i in range(n):
            # number of vertices
            n_vertex = int(f.readline())
            # list of vertices
            vertices = eval(f.readline())
            assert n_vertex == len(vertices), \
                f"vertices length not match, please check ouput of polygon {i+1} (count from 1)"
            
            vertex_list.append(vertices)

    print(vertex_list[0])
    plt.plot(*get_xy(vertex_list[0]), 'o')
    plt.plot(*get_xy(vertex_list[1]), '+')
    plt.show()