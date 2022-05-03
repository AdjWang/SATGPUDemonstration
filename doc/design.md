## dev plan

1. python polygon generator
  
    Randomly generate polygon.

    Generator output format && SAT algorithm input format:

    ```
    # line1
    polygon_num
    # rest lines
    point_num
    point_list
    2
    4
    [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0),]
    5
    [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.5,1.5),(0.0,1.0),]
    ```

2. python polygon viewer

    Read and draw SAT result.

    SAT algorithm output format && viewer input format:

    ```
    # line1:
    polygon_num
    # rest lines:
    point_num
    point_list
    overlap_num
    overlap_index_list
    2
    4
    [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0),]
    1
    [1]
    5
    [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.5,1.5),(0.0,1.0),]
    1
    [0]
    ```

    viewer output format: png file.

3. C polygon loader

    Read SAT algorithm input to structures(see sat.c)

4. C version SAT algorithm

5. GPU version SAT algorithm

6. performance test
