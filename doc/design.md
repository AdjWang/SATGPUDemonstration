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
   
   1. flatten (CPU)
      
      description: flatten vertices of polygons for further parallelizing.
      
      input: 
      
      - polyton_t* polygon_list[n_polygon]
      
      output: 
      
      - point_t vertices[n_vertex]
   
   2. axes
      
      nx = n_vertex, ny = 1
      
      description: calculate axes of polygons.
      
      input: 
      
      - point_t vertices[n_vertex]
      
      - int next[n_vertex]
      
      output:
      
      - vector_t axes[n_vertex]
   
   3. dot to get every single projection endpoint
      
      nx = n_vertex, ny = n_vertex
      
      
   
   4. min_max
   
   5. is_overlap

6. performance test
