## Text data format

1. Polygon generator output format && SAT algorithm input format:
   
   ```
   # line1
   polygon_num
   # rest lines
   point_num
   point_list
   ```
   
   e.g.

   ```
   2
   4
   [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0),]
   5
   [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.5,1.5),(0.0,1.0),]
   ```

2. SAT algorithm output format && Polygon viewer input format:
   
   ```
   # line1:
   polygon_num
   # rest lines:
   point_num
   point_list
   overlap_num
   overlap_index_list
   ```
   
   e.g.

   ```
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
   
3. viewer output format: png file.

