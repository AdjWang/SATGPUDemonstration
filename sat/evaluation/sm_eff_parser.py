import re
from matplotlib import pyplot as plt

with open('sm_eff.txt', 'r') as f:
    sm_eff_items = f.read()

res = re.findall(r'(\d+\.\d+)%\n', sm_eff_items)
# print(res)

count = 60*4
axis_avg = [float(res[i])/100 for i in range(0, count, 4)]
projection_segments_avg = [float(res[i])/100 for i in range(1, count, 4)]
overlapping_avg = [float(res[i])/100 for i in range(2, count, 4)]
projection_endpoints_avg = [float(res[i])/100 for i in range(3, count, 4)]

# print(axis_avg)
# print(projection_segments_avg)
# print(overlapping_avg)
# print(projection_endpoints_avg)

polygon_n = range(50, 3000+50, 50)
assert len(polygon_n) == len(axis_avg) == len(projection_segments_avg) == \
    len(overlapping_avg) == len(projection_endpoints_avg)

plt.title('stream multiprocessor efficiency')
plt.xlabel('number of polygons')
plt.ylabel('sm_efficiency')
plt.plot(polygon_n, axis_avg, label='kernel_get_axis')
plt.plot(polygon_n, projection_segments_avg, label='kernel_get_projection_segments')
plt.plot(polygon_n, overlapping_avg, label='kernel_get_overlapping')
plt.plot(polygon_n, projection_endpoints_avg, label='kernel_get_projection_endpoints')
plt.legend()
plt.show()
