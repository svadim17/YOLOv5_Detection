a = ['2 0.46875 0.9234375 0.85 0.096875']

b = a[0].split(' ', 1)
print(b)

classes = []
boxes = []

classes.append(int(b[0]))
boxes_temp = list(b[1].split(' '))
boxes.append([eval(i) for i in boxes_temp])

print(classes, boxes)
