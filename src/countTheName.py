f = open('src/inception_resnet.txt','r')
list = []
for line in f.readlines():
    names = line.split('/')
    if names[1] not in list:
        list.append(names[1])

print(list)
print(len(list))