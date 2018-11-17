from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import re
import shutil

#将4000个人混在一起的图片(也可以是增广之后的数据集)分别放到对应文件夹
#筛选出下标为1的图片
#将图片按照人的标号分别copy到各个以标号命名的文件夹

def main():
    print("main")
    image_dir = "data/HW_1_Face/raw/"

    for file in os.listdir(image_dir):
        print(file)
        matchObj1 = re.match(r'(.*)_([12])(.*?).jpg', file, re.M | re.I)
        print(matchObj1.groups())
        if matchObj1.group(2) != "1":
            continue
        number = matchObj1.group(1)
        newDir = "data/HW_1_Face/finetune1/"+number
        if not os.path.exists(newDir):
            # print("makedir:",newDir)
            os.makedirs(newDir)
        else:
            print("exist:", newDir)
        newFile = "data/HW_1_Face/finetune1/"+number +"/"+file
        shutil.copyfile(os.path.join(image_dir, file),newFile)

if __name__ == '__main__':
    main()

