import cv2
import os
import glob

home = 'mov_images'

movs = glob.glob('yolo.mp4')
print(movs)

for file in movs:
    filesplit = file.split('.')
    title = filesplit[-2]
    os.makedirs(f'./{home}/{title}', exist_ok= True)

    cap = cv2.VideoCapture(file)
    cnt = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            if cnt % 5 == 0:
                str = f'./{home}/{title}/{title}{cnt}.jpg'
                cv2.imwrite(str, frame)

            cnt = cnt + 1
            
        else:
            break




