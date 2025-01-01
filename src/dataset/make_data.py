import os 
import cv2


data_dir = 'data\\raw_data'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
number_of_signs = 3
dataset_size = 20

cap = cv2.VideoCapture(0)
for i in range(number_of_signs):
    if not os.path.exists(os.path.join(data_dir, str(i))):
        os.makedirs(os.path.join(data_dir, str(i)))
        
    print(f'Collecting data for sign {i}')
    
    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press "Q" to collect data', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
        
        
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('fame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_dir, str(i), '{}.jpg'.format(counter)), frame)
        
        counter += 1
        
cap.release()
cv2.destroyAllWindows()