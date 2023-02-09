# Import Libraries
import cv2
import numpy as np
import sys




MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Represent the gender and age classes
GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# load face Caffe model
faceProto="C:\\app\\opencv_face_detector.pbtxt"
faceModel="C:\\app\\opencv_face_detector_uint8.pb"
# age model
ageProto="C:\\app\\age_deploy.prototxt"
ageModel="C:\\app\\age_net.caffemodel"
# gender model
genderProto="C:\\app\\gender_deploy.prototxt"
genderModel="C:\\app\\gender_net.caffemodel"

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


def get_faces(frame, confidence_threshold=0.5):
    # convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # set the image as input to the NN
    faceNet.setInput(blob)
    # perform inference and get predictions
    output = np.squeeze(faceNet.forward())
    # initialize the result list
    faces = []
    # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces

def display_img(title, img):
   
    # Display Image on screen
    cv2.imshow(title, img)
    # Mantain output until user presses a key
    cv2.waitKey(0)
    # Destroy windows when user presses a key
    cv2.destroyAllWindows()

def get_optimal_font_scale(text, width):
    
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

# from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    return cv2.resize(image, dim, interpolation = inter)    

def predict_gender_age(input_path: str):

    # Read Input Image
    img = cv2.imread(input_path)
    # resize the image, uncomment if you want to resize the image
    # img = cv2.resize(img, (frame_width, frame_height))
    # Take a copy of the initial image and resize it
    frame = img.copy()
    if frame.shape[1] > 1000:
        frame = image_resize(frame, width=1000)
    # predict the faces
    faces = get_faces(frame)
    # Loop over the faces detected
    # for idx, face in enumerate(faces):
    for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
        face_img = frame[start_y: end_y, start_x: end_x]
        
        blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(
            227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
        # Predict Gender
        genderNet.setInput(blob)
        gender_preds = genderNet.forward()
        i = gender_preds[0].argmax()
        gender = GENDER_LIST[i]
        gender_confidence_score = gender_preds[0][i]
        # Draw the box
        label1 = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
        print(label1)
        yPos = start_y - 15
        while yPos < 15:
            yPos += 15

         # Predict Age
        ageNet.setInput(blob)
        age_preds = ageNet.forward()
        print("="*30, f"Face {i+1} Prediction Probabilities", "="*30)
        for i in range(age_preds[0].shape[0]):
            print(f"{AGE_LIST[i]}: {age_preds[0, i]*100:.2f}%")
        i = age_preds[0].argmax()
        age = AGE_LIST[i]
        age_confidence_score = age_preds[0][i]
        # Draw the box
        label2 = f"Age:{age} - {age_confidence_score*100:.2f}%"
        print(label2)
        # get the position where to put the text
        yPos1 = start_y - 50
        while yPos1 < 50:
            yPos1 += 50
        # write the text into the frame
        cv2.putText(frame, label2, (start_x, yPos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
        # draw the rectangle around the face
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)

        # get the font scale for this image size
        optimal_font_scale = get_optimal_font_scale(label1,((end_x-start_x)+25))
        box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
        # Label processed image
        cv2.putText(frame, label1, (start_x, yPos1),
                    cv2.FONT_HERSHEY_SIMPLEX, optimal_font_scale, box_color, 2)

        # Display processed image
    display_img("Gender and age Estimator", frame)
    # uncomment if you want to save the image
    # cv2.imwrite("output.jpg", frame)
    # Cleanup
    cv2.destroyAllWindows()

if __name__ == '__main__':
        # Parsing command line arguments entered by user
    print('test')
    predict_gender_age(sys.argv[1])
    
    sys.stdout.flush()
