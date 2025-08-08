import cv2
import numpy as np
import face_recognition
import os

path = 'Faces'
path2 = 'Info'
images = []
classnames = []
info = []

mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(os.path.join(path, cl))
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])


def find_keyword_in_file(filename, keyword):
    """
    Reads a text file, searches for a keyword, and prints lines containing it.

    Args:
        filename (str): The path to the text file.
        keyword (str): The keyword to search for.
    """


    try:
        with open(filename, 'r') as file:
            for line in file:
                if keyword.lower() in line.lower():  # Case-insensitive search
                    statement=line.strip() # Print the line, removing extra whitespace
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        statement = "Unknown"
    except Exception as e:
        print(f"An error occurred: {e}")
        statement = "Unknown"
    return statement

def findEncoding(images):
    encodelist = []
    for imag in images:
        imag = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(imag)[0]
        encodelist.append(encode)
    return encodelist

encodeListknown = findEncoding(images)
print('Encoding complete')

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    imgS= cv2.resize(img,(0,0),None,0.25,0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    FaceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,FaceCurFrame)


    for encodeFace,faceLoc in zip(encodesCurFrame,FaceCurFrame):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            info2 =  find_keyword_in_file("Dwayne.txt", name)

            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img, info2, (x1,y2), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)

     # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Video', img)
    cv2.waitKey(1)



