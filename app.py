from flask import Flask, render_template, Response, request, redirect
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import csv
import boto3

global capture, rec_frame, grey, switch, neg, face, rec, out, matchingPer
capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0
matchingPer = ""
faceMatching = ""


net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt',
                               './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)

def compareFaces():
    with open('new_user_credentials.csv', 'r') as input:
        next(input)
        reader = csv.reader(input)
        for line in reader:
            access_key_id = line[2]
            secret_access_key = line[3]
    sourceFile = 'shots/LakshayKumar.png'
    targetFile = 'shots/shot_targetImage.png'
    client = boto3.client('rekognition',
                          aws_access_key_id=access_key_id,
                          aws_secret_access_key=secret_access_key,
                          region_name='us-west-2')

    imageSource = open(sourceFile, 'rb')
    imageTarget = open(targetFile, 'rb')

    response = client.compare_faces(SimilarityThreshold=80,
                                    SourceImage={'Bytes': imageSource.read()},
                                    TargetImage={'Bytes': imageTarget.read()})
    similarity = 0.0
    for faceMatch in response['FaceMatches']:
        position = faceMatch['Face']['BoundingBox']
        similarity = str(faceMatch['Similarity'])

    imageSource.close()
    imageTarget.close()
    print(similarity)
    return similarity




def record(out):
    global rec_frame
    while (rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net

    face_cascade = cv2.CascadeClassifier('saved_model/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return frame


def gen_frames():
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            frame = detect_face(frame)

            if (capture):
                capture = 0
                p = os.path.sep.join(['shots', 'shot_targetImage.png'])
                cv2.imwrite(p, frame)
                faceMatching = compareFaces()
                print("Image saved")
                capture = 0
                global matchingPer
                matchingPer = faceMatching + ""

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


@app.route('/')
def index():
    return render_template('facerecog.html', faceMat=matchingPer)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/')
def redirection():
    if(int(matchingPer)>70.00):
        return redirect("static/successPayment.html",code=302)


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera, matchingPer
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1


    elif request.method == 'GET':
        return render_template('facerecog.html', faceMat=matchingPer)
    return render_template('facerecog.html', faceMat=matchingPer)


if __name__ == '__main__':
    app.run()

camera.release()
cv2.destroyAllWindows()