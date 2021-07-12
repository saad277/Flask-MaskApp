from flask import Flask,request,jsonify,Response
import os
import numpy 
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

port = int(os.environ.get("PORT", 5000))
app = Flask(__name__)

@app.route("/classify",methods=["GET","POST"])
def index():
    if request.method == 'POST':
        count={"mask":0,"withoutMask":0}
        img=request.files["img"]
        image=img.read()
        npimg = numpy.fromstring(image, numpy.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                # load our serialized face detector model from disk
        print("[INFO] loading face detector model...")
        prototxtPath = ( "./face_detector/deploy.prototxt")
        weightsPath = ("./face_detector/res10_300x300_ssd_iter_140000.caffemodel")
        net = cv2.dnn.readNet(prototxtPath, weightsPath)

        # load the face mask detector model from disk
        print("[INFO] loading face mask detector model...")
        model = load_model("./mask_detector.model")

        # load the input image from disk, clone it, and grab the image spatial
        # dimensions
        
        orig = img.copy()
        (h, w) = img.shape[:2]

        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        print("[INFO] computing face detections...")
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object
                    box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # ensure the bounding boxes fall within the dimensions of
                    # the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # extract the face ROI, convert it from BGR to RGB channel
                    # ordering, resize it to 224x224, and preprocess it
                    face = img[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = numpy.expand_dims(face, axis=0)

                    # pass the face through the model to determine if the face
                    # has a mask or not
                    (mask, withoutMask) = model.predict(face)[0]
                    print(mask, withoutMask)

                    # determine the class label and color we'll use to draw
                    # the bounding box and text
                    
                    label = "Mask" if mask > withoutMask else "No Mask"
                    if mask > withoutMask :
                        count["mask"]=count["mask"]+1
                    else:
                        count["withoutMask"]=count["withoutMask"]+1
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                    # include the probability in the label
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                    # display the label and bounding box rectangle on the output
                    # frame
                    cv2.putText(img, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)

	# show the output image
        print(count)
        return count
        #cv2.imshow("Output", img)
        #cv2.waitKey(0)        

if __name__=="__main__":
    app.run(host='0.0.0.0', port=port, debug=True)
