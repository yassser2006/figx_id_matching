from flask import Flask, render_template, Response
import face_recognition
import numpy as np
from PIL import ImageDraw
from PIL import Image as Img
from cv2 import VideoCapture, imencode, imwrite

app = Flask(__name__)

camera = VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
def encoding_face(image_url):
    image = face_recognition.load_image_file(image_url)
    face_encoding = face_recognition.face_encodings(image)[0]
    return face_encoding

def recofnizeFace(imPath,known_face_encodings,known_face_names):
# Load an image with an unknown face
    unknown_image = face_recognition.load_image_file(imPath)

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Img.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        verified = ""

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            verified = "Authorised"
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))

            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(verified)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 255, 0), outline=(0, 255, 0))
            draw.text((left + 6, bottom - text_height - 5), verified, fill=(255, 255, 255, 255))
        else:
            verified = "Not Authorised"
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(255,0, 0))

            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(verified)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255,0, 0), outline=(255,0, 0))
            draw.text((left + 6, bottom - text_height - 5), verified, fill=(255, 255, 255, 255))


    # Remove the drawing library from memory as per the Pillow docs
    del draw
    # Display the resulting image
#     display(pil_image)

    return pil_image, verified

@app.route('/auth_with_ocr')
def auth_with_ocr():
    id_photo, person_photo = 'id_photo.jpg', 'person.jpg'
#     ocr_text,name = detectIDDetails(id_photo)
    id_face_encoding = encoding_face(id_photo)

    known_face_encodings = [id_face_encoding]
    known_face_names = ["name"]

    image_verified, verified = recofnizeFace(person_photo,known_face_encodings,known_face_names)
    image_verified.save("static/out.jpg")
    return verified

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
#             cv2.waitKey(1000)
#             cv2.imwrite('id_photo.jpg',frame)
            ret, buffer = imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/id_photo')
def id_photo():
    success, frame = camera.read()  # read the camera frame
    if success:
        imwrite('id_photo.jpg',frame)
    return video_feed()

@app.route('/person')
def person():
    success, frame = camera.read()  # read the camera frame
    if success:
        imwrite('person.jpg',frame)
    return video_feed()     
            
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
