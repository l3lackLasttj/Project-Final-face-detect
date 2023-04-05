from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
import time
from datetime import date
import datetime
import random
import base64
import io
import dlib

app = Flask(__name__)
camera = cv2.VideoCapture(0)
cnt = 0
pause_cnt = 0
justscanned = False

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="student_walailak_db"
)
mycursor = mydb.cursor()
def get_db_connection():
    # Replace the placeholders with your database credentials
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="student_walailak_db"
    )
    return db

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(id):
    face_classifier = cv2.CascadeClassifier("C:/faceRecognition_files/resources/haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)

    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]

    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0

    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = "dataset/"+id+"." + str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`) VALUES
                                ('{}', '{}')""".format(img_id, id))
            mydb.commit()

            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
                cap.release()
                cv2.destroyAllWindows()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

@app.route('/train_classifier/<id>')
def train_classifier(id):
    dataset_dir = "C:/faceRecognition_files/dataset"

    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    return redirect('/')



@app.route('/vfdataset_page/<student>')
def vfdataset_page(student):
    # Render the gendataset.html template with the student ID
    return render_template('gendataset.html', student=student)


@app.route('/vidfeed_dataset/<id>')
def vidfeed_dataset(id):
    # Video streaming route for generating dataset
    return Response(generate_dataset(id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    # Video streaming route for face recognition
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/')
def home():
    # Query the database for student information
    mycursor.execute("select student_id, student_code, student_name, student_major, student_dormitory, student_room, student_added from student_information")
    data = mycursor.fetchall()

    # Render the index.html template with the retrieved data
    return render_template('index.html', data=data)


@app.route('/addstudent')
def addstudent():
    # Get the next available student ID
    mycursor.execute("select ifnull(max(student_id) + 1, 101) from student_information")
    row = mycursor.fetchone()
    id = row[0]

    # Render the addstudent.html template with the new ID
    return render_template('addstudent.html', newid=int(id))


@app.route('/addstudent_code', methods=['POST'])
def addstudent_code():
    # Get the form data from the addstudent.html page
    studentid = request.form.get('txtid')
    studentcode = request.form.get('studentcode')
    studentname = request.form.get('txtname')
    studentmajor = request.form.get('major')
    studentdormitory = request.form.get('dormitory')
    studentroom = request.form.get('room')
    
    # Insert the new student into the database
    sql = """INSERT INTO `student_information` (`student_id`, `student_code`, `student_name`, `student_major`, `student_dormitory`, `student_room`) VALUES
            (%s, %s, %s, %s, %s, %s)"""
    values = (studentid, studentcode, studentname, studentmajor, studentdormitory, studentroom)

    mycursor.execute(sql, values)
    mydb.commit()

    # Redirect to the vfdataset_page with the new student ID
    return redirect(url_for('vfdataset_page', student=studentid))

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def face_recognition():  # generate frame by frame from camera
    from datetime import datetime
    from mysql.connector import Binary
    
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(
            gray_image, scaleFactor, minNeighbors)

        global justscanned
        global pause_cnt
        global last_saved_time
        
        pause_cnt += 1
        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1

                n = (100 / 30) * cnt
                # w_filled = (n / 100) * w
                w_filled = (cnt / 30) * w

                cv2.putText(img, str(int(n))+' %', (x + 20, y + h + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(img, (x, y + h + 40),
                              (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled),
                              y + h + 50), (153, 255, 255), cv2.FILLED)

                mycursor.execute("select a.img_person, b.student_name, b.student_major, b.student_dormitory, b.student_room"
                                 "  from img_dataset a "
                                 "  left join student_information b on a.img_person = b.student_id "
                                 " where img_id = " + str(id))
                row = mycursor.fetchone()
                id = row[0]
                name = row[1]
                major = row[2]

                if int(cnt) == 30:
                    cnt = 0

                    mycursor.execute("insert into check_in (enter_date, enter_number) values('"+str(date.today())+"', '" + id + "')")
                    mydb.commit()
                    cv2.putText(img, name, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,0.8, (153, 255, 255), 2, cv2.LINE_AA)
                    time.sleep(1)

                    justscanned = True
                    pause_cnt = 0

            else:
                if not justscanned:
                    current_time = time.time()
                    cv2.putText(img, 'UNKNOWN', (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    def save_unknown_image(unknown_face):
                        global mycursor, mydb

                        # Save the image in the unknown folder
                        if not os.path.exists("unknown"):
                            os.mkdir("unknown")
                            
                        mycursor.execute("INSERT INTO unknown (datetime, createdAt) VALUES (%s, NOW())", (date.today(),))
                        mydb.commit()

                        img_id = mycursor.lastrowid
                        img_path = os.path.join("unknown", f"{img_id}.jpg")
                        cv2.imwrite(img_path, unknown_face)

                        # Save the image in the unknown table
                        buffer = cv2.imencode(".jpg", unknown_face)[1].tobytes()
                        img_binary = Binary(buffer)
                        mycursor.execute("UPDATE unknown SET image_data = %s WHERE id = %s", (img_binary, img_id))
                        mydb.commit()

                        return img_id
                    
                    global last_saved_time
                    last_saved_time = time.time()

                    # Save the unknown person's image every 5 seconds
                    if current_time - last_saved_time >= 5:
                        unknown_face = img[y:y+h, x:x+w]
                        img_id = save_unknown_image(unknown_face)
                        print(f"Unknown face image saved with ID: {img_id}")
                        last_saved_time = current_time

                else:
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                if pause_cnt > 80:
                    justscanned = False

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10,(255, 255, 0), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier("C:/faceRecognition_files/resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    wCam, hCam = 400, 400

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break
        

@app.route('/check_in')
def check_in():
    """Video streaming home page."""
    mycursor.execute("select a.enter_id, a.enter_number, b.student_name, b.student_major, a.enter_added "
                     "  from check_in a "
                     "  left join student_information b on a.enter_number = b.student_id "
                     " where a.enter_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return render_template('check_in.html', data=data)


@app.route('/countTodayScan')
def countTodayScan():

    mycursor.execute("select count(*) "
                     "  from check_in "
                     " where enter_date = curdate() ")
    row = mycursor.fetchone()
    rowcount = row[0]

    return jsonify({'rowcount': rowcount})


@app.route('/loadData', methods=['GET', 'POST'])
def loadData():

    mycursor.execute("select a.enter_id, a.enter_number, b.student_name, b.student_major, date_format(a.enter_added, '%H:%i:%s') "
                     "  from check_in a "
                     "  left join student_information b on a.enter_number = b.student_id "
                     " where a.enter_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return jsonify(response=data)


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognitionout >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def face_recognitionout():  # generate frame by frame from camera
    from datetime import datetime
    from mysql.connector import Binary
    def save_unknown_image(unknown_face):
        global mycursor, mydb

        # Save the image in the unknown folder
        if not os.path.exists("unknown"):
            os.mkdir("unknown")

        mycursor.execute("INSERT INTO unknown (datetime, createdAt) VALUES (%s, NOW())", (date.today(),))
        mydb.commit()

        img_id = mycursor.lastrowid
        img_path = os.path.join("unknown", f"{img_id}.jpg")
        cv2.imwrite(img_path, unknown_face)

        # Save the image in the unknown table
        buffer = cv2.imencode(".jpg", unknown_face)[1].tobytes()
        img_binary = Binary(buffer)
        mycursor.execute("UPDATE unknown SET image_data = %s WHERE id = %s", (img_binary, img_id))
        mydb.commit()

        return img_id
    
    global last_saved_time
    last_saved_time = time.time()
    
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(
            gray_image, scaleFactor, minNeighbors)

        global justscanned
        global pause_cnt
        global last_saved_time
        
        pause_cnt += 1
        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1

                n = (100 / 30) * cnt
                # w_filled = (n / 100) * w
                w_filled = (cnt / 30) * w

                cv2.putText(img, str(int(n))+' %', (x + 20, y + h + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(img, (x, y + h + 40),
                              (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled),
                              y + h + 50), (153, 255, 255), cv2.FILLED)

                mycursor.execute("select a.img_person, b.student_name, b.student_major, b.student_dormitory, b.student_room"
                                 "  from img_dataset a "
                                 "  left join student_information b on a.img_person = b.student_id "
                                 " where img_id = " + str(id))
                row = mycursor.fetchone()
                if row is not None:
                    id = row[0]
                    name = row[1]
                    major = row[2]
                else:
                    # Handle the case when the SQL query returns no rows.
                    # You may want to set default values or display an error message.
                    id = -1
                    name = "Unknown"
                    major = "Unknown"


                if int(cnt) == 30:
                    cnt = 0

                    mycursor.execute("insert into check_out (out_date, out_number) values('"+str(date.today())+"', '" + id + "')")
                    mydb.commit()
                    cv2.putText(img, name, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,0.8, (153, 255, 255), 2, cv2.LINE_AA)
                    time.sleep(1)

                    justscanned = True
                    pause_cnt = 0

            else:
                if not justscanned:
                    current_time = time.time()
                    cv2.putText(img, 'UNKNOWN', (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                    # Save the unknown person's image every 5 seconds
                    if current_time - last_saved_time >= 5:
                        unknown_face = img[y:y+h, x:x+w]
                        img_id = save_unknown_image(unknown_face)
                        print(f"Unknown face image saved with ID: {img_id}")
                        last_saved_time = current_time

                else:
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                if pause_cnt > 80:
                    justscanned = False

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10,
                               (255, 255, 0), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier("C:/faceRecognition_files/resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    wCam, hCam = 400, 400

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break


@app.route('/vidfeed_datasetout/<id>')
def vidfeed_datasetout(id):
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_datasetout(id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feedout')
def video_feedout():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognitionout(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/check_out')
def check_out():
    """Video streaming home page."""
    mycursor.execute("select a.out_id, a.out_number, b.student_name, b.student_major, a.out_added "
                     "  from check_out a "
                     "  left join student_information b on a.out_number = b.student_id "
                     " where a.out_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return render_template('check_out.html', data=data)


@app.route('/countTodayScanout')
def countTodayScanout():

    mycursor.execute("select count(*) "
                     "  from check_out "
                     " where out_date = curdate() ")
    row = mycursor.fetchone()
    rowcount = row[0]

    return jsonify({'rowcount': rowcount})


@app.route('/loadDataout', methods=['GET', 'POST'])
def loadDataout():

    mycursor.execute("select a.out_id, a.out_number, b.student_name, b.student_major, date_format(a.out_added, '%H:%i:%s') "
                     "  from check_out a "
                     "  left join student_information b on a.out_number = b.student_id "
                     " where a.out_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return jsonify(response=data)


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Checkerror_face_in_dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

@app.route('/checkFaces')
def check_faces():
    mycursor.execute("SELECT COUNT(*) FROM img_dataset")
    count = mycursor.fetchone()[0]

    if count > 0:
        return jsonify({"faces_present": True})
    else:
        return jsonify({"faces_present": False})


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Report >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/search', methods=['POST'])
def search():
    start_date_str = request.form.get('start_date', '')
    end_date_str = request.form.get('end_date', '')

    if not start_date_str or not end_date_str:
        return jsonify({'error': 'Start date and end date are required.'}), 400

    try:
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid date format. Please use YYYY-MM-DD format.'}), 400

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    query = """
   (SELECT enter_id AS id, enter_number AS number, enter_date AS date,enter_added AS added, 'check_in' AS action
    FROM check_in AS DATETIME_BE
    WHERE enter_date BETWEEN %s AND %s)
    UNION ALL
    (SELECT out_id AS id, out_number AS number, out_date AS date,out_added AS added, 'check_out' AS action
    FROM check_out AS DATETIME_BE
    WHERE out_date BETWEEN %s AND %s)
    ORDER BY added ASC
    LIMIT 1000
    
    """

    cursor.execute(query, (start_date, end_date, start_date, end_date))
    results = cursor.fetchall()
    print(results)

    db.close()

    if not results:
        return jsonify({'error': 'No records found between the specified dates.'}), 404

    return jsonify(results)

@app.route('/reportpic')
def reportpic():
    return render_template('reportpic.html')

@app.route('/searchpic', methods=['POST'])
def searchpic():
    start_date_str = request.form.get('start_date', '')
    end_date_str = request.form.get('end_date', '')

    if not start_date_str or not end_date_str:
        return jsonify({'error': 'Start date and end date are required.'}), 400

    try:
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid date format. Please use YYYY-MM-DD format.'}), 400

    db = get_db_connection()
    cursor = db.cursor()

    query = """
        SELECT * FROM unknown
        WHERE DATE(createdAt) BETWEEN %s AND %s
        ORDER BY createdAt ASC, id ASC
    """
    cursor.execute(query, (start_date, end_date))
    
    results = []
    for row in cursor.fetchall():
        # Convert the image data to a base64 encoded string
        image_data = base64.b64encode(row[1]).decode('utf-8')
        results.append({
            'id': row[0],
            'datetime': row[3].isoformat(),
            'image_data': image_data
        })

    db.close()

    if not results:
        return jsonify({'error': 'No records found between the specified dates.'}), 404

    return jsonify(results)

@app.route('/delete/<string:student_id>', methods=['DELETE'])
def delete_student(student_id):
    try:
        # Get the image path from the img_dataset table
        mycursor.execute("SELECT img_path FROM img_dataset WHERE img_person = %s", (student_id,))
        row = mycursor.fetchone()
        
        if row:
            # Delete student image from the file system
            image_path = row[0]
            if os.path.exists(image_path):
                os.remove(image_path)

            # Delete the image record from the img_dataset table
            mycursor.execute("DELETE FROM img_dataset WHERE img_person = %s", (student_id,))
            mydb.commit()

        # Delete the student record from the student_information table
        mycursor.execute("DELETE FROM student_information WHERE student_id = %s", (student_id,))
        mydb.commit()

        return jsonify({"success": True, "message": "Student deleted successfully."}), 200
    except Exception as e:
        print(e)
        return jsonify({"success": False, "message": "Failed to delete student."}), 500



@app.route('/edit_student', methods=['PUT'])
def edit_student():
    try:
        student_id = request.json['student_id']
        updated_data = {}

        if 'student_name' in request.json:
            updated_data['name'] = request.json['student_name']

        if 'student_major' in request.json:
            updated_data['major'] = request.json['student_major']

        if 'student_dormitory' in request.json:
            updated_data['dormitory'] = request.json['student_dormitory']

        if 'student_room' in request.json:
            updated_data['room'] = request.json['student_room']

        set_clause = ', '.join([f"{key} = %s" for key in updated_data])
        query = f"UPDATE students SET {set_clause} WHERE id = %s"
        mycursor.execute(query, (*updated_data.values(), student_id))
        mydb.commit()

        return jsonify({"success": True, "message": "Student updated successfully."}), 200
    except Exception as e:
        print(e)
        return jsonify({"success": False, "message": "Failed to update student."}), 500

# Error handlers
@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({'error': 'Bad Request'}), 400

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource Not Found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
