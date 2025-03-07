from flask import Flask, render_template, request, redirect, url_for, session, g , Response
from live_feed import generate_frames , process_frames
import psycopg2
from psycopg2 import sql
import secrets
import threading
import os
from detection import detect_objects_in_video

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Global variable to store camera settings
camera_lock = threading.Lock()
camera_running = False
camera_thread = None
camera_source = None
camera_url = None
session_id = None

app.secret_key = secrets.token_hex(16)  # Set the secret key

# PostgreSQL configurations
DATABASE_CONFIG = {
    'dbname': 'person_detection',
    'user': 'postgres',
    'password': 'raj123456',
    'host': 'localhost'
}

# Function to get a database connection
def get_db():
    if 'db' not in g:
        g.db = psycopg2.connect(**DATABASE_CONFIG)
    return g.db

# Close the database connection at the end of the request
@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            modelFile = './project/demo_work/models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
            configFile = './project/demo_work/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
            classFile = './project/demo_work/coco_class_labels.txt'
            detect_objects_in_video(filepath, file.filename, app.config['PROCESSED_FOLDER'], modelFile, configFile, classFile)
            return render_template('index.html', filename=file.filename)
    return render_template('index.html', filename=None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(sql.SQL("SELECT * FROM accounts WHERE username = %s AND password = %s"), [username, password])
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['id'] = account[0]
            session['username'] = account[1]
            return redirect(url_for('home'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'confirm_password' in request.form:
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password == confirm_password:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(sql.SQL("INSERT INTO accounts (username, password) VALUES (%s, %s)"), [username, password])
            conn.commit()
            msg = 'You have successfully registered!'
            return redirect(url_for('login'))
        else:
            msg = 'Passwords do not match!'
    return render_template('register.html', msg=msg)

@app.route('/home')
def home():
    if 'loggedin' in session:
        conn = get_db()
        cursor = conn.cursor()
        
        # Get total detections for the logged-in user
        cursor.execute(sql.SQL("SELECT COUNT(*) FROM detected_person WHERE id = %s"), [session['id']])
        total_detections = cursor.fetchone()[0]

        # Get recent detections in the last hour for the logged-in user
        cursor.execute(sql.SQL("SELECT COUNT(*) FROM detected_person WHERE id = %s AND (detection_date || ' ' || detection_time)::timestamp >= NOW() - INTERVAL '1 hour'"), [session['id']])
        recent_detections = cursor.fetchone()[0]
        
        cursor.close()
        
        return render_template('dashboard.html', username=session['username'], total_detections=total_detections, recent_detections=recent_detections)
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/detections')
def detections():
    return render_template('detections.html')

@app.route('/images')
def show_images():

    if 'loggedin' in session:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(sql.SQL("""
            SELECT image_name FROM detected_person 
            WHERE id = %s
        """), [session['id']])
        images = cursor.fetchall()
        
        if not images:
            images = None  # Set images to None if no images are found
        
        return render_template('photos.html', images=images)
    return redirect(url_for('login'))

@app.route('/camera', methods=['GET', 'POST'])
def camera():
    if 'loggedin' in session:
        global session_id
        session_id = session['id']
        print(session_id)
        if request.method == 'POST':
            global camera_running, camera_thread, camera_source, camera_url

            camera_source = request.form.get('camera_source')
            camera_url = request.form.get('camera_url', None)

            if camera_running and camera_thread is not None:
                camera_running = False
                camera_thread.join()

            camera_running = True  # Start the camera feed

            with open('status.txt', 'w') as file:
                file.write(str(camera_running))
                print(camera_running)
                print("Status changed to True.")

                # Start process_frames in a separate thread
                if camera_source or camera_url:
                    source = camera_source if camera_source else camera_url
                    camera_running = threading.Thread(target=process_frames, args=(source, session_id))
                    camera_running.start()
                    return redirect(url_for('live_video'))
                else:
                    return "Error: No valid camera source or URL provided.", 400

        return render_template('camera.html')
    return redirect(url_for('login'))


@app.route('/video_feed')
def video_feed():
    if 'loggedin' in session:
        global camera_source
        global camera_url
        global session_id
        if camera_source:
            return Response(generate_frames(camera_source, session_id), mimetype='multipart/x-mixed-replace; boundary=frame')
        elif camera_url:
            return Response(generate_frames(camera_url, session_id), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            # Return an empty response or an error message when no valid source is provided
            return "Error: No valid camera source or URL provided.", 400
    return redirect(url_for('login'))


@app.route('/livevideo')
def live_video():
    if 'loggedin' in session:
        return render_template('livevideo.html')
    return redirect(url_for('login'))

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_running
    camera_running = False  # Stop the camera feed
    with open('status.txt', 'w') as file:
        file.write(str(camera_running))
    print("Status changed to False.")
    return redirect(url_for('camera'))

if __name__ == '__main__':
    app.run(debug=True)
