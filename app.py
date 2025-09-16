import cv2
from flask import Flask, request, jsonify
from datetime import datetime
import tempfile
import os
from werkzeug.utils import secure_filename
from pocketbase import PocketBase
from pocketbase.client import FileUpload
import logging
from dotenv import load_dotenv
from facedetector import FaceDetector

# Load environment variables
load_dotenv(dotenv_path="./env/.env")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize PocketBase client with error handling
try:
    pb = PocketBase(base_url=os.getenv("DB_URL"))
    
    # Login with credentials from environment variables
    pb.admins.auth_with_password(
        email=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD")
    )
    logger.info("Successfully connected to PocketBase")
except Exception as e:
    logger.error(f"Failed to connect to PocketBase: {e}")
    pb = None


# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'm4v', 'webm'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_faces_from_video(video_path, frame_interval=10, padding=1.0):
    """Extract faces from video with timestamps and locations using FaceDetector"""
    faces_data = []
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return [], 0, 0
            
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        frame_count = 0
        while True:
            success, frame = video.read()
            if not success:
                break
                
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps if fps > 0 else 0
                
                # Detect faces using your FaceDetector
                if FaceDetector:
                    faces = FaceDetector.detect(frame)
                    
                    for face in faces:
                        bbox = face['bounding_box']
                        pivotX, pivotY = face['pivot']
                        
                        # Skip small faces
                        if bbox['width'] < 50 or bbox['height'] < 50:
                            continue
                        
                        # Extract face region with padding (using your method)
                        left = int(pivotX - bbox['width'] / 2.0 * padding)
                        top = int(pivotY - bbox['height'] / 2.0 * padding)
                        right = int(pivotX + bbox['width'] / 2.0 * padding)
                        bottom = int(pivotY + bbox['height'] / 2.0 * padding)
                        
                        # Ensure coordinates are within frame bounds
                        left = max(0, left)
                        top = max(0, top)
                        right = min(frame.shape[1], right)
                        bottom = min(frame.shape[0], bottom)
                        
                        face_img = frame[top:bottom, left:right]
                        
                        # Only process if the face image is valid
                        if face_img.size > 0:
                            # Store face data
                            face_data = {
                                'timestamp': timestamp,
                                'x': left,
                                'y': top,
                                'width': right - left,
                                'height': bottom - top,
                                'frame': frame_count,
                                'image': face_img,
                                'confidence': face.get('confidence', 0.95)
                            }
                            faces_data.append(face_data)
                else:
                    # Fallback to Haar Cascade if FaceDetector is not available
                    logger.warning("FaceDetector not available, using Haar Cascade fallback")
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
                    
                    for (x, y, w, h) in faces:
                        padding_val = 20
                        x1 = max(0, x - padding_val)
                        y1 = max(0, y - padding_val)
                        x2 = min(frame.shape[1], x + w + padding_val)
                        y2 = min(frame.shape[0], y + h + padding_val)
                        
                        face_img = frame[y1:y2, x1:x2]
                        
                        if face_img.size > 0:
                            face_data = {
                                'timestamp': timestamp,
                                'x': x1,
                                'y': y1,
                                'width': x2 - x1,
                                'height': y2 - y1,
                                'frame': frame_count,
                                'image': face_img,
                                'confidence': 0.95
                            }
                            faces_data.append(face_data)
                    
            frame_count += 1
            
        video.release()
        return faces_data, duration, fps
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        if 'video' in locals():
            video.release()
        return [], 0, 0

@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    logger.debug(f"Request received: {request.method} {request.url}")
    
    # Check if files are present in the request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    if not allowed_file(video_file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, secure_filename(video_file.filename))
    
    try:
        # Save video file
        video_file.save(video_path)
        
        # Extract video metadata
        video_cap = cv2.VideoCapture(video_path)
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        video_cap.release()
        
        # Extract faces from video
        faces_data, duration, actual_fps = extract_faces_from_video(video_path)
        
        # Upload video to PocketBase
        video_record = pb.collection('videos').create(
            {
                'title': video_file.filename,
                'upload_date': datetime.now().isoformat(),
                'duration': duration,
                'resolution': f"{width}x{height}",
                'fps': fps,
                'original_filename': video_file.filename,
                'video_file': FileUpload((video_file.filename, open(video_path, 'rb')))
            }
        )
        
        # Process and store each detected face
        processed_faces = 0
        for i, face_data in enumerate(faces_data):
            try:
                # Save face image temporarily
                face_filename = f"face_{video_record.id}_{i}.jpg"
                face_path = os.path.join(temp_dir, face_filename)
                
                if face_data['image'] is not None and face_data['image'].size > 0:
                    success = cv2.imwrite(face_path, face_data['image'])
                    if success and os.path.exists(face_path):
                        # Upload face to PocketBase
                        face_record_data = {
                            'video': video_record.id,
                            'timestamp': float(face_data['timestamp']),
                            'x_position': int(face_data['x']),
                            'y_position': int(face_data['y']),
                            'width': int(face_data['width']),
                            'height': int(face_data['height']),
                            'frame_number': int(face_data['frame']),
                            'confidence': float(face_data['confidence']),
                            'image_file': FileUpload((face_filename, open(face_path, 'rb')))
                        }
                        
                        face_record = pb.collection('faces').create(face_record_data)
                        processed_faces += 1
            except Exception as e:
                logger.error(f"Error processing face {i}: {e}")
                continue
        
        # Cleanup
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass
        os.rmdir(temp_dir)
        
        return jsonify({
            'message': 'Video processed successfully',
            'video_id': video_record.id,
            'faces_detected': len(faces_data),
            'faces_processed': processed_faces,
            'duration': duration,
            'detector_used': 'FaceDetector' if FaceDetector else 'HaarCascade'
        })
        
    except Exception as e:
        logger.error(f"Error in upload_video: {str(e)}")
        # Cleanup on error
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
            
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/videos/<video_id>/faces', methods=['GET'])
def get_video_faces(video_id):
    try:
        if pb is None:
            return jsonify({'error': 'Database connection failed'}), 500
            
        # Get all faces for a specific video
        faces = pb.collection('faces').get_full_list({
            'filter': f'video = "{video_id}"',
            'sort': 'timestamp'
        })
        
        faces_data = []
        for face in faces:
            try:
                face_url = pb.files.get_url(face, face.image_file)
                faces_data.append({
                    'id': face.id,
                    'timestamp': face.timestamp,
                    'position': {
                        'x': face.x_position,
                        'y': face.y_position,
                        'width': face.width,
                        'height': face.height
                    },
                    'frame_number': face.frame_number,
                    'confidence': face.confidence,
                    'image_url': face_url,
                    'created': face.created
                })
            except Exception as e:
                logger.error(f"Error processing face {face.id}: {e}")
                continue
        
        return jsonify({'faces': faces_data})
        
    except Exception as e:
        logger.error(f"Error in get_video_faces: {e}")
        return jsonify({'error': str(e)}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    status = {
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'pocketbase_connected': pb is not None,
        'face_detector_available': FaceDetector is not None
    }
    return jsonify(status)

@app.route('/api/debug/upload', methods=['POST'])
def debug_upload():
    """Debug endpoint to check upload functionality"""
    logger.info("Debug upload request received")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Content type: {request.content_type}")
    logger.info(f"Files: {dict(request.files)}")
    logger.info(f"Form: {dict(request.form)}")
    
    if 'video' in request.files:
        file = request.files['video']
        logger.info(f"Video file: {file.filename}")
        logger.info(f"Video content type: {file.content_type}")
        
        # Save the file temporarily
        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)
        file_size = os.path.getsize(temp_path)
        logger.info(f"File size: {file_size} bytes")
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            'status': 'success',
            'filename': file.filename,
            'content_type': file.content_type,
            'size': file_size
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'No video file in request',
            'available_files': list(request.files.keys())
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)