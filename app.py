import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, WebSocket, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from motor.motor_asyncio import AsyncIOMotorClient
from gtts import gTTS
from faster_whisper import WhisperModel
from twilio.rest import Client
import uvicorn
import firebase_admin
from firebase_admin import credentials, firestore
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from bson import ObjectId
from jose import jwt, JWTError
from pymongo import MongoClient
import gridfs
import pytz
import re

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
MONGO_URI = os.getenv("MONGO_URI")
public_url = os.getenv("public_url")
SECRET_KEY = os.getenv("SECRET_KEY")

# Print the public_url to verify it's set correctly
print(f"Public URL: {public_url}")

# Validate API keys
if not all([GROQ_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN]):
    raise ValueError("Missing API keys. Please set them in the .env file.")

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# MongoDB Connection
async_client = AsyncIOMotorClient(MONGO_URI)  # Asynchronous client for general operations
db = async_client["speech_db"]
collection = db["transcriptions"]
message_collection = db["messages"]

# Initialize GridFS with a synchronous client
sync_client = MongoClient(MONGO_URI)  # Synchronous client for GridFS
sync_db = sync_client["speech_db"]
fs = gridfs.GridFS(sync_db)  # Use synchronous database for GridFS

# Directories for uploads
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Load Whisper Model (Use "cuda" for GPU acceleration, "cpu" for CPU)
model = WhisperModel("small", device="cpu", compute_type="int8")

# Initialize Firebase (only for Firestore, not Storage)
cred = credentials.Certificate("C:/HV/Speech_training/Whisper/whisper_api/firebase_credentials.json")
firebase_admin.initialize_app(cred)
firestore_db = firestore.client()

# WebRTC Peer Connections
peer_connections = {}

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Token validity (1 hour)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Generate a JWT token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Validate the JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def generate_tts(text: str, filename: str) -> str:
    """Convert transcription to speech and save as MP3."""
    if not text.strip():
        return None

    # Split text into sentences
    sentences = split_text_into_sentences(text)

    # Generate TTS for each sentence
    tts_path = UPLOAD_DIR / f"{filename}.mp3"
    with open(tts_path, "wb") as output_file:
        for sentence in sentences:
            tts = gTTS(text=sentence, lang="en", slow=False, lang_check=False)
            tts.save("temp.mp3")
            with open("temp.mp3", "rb") as temp_file:
                output_file.write(temp_file.read())
        os.remove("temp.mp3")

    return str(tts_path)

def preprocess_text_for_tts(text: str) -> str:
    """Preprocess text to add pauses for natural speech."""
    # Add pauses for punctuation
    text = text.replace(",", ", <break time='200ms'/>")
    text = text.replace(".", ". <break time='300ms'/>")
    text = text.replace("!", "! <break time='300ms'/>")
    text = text.replace("?", "? <break time='300ms'/>")
    return text

def split_text_into_sentences(text: str) -> list:
    """Split text into sentences for better TTS processing."""
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split on punctuation followed by a space
    return sentences

def emphasize_keywords(text: str, keywords: list) -> str:
    """Emphasize keywords in the text."""
    for keyword in keywords:
        text = text.replace(keyword, f"<emphasis level='strong'>{keyword}</emphasis>")
    return text

# Define IST timezone
IST = pytz.timezone("Asia/Kolkata")

# Convert UTC to IST
def get_ist_timestamp():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...), user_id: str = Form(...)):
    """Upload and transcribe audio from CP user."""
    try:
        if file:
            # Save the uploaded audio file locally
            file_ext = file.filename.split(".")[-1]
            filename = f"{datetime.utcnow().timestamp()}.{file_ext}"
            file_path = UPLOAD_DIR / filename

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            print(f"✅ Audio file saved locally at: {file_path}")

            # Save the raw audio file in MongoDB
            with open(file_path, "rb") as audio_file:
                print(f"Saving file to GridFS with filename: {filename}")
                raw_audio_id = fs.put(audio_file, filename=filename, content_type=f"audio/{file_ext}")
                print(f"✅ Raw audio file saved in GridFS with ID: {raw_audio_id}")

            # Transcribe audio using Whisper
            segments, _ = model.transcribe(str(file_path), language="en")
            transcription_text = " ".join([segment.text for segment in segments])

            print(f"📜 Transcription: {transcription_text}")

            # Convert transcription to speech
            tts_audio_path = generate_tts(transcription_text, filename)
            tts_audio_id = None
            tts_audio_url = None
            if tts_audio_path:
                with open(tts_audio_path, "rb") as tts_file:
                    tts_audio_id = fs.put(tts_file, filename=f"{filename}.mp3", content_type="audio/mpeg")
                    print(f"✅ TTS audio file saved in GridFS with ID: {tts_audio_id}")
                    tts_audio_url = f"{public_url}/audio/{filename}.mp3"

            # Remove local files after processing
            os.remove(file_path)
            if tts_audio_path:
                os.remove(tts_audio_path)

        # Store transcription and audio references in MongoDB
        document = {
            "user_id": user_id,
            "filename": filename,
            "transcription": transcription_text,
            "raw_audio_id": str(raw_audio_id),  # Store GridFS ID for raw audio
            "tts_audio_id": str(tts_audio_id) if tts_audio_id else None,  # Store GridFS ID for TTS audio
            "tts_audio_url": tts_audio_url,  # Public URL for TTS audio
            "timestamp": get_ist_timestamp(),  # Use IST timestamp
        }
        result = await collection.insert_one(document)
        document["_id"] = str(result.inserted_id)  # Convert ObjectId to string

        return JSONResponse(content=document)

    except Exception as e:
        logging.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Audio processing failed.")

@app.post("/upload_message/")
async def upload_message(
    file: UploadFile = File(None),  # Audio file is optional
    user_id: str = Form(...),  # This is the linked CP User ID
    user_email: str = Form(...),
    text: str = Form("")
):
    """Upload audio and text message from the receiver."""
    try:
        audio_id = None
        transcription_text = None
        tts_audio_url = None

        if file:
            # Save the uploaded audio file locally
            file_ext = file.filename.split(".")[-1]
            filename = f"{datetime.utcnow().timestamp()}.{file_ext}"
            file_path = UPLOAD_DIR / filename

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            print(f"✅ Audio file saved locally at: {file_path}")

            # Save the audio file in MongoDB
            with open(file_path, "rb") as audio_file:
                audio_id = fs.put(audio_file, filename=filename, content_type=f"audio/{file_ext}")

            # Transcribe audio using Whisper
            segments, _ = model.transcribe(str(file_path), language="en")
            transcription_text = " ".join([segment.text for segment in segments])

            print(f"📜 Transcription: {transcription_text}")

            # Convert transcription to speech
            tts_audio_path = generate_tts(transcription_text, filename)
            if tts_audio_path:
                with open(tts_audio_path, "rb") as tts_file:
                    tts_audio_id = fs.put(tts_file, filename=f"{filename}.mp3", content_type="audio/mpeg")
                    print(f"✅ TTS audio file saved in GridFS with ID: {tts_audio_id}")
                    tts_audio_url = f"{public_url}/audio/{filename}.mp3"

            # Remove local file after processing
            os.remove(file_path)
            if tts_audio_path:
                os.remove(tts_audio_path)

        # Store the message in MongoDB
        message = {
            "user_id": user_id,
            "user_email": user_email,
            "audio_id": str(audio_id) if audio_id else None,  # Store GridFS ID for audio
            "text": text or transcription_text,  # Use provided text or transcription
            "transcription": transcription_text,  # Transcription of the audio
            "tts_audio_url": tts_audio_url,  # URL of the TTS audio
            "timestamp": datetime.utcnow().isoformat(),  # Current timestamp
        }
        result = await message_collection.insert_one(message)
        message["_id"] = str(result.inserted_id)  # Convert ObjectId to string

        return JSONResponse(content={"message": "Message uploaded successfully", "data": message})

    except Exception as e:
        logging.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Message processing failed.")

@app.get("/messages/{user_id}")
async def get_messages(user_id: str):
    """Retrieve all messages for a user."""
    messages = await message_collection.find({"user_id": user_id}).to_list(length=100)
    # Convert ObjectId to string
    for message in messages:
        message["_id"] = str(message["_id"])
    return JSONResponse(content=messages)

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve audio files from MongoDB."""
    try:
        print(f"Fetching file from GridFS with filename: '{filename}'")
        # Fetch the audio file from GridFS
        grid_out = fs.find_one({"filename": filename})
        if not grid_out:
            logging.error(f"Audio file not found in GridFS: {filename}")
            raise HTTPException(status_code=404, detail=f"Audio file '{filename}' not found in GridFS. Please check the filename.")
        print(f"✅ Audio file found in GridFS: {filename}")
        return StreamingResponse(grid_out, media_type=grid_out.content_type)
    except Exception as e:
        logging.error(f"Failed to fetch audio file: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch audio file")

@app.get("/get_transcriptions/{user_id}")
async def get_transcriptions(user_id: str):
    """Retrieve the last 30 transcriptions for a user."""
    documents = await collection.find({"user_id": user_id}).sort("timestamp", -1).limit(30).to_list(length=30)
    # Convert ObjectId to string
    for document in documents:
        document["_id"] = str(document["_id"])
    return JSONResponse(content=documents)

@app.get("/get_all_transcriptions/")
async def get_all_transcriptions():
    """Retrieve all transcriptions."""
    documents = await collection.find().to_list(length=100)
    # Convert ObjectId to string
    for document in documents:
        document["_id"] = str(document["_id"])
    return JSONResponse(content=documents)

@app.post("/send_alert/")
async def send_alert(receiver_phone: str, user_id: str):
    """Send an alert to the receiver's phone."""
    try:
        message = twilio_client.messages.create(
            body=f"⚠ Alert! The CP User {user_id} needs your attention.",
            from_=TWILIO_PHONE_NUMBER,
            to=receiver_phone
        )
        return {"message": "Alert sent successfully!", "sid": message.sid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebRTC Signaling
@app.websocket("/webrtc")
async def websocket_endpoint(websocket: WebSocket):
    """WebRTC signaling for real-time communication."""
    await websocket.accept()
    peer_connection = RTCPeerConnection()
    peer_connections[websocket] = peer_connection

    try:
        while True:
            data = await websocket.receive_json()
            if "offer" in data:
                # Handle WebRTC offer
                offer = RTCSessionDescription(sdp=data["offer"]["sdp"], type=data["offer"]["type"])
                await peer_connection.setRemoteDescription(offer)
                answer = await peer_connection.createAnswer()
                await peer_connection.setLocalDescription(answer)
                await websocket.send_json({"answer": {"sdp": peer_connection.localDescription.sdp, "type": peer_connection.localDescription.type}})
            elif "candidate" in data:
                # Handle ICE candidate
                candidate = data["candidate"]
                await peer_connection.addIceCandidate(candidate)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await peer_connection.close()
        del peer_connections[websocket]

@app.websocket("/audio_stream")
async def audio_stream(websocket: WebSocket):
    """WebSocket for real-time audio and text communication."""
    await websocket.accept()
    peer_connections[websocket] = {"websocket": websocket, "user_id": None}

    try:
        while True:
            data = await websocket.receive_json()

            # Handle user registration
            if "user_id" in data:
                peer_connections[websocket]["user_id"] = data["user_id"]
                print(f"User connected: {data['user_id']}")

            # Handle audio streaming
            elif "audio_data" in data:
                audio_data = data["audio_data"]
                sender_user_id = peer_connections[websocket]["user_id"]

                # Save the raw audio data as a file
                filename = f"{datetime.utcnow().timestamp()}.m4a"
                file_path = UPLOAD_DIR / filename
                with open(file_path, "wb") as audio_file:
                    audio_file.write(bytes(audio_data))

                print(f"✅ Audio file saved locally at: {file_path}")

                # Save the audio file in MongoDB
                with open(file_path, "rb") as audio_file:
                    fs.put(audio_file, filename=filename, content_type="audio/m4a")
                    print(f"✅ Audio file saved in GridFS with filename: {filename}")

                # Generate the public URL for the audio file
                audio_url = f"{public_url}/audio/{filename}"
                print(f"Generated audio URL: {audio_url}")

                # Broadcast the audio URL to the linked user
                for client, connection in peer_connections.items():
                    if connection["user_id"] == sender_user_id and client != websocket:
                        try:
                            await connection["websocket"].send_json({"audio_url": audio_url})
                        except Exception as e:
                            print(f"Failed to send audio URL: {e}")

            # Handle text message
            elif "text" in data:
                text_message = data["text"]
                sender_user_id = peer_connections[websocket]["user_id"]

                # Broadcast text message to the linked user
                for client, connection in peer_connections.items():
                    if connection["user_id"] == sender_user_id and client != websocket:
                        try:
                            await connection["websocket"].send_json({"text": text_message})
                        except Exception as e:
                            print(f"Failed to send text message: {e}")

            # Handle ping-pong to keep the connection alive
            elif "type" in data and data["type"] == "ping":
                try:
                    await websocket.send_json({"type": "pong"})
                except Exception as e:
                    print(f"Failed to send pong: {e}")

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Ensure the connection is removed from peer_connections
        if websocket in peer_connections:
            del peer_connections[websocket]
        try:
            await websocket.close()
        except Exception as e:
            print(f"Error while closing WebSocket: {e}")

@app.post("/login/")
async def login(username: str = Form(...), password: str = Form(...)):
    """Authenticate user and return a JWT token."""
    # Replace this with your user validation logic (e.g., check MongoDB or Firebase)
    user = await firestore_db.collection("users").where("username", "==", username).get()
    if not user or user[0].to_dict().get("password") != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Generate JWT token
    user_id = user[0].id
    access_token = create_access_token(data={"sub": user_id})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/protected/")
async def protected_route(current_user: str = Depends(get_current_user)):
    """Example of a protected route."""
    return {"message": f"Hello, {current_user}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # fallback to 8000 if PORT not set
    uvicorn.run(app, host="0.0.0.0", port=port)
db.fs.files.find({}, { "filename": 1, "_id": 0 })

