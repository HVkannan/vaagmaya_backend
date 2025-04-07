# Speech Training Backend

This project is a backend application built using FastAPI that facilitates audio processing, transcription, and text-to-speech generation. It connects to MongoDB for data storage and utilizes Twilio for sending alerts. The application also supports real-time communication through WebRTC.

## Features

- Upload audio files for transcription.
- Generate text-to-speech (TTS) audio from transcriptions.
- Store audio files and transcriptions in MongoDB.
- Send alerts via SMS using Twilio.
- Real-time audio and text communication using WebRTC.

## Project Structure

```
hosted_backend
├── app.py               # Main application file
├── .env.example         # Example environment variables
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Requirements

- Python 3.7 or higher
- MongoDB
- Twilio account for SMS alerts

## Setup Instructions

1. **Clone the repository:**

   ```
   git clone https://github.com/yourusername/speech_training_backend.git
   cd speech_training_backend
   ```

2. **Create a virtual environment:**

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**

   Copy the `.env.example` file to `.env` and fill in the required values:

   ```
   cp .env.example .env
   ```

   Make sure to set the following variables in your `.env` file:

   ```
   GROQ_API_KEY=your_groq_api_key
   TWILIO_ACCOUNT_SID=your_twilio_account_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   TWILIO_PHONE_NUMBER=your_twilio_phone_number
   MONGO_URI=your_mongo_uri
   public_url=your_public_url
   SECRET_KEY=your_secret_key
   ```

5. **Run the application:**

   ```
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

## Usage

- **Upload Audio:** Send a POST request to `/upload_audio/` with the audio file and user ID.
- **Get Transcriptions:** Retrieve transcriptions for a user by sending a GET request to `/get_transcriptions/{user_id}`.
- **Send Alert:** Send an alert to a receiver's phone using the `/send_alert/` endpoint.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.