from gtts import gTTS
import os
import uuid

AUDIO_FOLDER = 'static/audio'
os.makedirs(AUDIO_FOLDER, exist_ok=True)

def generate_audio(detections):
    if not detections:
        return None
    labels = list(set(d['label'] for d in detections))
    text = "تم اكتشاف " + " و ".join(labels)
    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(AUDIO_FOLDER, filename)
    tts = gTTS(text, lang='ar')
    tts.save(filepath)
    return filename