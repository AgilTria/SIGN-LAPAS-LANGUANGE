from gtts import gTTS
from playsound import playsound
import threading
import uuid
import os

def _play(text, lang):
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    gTTS(text=text, lang=lang).save(filename)
    playsound(filename)
    os.remove(filename)

def speak_id_jawa(text_id, text_jawa):
    # Indonesia
    threading.Thread(
        target=_play,
        args=(text_id, "id"),
        daemon=True
    ).start()

    # Jawa (diputar SETELAH Indonesia)
    def play_jawa():
        _play(text_jawa, "jw")

    threading.Timer(1.5, play_jawa).start()
