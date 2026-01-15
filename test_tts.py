from src.tts import enqueue_tts, process_tts_queue
import time

enqueue_tts("ini tes suara indonesia")
enqueue_tts("mangan", lang="jw")

while True:
    process_tts_queue()
    time.sleep(0.1)
