import os
import pyttsx3
from gtts import gTTS

text = "Attention Crosswalk ahead in 1.5 meters to the left"

# Test pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 135)
engine.save_to_file(text, 'pyttsx3_test.wav')
# engine.say(text)
engine.runAndWait()

# Test gTTS
gTTS(text=text, lang='en', slow=False).save('gtts_test.mp3')
# os call (or subprocess) 'mpg123 test.mp3'
