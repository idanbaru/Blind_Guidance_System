pyttsx3 on jetpack 4.6.1 image:
	sudo apt install mpg123
	sudo apt update
	sudo apt install python3-pip -y
	sudo pip3 install pyttsx3

Program to test pyttsx3 (works offline!):
	python3
	>>> import os
	>>> import pyttsx3
	>>> engine = pyttsx3.init()
	>>> engine.setProperty('rate', 150) ## for slower speed
	>>> engine.setProperty('voice', 'english+m1')	# choose between 'english+{m/f}{1-4}'
	>>> text = 'write something here'
	>>> engine.say(text)
	>>> engine.runAndWait()



gTTS on jetpack 4.6.1 image:
	sudo apt install mpg123
	sudo apt update
	sudo apt install python3-pip -y
	sudo pip3 install gTTS


Program to test gTTS (gTTS REQUIRES INTERNET CONNECTION):
	python3
	>>> import os
	>>> from gtts import gTTS
	>>> myText = 'Check 1 2 3, Jetson AI is working.'
	>>> myOutput = gTTS(text=myText, lang='en', slow=False)
	>>> myOutput.save('../talk.mp3')
	>>> os.system('mpg123 ../talk.mp3')


Interesting method, combining both:
	python3
	>>> import os
	>>> import pyttsx3
	>>> from gtts import gTTS
	>>> from gtts.tts import gTTSError
	>>> try:
	>>> 	gTTS(text=text, lang='en', slow=False).save('talk.mp3')
	>>>	os.system('mpg123 talk.mp3')
	>>> except gTTSError as e:
	>>> 	engine = pyttsx3.init()
	>>>	engine.say(text)
	>>>	engine.runAndWait()

