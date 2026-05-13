import pyttsx3

engine = pyttsx3.init(driverName='sapi5')  # 🔴 IMPORTANT
engine.say("A")
engine.runAndWait()

