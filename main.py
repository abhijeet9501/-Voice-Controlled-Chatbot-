from faster_whisper import WhisperModel
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3 as tts
import re
import os


core = os.cpu_count()
genai.configure(api_key="AIzaSyCpBB_5BjbbekcAaCiU2q1wIi4mBmL-h-g") 

model_whisper = WhisperModel('tiny.en', 
                compute_type="int8",
                device= 'cpu',
                cpu_threads=core,
                num_workers=core)
engine = tts.init()
#rate = engine.getProperty('rate') 
#engine.setProperty('rate', 180)     
voices = engine.getProperty('voices')       
engine.setProperty('voice', voices[0].id)   

generation_config = {
  "temperature": 0.7,
  "top_p": 1,       
  "top_k": 1,
  "max_output_tokens": 2000,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                generation_config=generation_config,
                safety_settings=safety_settings)

convo = model.start_chat() 

prompt = '''Hey Gemini, we're gearing up for some serious AI action. Your mission? Think fast, stay sharp, and be ready to assist at a moment's notice. From analyzing data to providing insights, your job is to be the ultimate assistant. So, let's get to it and make every interaction count.'''

convo.send_message(prompt)

r = sr.Recognizer() 


def get_voice_input():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=2)
        print("Listening...")
        audio = r.listen(source)
        print("Processing...")
        try:
            with open("audio.wav", "wb") as file:
                file.write(audio.get_wav_data())
            segments, _ = model_whisper.transcribe("audio.wav")
            text = "".join(segment.text for segment in segments)
            if len(text.split())==0:
                print(f"Sorry, I was unable to hear it properly.")
            else:
                print(f"User message: {text}\n")
                txt_to_speech(text)
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            print("Error: Unable to recognize speech.")
            return None
        except Exception as e:
            print(f"Error in get_voice_input: {e}")
            return None


def txt_to_speech(text):
    output = model_gem(text)
    print(f"Gemini response: {output}\n")
    engine.say(output)
    engine.runAndWait()

def model_gem(text):
    convo.send_message(text)
    output = convo.last.text
    return output

while True:
    get_voice_input()
