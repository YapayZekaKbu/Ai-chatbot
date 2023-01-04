
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import pygame
import language_tool_python 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

r = sr.Recognizer()

def speak(text, language='en'):
        mp3_fo = BytesIO()
        tts = gTTS(text, lang=language)
        tts.write_to_fp(mp3_fo)
        pygame.mixer.music.load(mp3_fo, 'mp3')
        pygame.mixer.music.play()
            
def speechToText():
    try:
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.2)
            print("Please start speaking, I am Listening...\n")
            r.pause_threshold = 1
            audio2 = r.listen(source2,timeout=3, phrase_time_limit=11)# Microphone listening time is not very long because the model should be used with short sentences
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()
            return MyText
    except sr.RequestError as e:
        return ("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        return "unknown error occurred"
    


######################### DO NOT CHANGE#########################
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
######################### DO NOT CHANGE#########################

## for dongusunu silmeyin ##
## Do not remove the for loop ##
for step in range(4):

    ### Burada, cümleyi kullanıcıdan yazarak aliyoruz ama biz mikrofondan alacağız sonra tekrar metne donusturecegiz
    ### Here,sentence is taken from user by typing, but we will take it from microphone and then convert it to text
    print("1. text input\n2. voice input")
    n = int(input())
    if(n==1):
        text = input(">> You:")
    elif(n==2):
        text = speechToText()
        print(">> You:", text)
    else:
        print("your input is wrong try")
        break
    ######################### DO NOT CHANGE#########################
    input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids
    chat_history_ids = model.generate(
    bot_input_ids,
    max_length=1000,
    do_sample=True,
    top_p=0.9,
    top_k=50,
    temperature=0.5,
    pad_token_id=tokenizer.eos_token_id
    )   #Project Leader's edit on chat_history_ids: adjusting top_p and top_k for better parameters.
    ######################### DO NOT CHANGE#########################    

    ### Kullanici "correct my grammar" dedi ise eger, gramer duzeltme islemini yapacagiz ve islemin sonucu output'umuz olacak"
    ### when the user says "correct my grammar", we will do the grammar correction and the result of this operation will be our output"
    if text == "Correct my grammar":
        print("Choose how to input the sentence you want to correct :\n 1. text input\n2. voice input")
        n = int(input())
        if(n==1):
            text = input(">> You:")
        elif(n==2):
            text = speechToText()
            print(">> You:", text)
        else:
            print("your input is wrong try")

        my_tool = language_tool_python.LanguageTool('en-US') 
        my_text = text
        correct_text = my_tool.correct(my_text)  
        # correction 
        if  correct_text.lower() == text.lower(): 
            correct_text = "You said \"" + text + "\" and it is correct."
            pygame.init()
            pygame.mixer.init()
            output = speak(correct_text)
            print(f"{correct_text}") 
        else:
            correct_text = "The correct way to say that: \"" + correct_text + "\""
            pygame.init()
            pygame.mixer.init()
            output = speak(correct_text)
            print(f"{correct_text}") 
            print(f"What you said: {text}")
    else:
        ### Kullanici gramer duzeltme islemini belirtmedi ise, output budur
        ### If the user did not specify the grammar correction command, this is our output 
        output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        pygame.init()
        pygame.mixer.init()
        # sound.seek(0)
        speak(output)
        #### Output'u metin olarak değil, metin olarak da cikartacagiz
        #### We will display the output as a voice and also as text
        print(f"DialoGPT: {output}\n")
