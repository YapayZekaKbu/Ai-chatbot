from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from gtts import gTTS
from io import BytesIO
import pygame
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

######################### DO NOT CHANGE#########################
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
######################### DO NOT CHANGE#########################

## for dongusunu silmeyin ##
## Do not remove the for loop ##
for step in range(5):


    ### Burada, cümleyi kullanıcıdan yazarak aliyoruz ama biz mikrofondan alacağız sonra tekrar metne donusturecegiz
    ### Here,sentence is taken from user by typing, but we will take it from microphone and then convert it to text
    text = input(">> You:")

    ######################### DO NOT CHANGE#########################
    input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        do_sample=True,
        top_p=0.95,
        top_k=0,
        temperature=0.75,
        pad_token_id=tokenizer.eos_token_id
    )
    ######################### DO NOT CHANGE#########################
   

    ### Kullanici gramer duzeltme islemini belirtmedi ise, bu output degismeyecek
    ### If the user did not specify the grammar correction command, this is our output 
    output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    ### Kullanici "correct my grammar" dedi ise eger, gramer duzeltme islemini yapacagiz ve islemin sonucu output'umuz olacak"
    ### when the user says "correct my grammar", we will do the grammar correction and the result of this operation will be our output"
   

    #### Output'u metin olarak değil, ses olarak cikartacagiz
    #### We will display the output as a voice not as a text 
    print(f"DialoGPT: {output}")
