import gradio as gr
import pytesseract
from PIL import Image
from llama_cpp import Llama
import time

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Amani LLM</h1>
    <p style="text-align: center;">You can chat with Amani AI LLM! <br />
    Amani AI helps you to have a better understanding from your Amani AI profile information! <br /> 
    </p>
</div>
"""

llm = Llama(model_path="/home/ceyda/Desktop/llm-mistral/test/model.gguf", n_ctx=4000, n_threads=2, chat_format="chatml",n_gpu_layers=20,device= 'cuda:0')

def loading_image():
    return "Loading..."

def add_text(history, text):
    history = history + [(text, "")]
    return history, ""

def ocr_and_chat(image, history):
    ocr_text = pytesseract.image_to_string(Image.open(image))
    
    message = "Here is my document optical character recognition (OCR) result: \n"

    message = message + ocr_text

    print(f"OCR HISTORY BEFOR === >>> {history}")
    history = history + [(message, "")]
    print(f"OCR HISTORY AFTER === >>> {history}")

    return "Document uploaded and extracted information", history

def generate(history):
    print("Generating response...")
    # Use your existing system_prompt here
    system_prompt = """
            You are an assistant developed by Amani AI. You will help the user to better explain the validated KYC (Know Your Customer) information obtained by Amani AI.
            If user asks about who you are give one brief sentence about your purpose which is helping users about Amani AI created products and services. 
            
            If user asks information about Amani AI, here is brief information about Amani AI; Amani AI is a RegTech company licensed in Dubai's DIFC, specializing in AI-powered identity verification and 
            biometrics for various sectors, including finance, telecoms, hospitality, travel, HR, transportation, and security. Founded in 2018 by CEO Hamid Khan, CTO Hazem Abdullah, and COO Hamdi Kellecioglu, 
            it offers flexible deployment options and employs a skilled team of engineers with expertise in machine learning, identity verification, artificial intelligence, biometrics,
            anti-money laundering, Know Your Customer (KYC), document management, facial recognition, document validation, customer screening, background checking, 
            compliance, fraud prevention, age verification, and KYC processes.
            
            [INST]
            Translate the questionnaire answer to English if they written in different language.
            If user asks about their KYC profile summary; 
                Summarize user's KYC profile as paragraph only using the KYC profile information, if profile has failed or overwritten validation step highlight (write it bold) it like a warning. 
            If user asks about their Customer Due Diligence (CDD) risk analysis; you should explain the overall risk for the profile from the user's KYC profile information and questionnaire answers. 
                For CDD risk analysis give user their risk in type (low-mid-high) for each step in both KYC profile and questionnaire and briefly explain. 
            If user asks how can they approve their profile review the profile and give short feedback (approve or reject) according to their risk score.
            [/INST]
    """
    message = history[-1][0]
    formatted_prompt = [{"role": "system", "content": system_prompt}]

    for user_prompt, bot_response  in history:
        formatted_prompt.append({"role": "user", "content": user_prompt})
        formatted_prompt.append({"role": "assistant", "content": bot_response })
    formatted_prompt.append({"role": "user", "content": message})
    stream_response = llm.create_chat_completion(messages=formatted_prompt, temperature=0.3, max_tokens=512, stream=True)
    print(stream_response)
    for chunk in stream_response:
        if len(chunk['choices'][0]["delta"]) != 0 and "content" in chunk['choices'][0]["delta"]:
            history[-1][1]  += chunk['choices'][0]["delta"]["content"]
            time.sleep(0.05)
            yield history

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        with gr.Column():
            image_upload = gr.Image(type="filepath", label="Upload Image for OCR")
            with gr.Row():
                image_upload_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_image = gr.Button("Load your image to Amanai LLM")
        chatbot = gr.Chatbot(elem_id='chatbot',
                               avatar_images=["user.png", "amani.jpeg"], 
                               bubble_full_width=False, 
                               show_label=False, 
                               show_copy_button=True, 
                               likeable=True)
        # create button
        txt = gr.Textbox(label="Chat", placeholder="Type your message...")
        # submit button
        submit_btn = gr.Button("Send Message")
        # pipline
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        generate, chatbot, chatbot, api_name="bot_response"
    )
    load_image.click(loading_image, None, image_upload_status, queue=False)
    load_image.click(ocr_and_chat, inputs=[image_upload, chatbot], outputs=[image_upload_status, chatbot], queue=True).then(
        generate, chatbot, chatbot
    )

demo.queue()
demo.launch()