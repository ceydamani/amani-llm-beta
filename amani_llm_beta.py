import gradio as gr
import pytesseract
from PIL import Image
from llama_cpp import Llama
import time

css="""
#col-container {max-width: 500px; margin-left: auto; margin-right: auto;}
#buttons button {min-width: min(120px,100%);}
#col-container-chatbot {max-width: 900px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Amani LLM</h1>
    <p style="text-align: center;">You can chat with Amani AI LLM! <br />
    Amani AI helps you to have a better understanding from your Amani AI profile information! <br /> 
    </p>
</div>
"""

llm = Llama(model_path="/home/ceyda/Desktop/llm-mistral/test/models/model.gguf", n_ctx=4000, n_threads=2, chat_format="chatml",n_gpu_layers=20,device= 'cuda:0')

def add_kyc_data():
    prompt_kyc = """
            My KYC Information profile checks and user information are listed below:

            Email :woskiva@sharklasers.com verified
            Phone :+905352961076 verified
            Government System database Check: passed
            Turkish ID Latest: passed
            Selfie: passed
            Address: Passed
            Identification:
                Threshold: 70% Confidence: 100.00%
                Success
            Document Liveness:
                Threshold: 70% Confidence: 100.00%
                Success
            NFC:
                Threshold: 0% Confidence: 0.00%
                Fail
            Biometrics Check:
                Threshold: 80% Confidence: 100.00%
                Success
            Biometric Liveness:
                Threshold: 80% Confidence: 100.00%
                Success
            Face Match:
                Threshold: 75% Confidence: 50.00%
                Fail
            Used:
                IP Address: 198.52.129.197
                Device: ID4b973ac37effd84b
                OS: android
                Brand: HUAWEI
            Location Data
                Latitude: 41.10301571015984
                Longtitude: 29.018631657569884
            AML: passed in all these list 
                Ofac SDN
                EU List
                UK List
                TUR List
                Ofac Consolidated
                UN-AL QAIDA
            """
    return prompt_kyc

def add_questionnaire_data():
    prompt_questionnaire = """
            My questionnaire results are:
           
            Bizden coin alım/satım yapma amacınız nedir?

            Satın aldığınız coinler ile yapacağınız işlemler neler?

            Uzun vadeli yatırım

            Bu yıl bizimle yaklaşık oalrak ne kadarlık bir işlem yapmayı bekliyorsunuz?

            Eğer kesin değilse yaklaşık bir tutar seçebilirsiniz.

            $100k-$300k USD

            Şu anki mesleğiniz nedir?

            Lütfen mevcut istihdam durumunuzu belirtin.

            Çalışan

            Uyruğunuz nedir?

            Turkish

            Hangi endüstride çalışıyorsunuz?

            Lütfen görev aldığınız iş endüstrilerini belirtiniz.

            Diğer

            Bilgi Teknolojileri

            Ödeme yapmak için kullandığınız para nereden geliyor?

            Lütfen bize fon kaynağınızı belirtin, diğer bir deyişle hesabınızdaki parayı nasıl elde ettiğinizi. Fonlarınız farklı kaynaklardan geliyorsa birden fazla seçenek seçebilirsiniz.

            İş Maaşı

            Sizi, çevrimiçi tanıştığınız birisi (kripto para satın almak için) yönlendiriyor mu?

            HAYIR

            PEP ve Yaptırımlar Onayı

            Bizimle doğrulayarak, lütfen Politik Ayrıcalıklı Kişi (PEP) olmadığınızı, bir PEP'in yakın ilişkisi veya ailesi olmadığınızı onaylayın.

            Bir PEP (Politik Ayrıcalıklı Kişi) olmadığımı ve herhangi bir yaptırım altında olmadığımı onaylıyorum.

            Bizi hangi platformdan buldunuz?

            Binance

            DOLANDIRICILIK UYARISI!

            Kripto para alanında pek çok dolandırıcılık vakası bulunmaktadır ve müşterilerimizin %99'unun tüm paralarını kaybettikleri gözlemlenmiştir.

            Uyarıyı okudum, coinlerimi Binance/Huobi/Bybit/OKX dışında başka bir yere çekmemi isteyen bir kişinin dolandırıcılık dışında mantıklı bir amacı olmadığını anlıyorum. 
    """
    return prompt_questionnaire

def add_text(history, text, image):
    if image is None and text is not None:
        image_upload_status = "You can load your image and get information from it..."
    if image is not None:
        ocr_text = pytesseract.image_to_string(Image.open(image))

        message = "Here is my document optical character recognition (OCR) result: \n\n"

        message = message + ocr_text + text

        image_upload_status = "Image uploaded and extracted information"
        
        text = message
    history = history + [(text, "")]
    return history, "", image_upload_status

def generate(history):
    print("Generating response...")
    # Use your existing system_prompt here
    # PROMPT ENGINEERING
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
    #print(f"MESSAGE ====> {message}")
    formatted_prompt = [{"role": "system", "content": system_prompt}]

    for user_prompt, bot_response  in history:
        formatted_prompt.append({"role": "user", "content": user_prompt})
        formatted_prompt.append({"role": "assistant", "content": bot_response })
    formatted_prompt.append({"role": "user", "content": message})
    
    #print(f"FORMATTED PROMPT FOR LLM =====> {formatted_prompt}")

    stream_response = llm.create_chat_completion(messages=formatted_prompt, temperature=0.3, max_tokens=512, stream=True)
    for chunk in stream_response:
        if len(chunk['choices'][0]["delta"]) != 0 and "content" in chunk['choices'][0]["delta"]:
            history[-1][1]  += chunk['choices'][0]["delta"]["content"]
            time.sleep(0.05)
            yield history

with gr.Blocks(css=css) as demo:
    gr.HTML(title)
    with gr.Row():
        with gr.Column(elem_id="col-container", scale=3):
            with gr.Column():
                image_upload = gr.Image(type="filepath", label="Upload Image for OCR")
                with gr.Row():
                    image_upload_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                add_kyc_information = gr.Button("Add KYC Information")
                add_questionnaire_information = gr.Button("Add questionnaire results")
        with gr.Column(elem_id="col-container-chatbot", scale=8):
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
            # pipeline
    txt_msg = txt.submit(add_text, [chatbot, txt, image_upload], [chatbot, txt, image_upload_status], queue=False).then(
        generate, chatbot, chatbot, api_name="bot_response"
    )
    add_kyc_information.click(add_kyc_data, None, txt).then(add_text, [chatbot, txt, image_upload], [chatbot, txt, image_upload_status], queue=False).then(
        generate, chatbot, chatbot, api_name="bot_response"
    )
    add_questionnaire_information.click(add_questionnaire_data, None, txt).then(add_text, [chatbot, txt, image_upload], [chatbot, txt, image_upload_status], queue=False).then(
        generate, chatbot, chatbot, api_name="bot_response"
    )


demo.queue()
demo.launch()