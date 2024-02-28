import gradio as gr
import pytesseract
from PIL import Image
from llama_cpp import Llama
import time
import os
import json

css="""
#col-container {max-width: 500px; margin-left: auto; margin-right: auto;}
#buttons button {min-width: min(120px,100%);}
#col-container-chatbot {max-width: 900px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 10000px;">
    <h1>Amani LLM</h1>
    <p style="text-align: center;">You can chat with Amani AI LLM! <br />
    Amani AI helps you to have a better understanding from your Amani AI profile information! <br /> 
    </p>
</div>
"""


llm = Llama(model_path="./models/model.gguf", n_ctx=4000, n_threads=2, chat_format="chatml",n_gpu_layers=20,device= 'cuda:0')

def fail_or_success(confidence, threshold):
    if round(float(confidence*100), 2) < threshold:
        return "Fail"
    elif round(float(confidence*100), 2) >= threshold:
        return "Success"

def add_kyc_data(identification, document_liveness, nfc_check, biometrics_check, biometric_liveness, face_match, ip_address, device, os_ph, brand, latitude, longtitude):
    prompt_kyc = f"""
            My KYC Information profile checks and user information are listed below:

            Email: woskiva@sharklasers.com verified
            Phone: +905352961076 verified
            Government System database Check: passed
            Turkish ID Latest: passed
            Selfie: passed
            Address: Passed
            Identification: Threshold: 70%, Confidence: {round(float(identification*100), 2)}%, {fail_or_success(identification, 70)}
            Document Liveness: Threshold: 70%, Confidence: {round(float(document_liveness*100), 2)}%, {fail_or_success(document_liveness, 70)}
            NFC: Threshold: 70%, Confidence: {round(float(nfc_check*100), 2)}%, {fail_or_success(nfc_check, 70)}
            Biometrics Check: Threshold: 80%, Confidence: {round(float(biometrics_check*100), 2)}%, {fail_or_success(biometrics_check, 80)}
            Biometric Liveness: Threshold: 80%, Confidence: {round(float(biometric_liveness*100), 2)}%, {fail_or_success(biometric_liveness, 80)}
            Face Match: Threshold: 75%, Confidence: {round(float(face_match*100), 2)}%, {fail_or_success(face_match, 75)}
            Used:
                \tIP Address: {ip_address}
                \tDevice: {device}
                \tOS: {os_ph}
                \tBrand: {brand}
            Location Data:
                \tLatitude: {latitude}
                \tLongtitude: {longtitude}
            AML: passed in all these list 
                Ofac SDN
                EU List
                UK List
                TUR List
                Ofac Consolidated
                UN-AL QAIDA
            """
    return prompt_kyc

def add_questionnaire_data(purpose, amount_of_transaction, employment_status, nationality, industry, payment_source, referred, pep_confirmation, platform, warning_confirmation):
    prompt_questionnaire = f"""
            My questionnaire results are:
           
            What is your purpose for buying/selling coins from us?

            What are the transactions you will do with the coins you buy?
            \t{purpose}

            Approximately how much do you expect to transact with us this year?
            If it is not exact, you can choose an approximate amount.
            \t{amount_of_transaction}

            What is your current occupation?
            Please indicate your current employment status.
            \t{employment_status}

            What is your nationality?
            \t{nationality}

            Which industry do you work in?
            Please indicate the business industries in which you work.
            \t{industry}

            Where does the money you use to make payments come from?
            Please tell us your source of funds, i.e. how you get the money in your account. You can choose more than one option if your funds come from different sources.
            \t{payment_source}

            Are you referred (to buy cryptocurrency) by someone you met online?
            \t{referred}

            PEP and Sanctions Approval
            By verifying with us, please confirm that you are not a Politically Exempt Person (PEP), a close associate or family member of a PEP.
            \t{pep_confirmation}

            Which platform did you find us on?
            \t{platform}

            FRAUD ALERT!
            \tThere are many cases of scams in the cryptocurrency space and %99 of our clients have lost all their coins.
            \t{warning_confirmation}
    """
    return prompt_questionnaire

def ask_summarized_kyc_profile():
    prompt = "Can you summarize my KYC (Know Your Customer) profile, according to my KYC profile information?"
    return prompt, "system_prompt_summary"

def ask_cdd_risk_analysis():
    prompt = "Can you perform CDD (Customer Due Diligence) risk analysis according to my KYC profile information and Questionnaire results?"
    return prompt, "system_prompt_risk_analysis"

global extracted_ocrs
extracted_ocrs = []

def add_text(history, text, image):
    message = ""
    if image is None and text is not None:
        image_upload_status = "You can load your image and get information from it..."
    if image is not None:
        ocr_text = pytesseract.image_to_string(Image.open(image))
        if len(extracted_ocrs) != 0: 
            if ocr_text != extracted_ocrs[-1]:
                extracted_ocrs.append(ocr_text)
                message = "Here is my document optical character recognition (OCR) result: \n\n"
                message = message + ocr_text + "\n" + text
            elif ocr_text == extracted_ocrs[-1]:
                message = text
        elif len(extracted_ocrs) == 0: 
            extracted_ocrs.append(ocr_text)
            message = "Here is my document optical character recognition (OCR) result: \n\n"
            message = message + ocr_text + "\n" + text


        image_upload_status = "Image uploaded and extracted information"
        
        text = message
    history = history + [(text, "")]
    return history, "", image_upload_status, extracted_ocrs

def load_prompts():
    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.json")
    with open(prompt_path, 'r') as file:
        data = json.load(file)
    return data

def generate(history, prompt_key):
    print("Generating response...")
    # Use your existing system_prompt here
    # PROMPT ENGINEERING
    prompts = load_prompts()
    system_prompt = prompts[prompt_key]
    message = history[-1][0]
    formatted_prompt = [{"role": "system", "content": system_prompt}]

    for user_prompt, bot_response  in history:
        formatted_prompt.append({"role": "user", "content": user_prompt})
        formatted_prompt.append({"role": "assistant", "content": bot_response })
    formatted_prompt.append({"role": "user", "content": message})
    
    #print(f"FORMATTED PROMPT FOR LLM =====> {formatted_prompt}")

    stream_response = llm.create_chat_completion(messages=formatted_prompt, temperature=0.5, max_tokens=512, stream=True)
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
                with gr.Column():
                    with gr.Accordion("KYC inputs", open=False) as kyc_row:
                            identification = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.01, interactive=True, label="Idetification Confidence")
                            document_liveness = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01, interactive=True, label="Document Liveness Confidence")
                            nfc_check = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.01, interactive=True, label="NFC Check Confidence")
                            biometrics_check = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01, interactive=True, label="Biometrics Check Confidence")
                            biometric_liveness = gr.Slider(minimum=0.0, maximum=1.0, value=0.85, step=0.01, interactive=True, label="Biometrics Liveness Confidence")
                            face_match = gr.Slider(minimum=0.0, maximum=1.0, value=0.89, step=0.01, interactive=True, label="Face Match Confidence")
                            ip_address = gr.Textbox(label="IP Address", value="198.52.129.197")
                            device = gr.Textbox(label="Device", value="ID4b973ac37effd84b")
                            os_ph = gr.Textbox(label="Operating System", value="android")
                            brand = gr.Textbox(label="Brand", value="HUAWEI")
                            latitude = gr.Textbox(label="Latitude", value="41.10301571015984")
                            longtitude = gr.Textbox(label="Longtitude", value="29.018631657569884")
                    with gr.Accordion("Questionnaire inputs", open=False) as questionnaire_row:
                            purpose = gr.Textbox(label="Transactions", value="Long term investment")
                            amount_of_transaction = gr.Textbox(label="Amont of transaction", value="$100k-$300k USD")
                            employment_status = gr.Textbox(label="Employment Status", value="Employee")
                            nationality = gr.Textbox(label="Nationality", value="Turkish")
                            industry = gr.Textbox(label="Industry you work", value="Other, Information Technology")
                            payment_source = gr.Textbox(label="Payment resource", value="Work salary")
                            referred = gr.Textbox(label="Are you referred by someone", value="NO")
                            pep_confirmation = gr.TextArea(label="PEP confirmation", value="I confirm that I am not a PEP (Politically Exempt Person) and am not under any sanctions.")
                            platform = gr.Textbox(label="Which platform did you find us", value="Binance")
                            warning_confirmation = gr.TextArea(label="Did you read the warnings and accpet them?", value="I read the warning, I understand that a person who asks me to withdraw my coins somewhere other than Binance/Huobi/Bybit/OKX has no logical purpose other than fraud. ")
                with gr.Row(elem_id="add-sum-button"):
                    summarize_btn = gr.Button("Summarize KYC profile")
                    risk_analysis_btn = gr.Button("Create CDD risk analysis")
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
        with gr.Column(elem_id="example-questions", scale=5): 
            gr.Examples(examples=[["./example_documents/bas.jpg", "Can you give me the addres of this document's owner?"], ["./example_documents/PHL_ID_0_F.png", "Can you give me the person's name and date of birth of this document's owner?"] ], inputs=[image_upload, txt])
    prompt_key = gr.Textbox(value="system_prompt", container=False, visible=False)
    txt_msg = txt.submit(add_text, [chatbot, txt, image_upload], [chatbot, txt, image_upload_status], queue=False).then(
        generate, [chatbot, prompt_key], chatbot, api_name="bot_response"
    )
    txt_msg = submit_btn.click(add_text, [chatbot, txt, image_upload], [chatbot, txt, image_upload_status], queue=False).then(
        generate, [chatbot, prompt_key], chatbot, api_name="bot_response"
    )
    add_kyc_information.click(add_kyc_data, [identification, document_liveness, nfc_check, biometrics_check, biometric_liveness, face_match, ip_address, device, os_ph, brand, latitude, longtitude], txt).then(add_text, [chatbot, txt, image_upload], [chatbot, txt, image_upload_status], queue=False).then(
        generate, [chatbot, prompt_key], chatbot, api_name="bot_response"
    )
    add_questionnaire_information.click(add_questionnaire_data, [purpose, amount_of_transaction, employment_status, nationality, industry, payment_source, referred, pep_confirmation, platform, warning_confirmation], txt).then(add_text, [chatbot, txt, image_upload], [chatbot, txt, image_upload_status], queue=False).then(
        generate, [chatbot, prompt_key], chatbot, api_name="bot_response"
    )
    summarize_btn.click(ask_summarized_kyc_profile, outputs=[txt, prompt_key]).then(add_text, [chatbot, txt, image_upload], [chatbot, txt, image_upload_status], queue=False).then(
        generate, [chatbot, prompt_key], chatbot, api_name="bot_response"
    )
    risk_analysis_btn.click(ask_cdd_risk_analysis, outputs=[txt, prompt_key]).then(add_text, [chatbot, txt, image_upload], [chatbot, txt, image_upload_status], queue=False).then(
        generate, [chatbot, prompt_key], chatbot, api_name="bot_response"
    )

demo.queue()
demo.launch()