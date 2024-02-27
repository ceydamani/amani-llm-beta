from llama_cpp import Llama

llm = Llama(model_path="/home/ceyda/Desktop/llm-mistral/test/model.gguf", n_ctx=4000, n_threads=2, chat_format="chatml",n_gpu_layers=20,device= 'cuda:0')


def main():
    print("chat started...")
    history = []

    while True:
        user_input = input()
        #if user_input == "reset":
        #    llm = llm.reset()
        #    llm = Llama(model_path="/home/ceyda/Desktop/llm-mistral/test/model.gguf", n_ctx=4000, n_threads=2, chat_format="chatml",n_gpu_layers=20,device= 'cuda:0')
        if user_input != "reset":
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
            formatted_prompt = [{"role": "system", "content": system_prompt}]

            for user_prompt, bot_response  in history:
                formatted_prompt.append({"role": "user", "content": user_prompt})
                formatted_prompt.append({"role": "assistant", "content": bot_response })
            formatted_prompt.append({"role": "user", "content": user_input})

            response = ""
            stream_response = llm.create_chat_completion(messages=formatted_prompt, temperature=0.3, max_tokens=512, stream=True)
            for chunk in stream_response:
                if len(chunk['choices'][0]["delta"]) != 0 and "content" in chunk['choices'][0]["delta"]:
                    response  += chunk['choices'][0]["delta"]["content"]
            print(f"Q: {user_input}")
            print(f"Response: {response}")
            history = history + [(user_input, response)]
        

if __name__ == "__main__":
    main()