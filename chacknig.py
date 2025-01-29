import subprocess

def list_ollama_models():
    # Run the "ollama list" command
  result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
  models = result.stdout.splitlines()
  return models 

# Get the list of models
model_list = list_ollama_models()

# Print the list (optional)
if model_list:
  for i in model_list:
    if (i.split()[0] != "NAME"):
      print(i.split()[0].strip(" "))
      break




































# from langchain_ollama import OllamaLLM

# # Initialize the Ollama client with the model and base_url
# ollama_client = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434")  

# try:
#     # Ask a question to the model (passing the question as a list of strings)
#     question = ["What is the capital of France?"]
#     response = ollama_client.generate(question)  # Sending the question to the model
    
#     # Print the model's response
#     print("Model's Response:", response)
    
# except Exception as e:
#     print(f"An error occurred: {e}")


