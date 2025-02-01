import subprocess
import os

def list_ollama_models():
  # Run the "ollama list" command
  result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
  models = result.stdout.splitlines()

  all_models_name =[]
  if models:
    for i in models:
      if (i.split()[0] != "NAME"):
        all_models_name.append(i.split()[0].strip(" "))
  return all_models_name

def suppourtFormats():
  formates = [i.split(".")[1] for i in os.listdir(os.getcwd()+"/all formats books")]  
  return formates 


if __name__ == "__main__":
  model_list = list_ollama_models()
  
  # print(model_list)
  # print(suppourtFormats())

  for model in model_list:
    print(model)
  

