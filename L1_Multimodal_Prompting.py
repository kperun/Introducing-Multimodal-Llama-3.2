import base64
import warnings

from utils import disp_image
from utils import llama31
# We want to show the difference between llama 3.1 and 3.2, so we load both
from utils import llama32
from utils import load_env

warnings.filterwarnings('ignore')
load_env()


## Text input only question
def textOnlyInputMessage():
    messages = [
        {"role": "user",
         "content": "Who wrote the book Charlotte's Web?"}
    ]

    # Second argument is the number of parameters, e.g., 90 billions model here
    response_32 = llama32(messages, 90)
    print(response_32)

    response_31 = llama31(messages, 70)
    print(response_31)

    ## Reprompting with new question
    messages = [
        {"role": "user",
         "content": "Who wrote the book Charlotte's Web?"},
        {"role": "assistant",
         "content": response_32},  # we add the previous response to the history
        {"role": "user",
         "content": "3 of the best quotes"}
    ]
    response_32 = llama32(messages, 90)
    print(response_32)
    response_31 = llama31(messages, 70)
    print(response_31)


def visionInputImageFromUrl():
    image_url = ("https://raw.githubusercontent.com/meta-llama/"
                 "llama-models/refs/heads/main/Llama_Repo.jpeg")
    messages = [
        {"role": "user",
         "content": [
             {"type": "text",
              "text": "describe the image in one sentence"
              },
             {"type": "image_url",
              "image_url": {"url": image_url}  # we hand over the image as an input
              }
         ]
         },
    ]
    disp_image(image_url)
    result = llama32(messages, 90)
    print(result)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def visionInputImageFromLocal():
    # We need the image as in base64 handed over to the request, because then it can be interpreted
    base64_image = encode_image("images/Llama_Repo.jpeg")
    messages = [
        {"role": "user",
         "content": [
             {"type": "text",
              "text": "describe the image in one sentence"
              },
             {"type": "image_url",
              "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}  # add the image as a file stream
              }
         ]
         },
    ]
    disp_image(f"data:image/jpeg;base64,{base64_image}")
    result = llama32(messages, 90)
    print(result)
    # continue the conversation with a followup question
    messages = [
        {"role": "user",
         "content": [
             {"type": "text",
              "text": "describe the image in one sentence"
              },
             {"type": "image_url",
              "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
              }
         ]
         },
        {"role": "assistant", "content": result},
        {"role": "user", "content": "how many of them are purple?"}
    ]
    result = llama32(messages)
    print(result)


def llama32pi(prompt, image_url, model_size=90):
    '''
    We can define the prompting for an image in a function which expects the model size, the prompt and the image
    to encapsulate the message.
    :param prompt: the prompt we want to execute on the image
    :param image_url: the url to the image (or base 64 encoded)
    :param model_size: the model size, e.g., 70 or 90 B
    :return: the message reponse
    '''
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": prompt},
                {"type": "image_url",
                 "image_url": {
                     "url": image_url}
                 }
            ]
        },
    ]
    result = llama32(messages, model_size)
    return result


def visionInputFromFunctionExamples():
    # Describe the image
    print(llama32pi("describe the image in one sentence",
                    "https://raw.githubusercontent.com/meta-llama/"
                    "llama-models/refs/heads/main/Llama_Repo.jpeg"))
    # Analyze dog breed
    question = (("What dog breed is this? Describe in one paragraph,"
                 "and 3-5 short bullet points"))
    base64_image = encode_image("images/ww1.jpg")
    result = llama32pi(question, f"data:image/jpg;base64,{base64_image}")
    print(result)
    # Analyze the same breed from a different image
    base64_image = encode_image("images/ww2.png")
    result = llama32pi(question, f"data:image/png;base64,{base64_image}")
    print(result)
    # Analyse image to retrieve information
    question = (("What's the problem this is about?"
                 " What should be good numbers?"))
    base64_image = encode_image("images/tire_pressure.png")
    result = llama32pi(question, f"data:image/png;base64,{base64_image}")
    print(result)


if __name__ == '__main__':
    print("Text only:" + '*'*100)
    textOnlyInputMessage()
    print("Vision from from url" + '*' * 100)
    visionInputImageFromUrl()
    print("Vision examples from local file:" + '*' * 100)
    visionInputImageFromLocal()
    print("Vision examples:" + '*' * 100)
    visionInputFromFunctionExamples()