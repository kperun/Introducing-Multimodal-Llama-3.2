import warnings
from utils import disp_image
from utils import load_env
from utils import llama32
import base64

warnings.filterwarnings('ignore')
load_env()

'''
L2 is a collection of use cases possible with the vision feature of Llama 3.2.
'''


def encode_image(image_path):
    """
    Images handed over to llama have to be encoded in base64.
    :param image_path: the path of the image on the hard drive.
    :return: the base 64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def llama32pi(prompt, image_url, model_size=90):
    """
    This function takes a prompt and the image to prompt again.
    :param prompt: the prompt
    :param image_url: the image url or base64 encoded image
    :param model_size: the image size
    :return: a response from the llm
    """
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


def read_receipts_uc():
    """
    Use Case 1: Read receipts from images and prints information from the image.
    :return:
    """
    for i in range(1, 4):
        disp_image(f"images/receipt-{i}.jpg")
    question = "What's the total charge in the receipt?"
    results = ""
    for i in range(1, 4):
        base64_image = encode_image(f"images/receipt-{i}.jpg")
        res = llama32pi(question, f"data:image/jpeg;base64,{base64_image}")
        results = results + f"{res}\n"
    print(results)


def read_multiple_images():
    """
    Use Case 2: It is possible to combine several images into one to execute one LLM call. It will then work on
    all images in parallel. It is also possible to ask questions about relations between the images.
    :return:
    """
    from utils import merge_images
    import matplotlib.pyplot as plt
    # Merge the images, i.e., create a new image where all images are placed next to each other
    merged_image = merge_images("images/receipt-1.jpg",
                                "images/receipt-2.jpg",
                                "images/receipt-3.jpg")
    plt.imshow(merged_image)
    plt.axis('off')
    plt.show()

    from utils import resize_image
    # We have to resize the image to 1120 height as only then llama is able to process it correctly
    resized_img = resize_image(merged_image)

    base64_image = encode_image("images/resized_image.jpg")
    question = "What's the total charge of all the receipts below?"
    result = llama32pi(question,
                       f"data:image/jpeg;base64,{base64_image}")
    print(result)


def right_drink_uc():
    question = "I am on a diet. Which drink should I drink?"
    base64_image = encode_image("images/drinks.png")
    result = llama32pi(question, f"data:image/png;base64,{base64_image}")
    print(result)
    question = ("Generate nutrition facts of the two drinks "
                "in JSON format for easy comparison.")
    result = llama32pi(question, f"data:image/png;base64,{base64_image}")
    print(result)


def explain_diagram_uc():
    question = ("I see this diagram in the Llama 3 paper. "
                "Summarize the flow in text and then return a "
                "python script that implements the flow.")
    base64_image = encode_image("images/llama32mm.png")
    result = llama32pi(question, f"data:image/png;base64,{base64_image}")
    print(result)


def generate_html_uc():
    question = "Convert the chart to an HTML table."
    base64_image = encode_image("images/llama31speed.png")
    result = llama32pi(question, f"data:image/png;base64,{base64_image}")
    print(result)


def generate_cooking_suggestions_uc():
    question = ("What are in the fridge? What kind of food can be made? Give "
                "me 2 examples, based on only the ingredients in the fridge.")
    base64_image = encode_image("images/fridge-3.jpg")
    result = llama32pi(question, f"data:image/jpg;base64,{base64_image}")
    print(result)


def interior_design_uc():
    question = ("Describe the design, style, color, material and other "
                "aspects of the fireplace in this photo. Then list all "
                "the objects in the photo.")
    base64_image = encode_image("images/001.jpeg")
    result = llama32pi(question, f"data:image/jpeg;base64,{base64_image}")
    print(result)


def math_grader_uc():
    prompt = ("Check carefully each answer in a kid's math homework, first "
              "do the calculation, then compare the result with the kid's "
              "answer, mark correct or incorrect for each answer, and finally"
              " return a total score based on all the problems answered.")
    base64_image = encode_image("images/math_hw3.jpg")
    result = llama32pi(prompt, f"data:image/jpg;base64,{base64_image}")
    print(result)


def tool_calling_uc():
    question = "Where is the location of the place shown in the picture?"
    base64_image = encode_image("images/golden_gate.png")
    result = llama32pi(question, f"data:image/png;base64,{base64_image}")
    print(result)
    weather_question = ("What is the current weather in the location "
                        "mentioned in the text below: \n"  f"{result}")
    print(weather_question)

    from datetime import datetime

    current_date = datetime.now()
    formatted_date = current_date.strftime("%d %B %Y")

    # For correct tool calling, we need to define the message with the correct structure
    messages = [
        {"role": "system",
         # the content is important here as it tells the LLM which tools are available
         "content": f"""
            Environment: ipython
            Tools: brave_search, wolfram_alpha
            Cutting Knowledge Date: December 2023
            Today Date: {formatted_date}
            """
         },
        {"role": "user",
         "content": weather_question}
    ]
    # it will return a call to the predefined, ootb calls brave_search. however, this call is not executed yet
    print(llama32(messages))
