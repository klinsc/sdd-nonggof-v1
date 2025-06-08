import base64
import os
from io import BytesIO
from typing import Callable

from openai import OpenAI
from PIL import Image
from typhoon_ocr.ocr_utils import get_anchor_text, render_pdf_to_base64png

PROMPTS_SYS = {
    "default": lambda base_text: (
        f"Below is an image of a document page along with its dimensions. "
        f"Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
        f"If the document contains images, use a placeholder like dummy.png for each image.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    ),
    "structure": lambda base_text: (
        f"Below is an image of a document page, along with its dimensions and possibly some raw textual content previously extracted from it. "
        f"Note that the text extraction may be incomplete or partially missing. Carefully consider both the layout and any available text to reconstruct the document accurately.\n"
        f"Your task is to return the markdown representation of this document, presenting tables in HTML format as they naturally appear.\n"
        f"If the document contains images or figures, analyze them and include the tag <figure>IMAGE_ANALYSIS</figure> in the appropriate location.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    ),
}


def get_prompt(prompt_name: str) -> Callable[[str], str]:
    """
    Fetches the system prompt based on the provided PROMPT_NAME.

    :param prompt_name: The identifier for the desired prompt.
    :return: The system prompt as a string.
    """
    return PROMPTS_SYS.get(prompt_name, lambda x: "Invalid PROMPT_NAME provided.")


def get_total_pages(filename: str) -> int:
    """
    Retrieves the total number of pages in a PDF file.

    :param filename: The path to the PDF file.
    :return: The total number of pages in the PDF.
    """
    from pypdf import PdfReader

    with open(filename, "rb") as f:
        reader = PdfReader(f)
        return len(reader.pages)


def extract_text_and_image_from_pdf(
    original_file_path, output_json_path, markdown=True
):
    if not os.path.exists(original_file_path):
        raise FileNotFoundError(f"The file {original_file_path} does not exist.")

    # Use uitils to get the total number of pages in the PDF
    total_pages = get_total_pages(original_file_path)

    # Set the task type to "default" for now.
    task_type = "default"

    # Create the text_content variable to store the extracted text.
    text_content = ""

    # Iterate through each page in the PDF
    for page_num in range(total_pages):
        page = page_num + 1  # Page numbers are 1-based in the prompt

        # Render the first page to base64 PNG and then load it into a PIL image.
        image_base64 = render_pdf_to_base64png(
            original_file_path, page, target_longest_image_dim=1800
        )
        # image_pil = Image.open(BytesIO(base64.b64decode(image_base64)))

        # Extract anchor text from the PDF (first page)
        anchor_text = get_anchor_text(
            original_file_path, page, pdf_engine="pdfreport", target_length=8000
        )

        # Retrieve and fill in the prompt template with the anchor_text
        prompt_template_fn = get_prompt(task_type)
        PROMPT = prompt_template_fn(anchor_text)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ]
        # send messages to openai compatible api
        openai = OpenAI(
            base_url="https://api.opentyphoon.ai/v1",
            api_key="sk-mldBudg0MTN8dAPxxt9woS31M2KPid8BKzqlDUgzf0OTZevM",
        )
        response = openai.chat.completions.create(
            model="typhoon-ocr-preview",
            messages=messages,
            max_tokens=16384,
            temperature=0.1,
            top_p=0.6,
            extra_body={
                "repetition_penalty": 1.2,
            },
        )
        text_output = response.choices[0].message.content
        print(text_output)

        # # Save the output to a file
        # filename_without_ext = os.path.splitext(os.path.basename(filename))[0]
        # output_filename = f"{filename_without_ext}_output_{page}.json"
        # with open(output_filename, "w", encoding="utf-8") as f:
        #     f.write(text_output)
        # print(f"Output for page {page} saved to {output_filename}")

        # Append the text output to the text_content variable
        text_content += text_output + "\n"

        # If the text output contains word "จึงเรียน", that means the page is the last page.
        if "จึงเรียน" in text_output:
            print("Last page detected, stopping further processing.")
            break

    # Prepare the final output in JSON format
    final_output = {
        "natural_text": text_content.strip(),
        "markdown": markdown,
        "task_type": task_type,
        "total_pages": total_pages,
        "original_file_path": original_file_path,
        "output_json_path": output_json_path,
    }

    # Save the final output to the specified JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        import json

        json.dump(final_output, f, ensure_ascii=False, indent=4)


# extract_text_and_image_from_pdf(
#     "test.pdf", output_json_path="test_orc.json", markdown=True
# )
