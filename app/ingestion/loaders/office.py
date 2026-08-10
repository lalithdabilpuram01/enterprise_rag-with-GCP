import logfire
from unstructured.partition.auto import partition




def parse_office(file_path: str):
    """
    Parses Office Document (.docx, .pptx) using the unstructured library.
    unlike pdfs, these formats are structured and lightweight, so they are processed locally.
    """

    with logfire.span(f"Office document parsing", filename = file_path):
        try: 
            # unstructured automatically detects if it's docx or pptx
            elements = partition(filename=file_path)
            full_text = '\n'.join([str(el) for el in elements])

            if not full_text.strip():
                logfire.warning(f"Unstructured returned empty text for {file_path}.")

            else :
                logfire.info(f"Successfully parsed {len(full_text)} characters")

            return full_text
        except Exception as e:
            logfire.error(f" Office parse failed : {e}")
            raise e