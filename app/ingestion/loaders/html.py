from bs4 import BeautifulSoup
import logfire

def parse_html(file_path: str):
    """
    Parse HTML content using BeautifulSoup.
    cleans scripts, styles, and extracts readable text for RAG.
    """

    with logfire.span("HTML Parsing", file_path= file_path):
        try: 
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            soup = BeautifulSoup(content, "html.parser")

            # 1. Remove Junk(Scripts, Styles, Metadata)
            for scripts in soup(["script","style", "meta", "noscript"]):
                scripts.decompose()

            # 2. Extract Text
            text = soup.get_text(separator="\n")

            # 3. Clean Whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_clean = '\n'.join(chunk for chunk in chunks if chunk)


            return text_clean

        except Exception as e:
            logfire.error(f" HTML Parser Failed: {e} ")
            raise e
