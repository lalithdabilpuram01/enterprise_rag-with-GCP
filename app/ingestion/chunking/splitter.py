from typing import List
import logfire

def _split_long_paragraph(paragraph: str, chunk_size: int) -> List[str]:
    """
    Splits a paragraph that is on its own larger than chunk_size.

    Document AI often returns a whole page as one block with no blank lines, so
    splitting on paragraphs alone can emit a single 100k-character chunk.
    """
    pieces = []
    current = ""

    for word in paragraph.split():
        # A single token longer than the limit (base64 blobs, unspaced tables)
        # still has to be cut somewhere, so slice it hard.

        while len(word)>= chunk_size:
            if current.strip():
                pieces.append(current.strip())
                current = ""
            pieces.append(word[:chunk_size])
            word =  word[chunk_size:]

        if len(current) + len(word) +1 < chunk_size:
            current += word + " "

        else:
            if current.strip():
                pieces.append(current.strip())
            current = word + " "

    if current.strip():
        pieces.append(current.strip())

    return pieces






def chunk_text(text: str, chunk_size: int = 1500)-> List[str]:
    """
    Simple sematic-ish chunker that splits by paragraphs.
    Ensures chunk do not exceed the specified size.
    """

    with logfire.span("Text chunking", text_length=len(text)):
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk= ""

        for p in paragraphs:

            # A paragraph can exceed chunk_size on its own; flush what we have and
            # split it, otherwise it rides through whole and blows the embedding
            # request's token budget.

            if len(p) >= chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.extend(_split_long_paragraph(p, chunk_size))

            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p+ "\n\n"

            else: 
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"Generated {len(valid_chunks)} chunks")
        return valid_chunks
            