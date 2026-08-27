import os
import sys
import uuid
import json
import logfire
import vertexai


from typing import Optional
from google.cloud import storage
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Import Local modules
from app.config import settings
from app.services.retrieval.embedding import embed_texts
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_text

#Initialize Logfire with the Enterprise Ingestion Service Name
logfire.configure(service_name="enterprise-ingestion-services")

# Initialize Vertex AI for embeddings
vertexai.init(project=settings.PROJECT_ID, location=settings.LOCATION)

#Initilize GCS Client
storage_client = storage.Client(project=settings.PROJECT_ID)

# Initialize Qdrant Client

qdrant_client = QdrantClient(
    url= settings.QDRANT_URL,
    api_key= settings.QDRANT_API_KEY
)

# text-embedding-004 returns 768-dimensional vectors
EMBEDDING_DIM = 768

SUPPORTED_EXTENSIONS = {"pdf", "html", "htm", "txt", "docx", "pptx"}

# Stable namespace so re-ingesting a file overwrites its points instead of duplicating them
POINT_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


def upload_to_gcs(data, bucket_name:str , destination_blob_name: str, is_json : bool = False):
    """
    Uploads a file or JSON data to GCS.
    """

    with logfire.span("GCS Upload", bucket= bucket_name, blob = destination_blob_name):
        try: 
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)

            if is_json:
                blob.upload_from_string(json.dumps(data), content_type='application/json')

            else : 
                blob.upload_from_filename(data)
            logfire.info(f"Uploaded to {bucket_name}")

        except Exception as e:
            logfire.error(f"GCS Upload Failed: {e}")
            raise e

def extract_text(file_path: str, ext: str) -> str:
    """
    Dispatches to the right loader for the given extension.
    """
    if ext == "pdf":
        return parse_pdf(file_path)

    if ext in ("html", "htm"):
        return parse_html(file_path)

    if ext == "txt":
        return parse_text(file_path)

    # Imported lazily: `unstructured` is heavy and only needed for Office files.
    from app.ingestion.loaders.office import parse_office
    return parse_office(file_path)

def process_file(file_path: str, filename: str, source_type: str ):
    """
    orchestrates the parsing, chunking, embedding, and indexing of a single file.
    """
    with logfire.span("Processing file", file = filename, source= source_type):
        try:
            # 1. Reject unsupported types before touching GCS
            ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ""
            if ext not in SUPPORTED_EXTENSIONS:
                logfire.warning(f"Skipping unsupported file type: {filename} ")
                return

            # 2. Upload RAW file to GCS
            raw_gcs_path = f"{source_type}/{filename}"
            upload_to_gcs(file_path, settings.RAW_BUCKET, raw_gcs_path)

            # 3. Extract Text based on extension
            full_text = extract_text(file_path, ext)

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} ")
                return

            # 4. Chunk Text

            chunks = chunk_text(full_text)

            if not chunks :
                logfire.warning(f"No chunks produced for {filename} ")
                return

            # 5. Upload PROCESSED metadata to GCS
            processed_data = {"filename": filename, "chunks": chunks, "source_type": source_type}
            processed_gcs_path = f"{source_type}/{filename}.json"
            upload_to_gcs(processed_data, settings.PROCESSED_BUCKET, processed_gcs_path, is_json=True)

            # 6. Embed and Index in Qdrant
            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_texts(chunks)
                points = []
                for i , (chunk, vector) in enumerate(zip(chunks,embeddings)):
                    points.append(models.PointStruct(
                        id=str(uuid.uuid5(POINT_NAMESPACE, f"{source_type}/{filename}#{i}")),
                        vector= vector,
                        payload={
                            "text": chunk,
                            "chunk_index": i,
                            "source": filename,
                            "source_type": source_type,
                            "raw_gcs_path" : f"gs://{settings.RAW_BUCKET}/{raw_gcs_path}"
                        }
                    ))

                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points
                )
                logfire.info(f"Indexed {len(points)} points to Qdrant")

        except Exception as e :
            logfire.error(f"Error processing file {filename}: {e}")
            raise e


def run_universal_ingestion(base_dir: str, explicit_source_type: Optional[str] = None , wipe: bool = False):
    """
    Automatically scans the directory.
    if it has subfolders, maps them to source_types.
    if it has no subfolders, uses the explict_source_type or infers from the folder name.
    """
    with logfire.span(" Universal Ingestion Started", base_directory = base_dir):

        # Handle Collection Wipe
        if wipe :
            with logfire.span("wiping collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f" collection {settings.QDRANT_COLLECTION} deleted")
        # Ensure Collection Exists
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            qdrant_client.create_collection(
                collection_name= settings.QDRANT_COLLECTION,
                vectors_config= models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE)
                )

        # Scan for subfolders (ignore hidden ones)
        #subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        subdirs = [ d for d in sorted(os.listdir(base_dir))
            if not d.startswith('.') and os.path.isdir(os.path.join(base_dir, d))
        ]

        failures = 0

        if not subdirs:
            # If no subdirs, use explict type or infer from the base directory.

            if explicit_source_type:
                source_type = explicit_source_type

            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = "true" if "true" in base_name else "noisy" if "noisy" in base_name else "general"

            logfire.info(f" No Subdirs found, processing {base_dir} as '{source_type}'")

            failures += process_directory(base_dir, source_type)

        else :
            for subdir in subdirs:
                source_type = "true" if "true" in subdir.lower() else "noisy" if "noisy" in subdir.lower() else subdir
                failures += process_directory(os.path.join(base_dir, subdir), source_type)

        if failures:
            logfire.warning(f"Ingestion finished with {failures} failed file(s)")

        return failures

def process_directory(dir_path: str, source_type: str) -> int:
    """
    Process all files in a specific directory. Returns the number of files that failed.

    A single bad file must not abort the whole batch, so failures are logged and counted.
    """
    with logfire.span("Scanning Directory", path=dir_path, source_type= source_type):
        files = [
            f for f in sorted(os.listdir(dir_path))
            if not f.startswith('.') and os.path.isfile(os.path.join(dir_path, f))
        ]
        logfire.info(f"Found {len(files)} files")

        failures = 0
        for filename in files:
            try:
                process_file(os.path.join(dir_path, filename),filename, source_type)
            except Exception as e:
                failures += 1
                logfire.error(f"Skipping {filename} after failure: {e}")

        return failures


if __name__ == "__main__":
    # Usage: python -m app.ingestion.processor [dir_path] [source]      

    wipe_requested = "--wipe" in sys.argv

    clean_args = [a for a in sys.argv if a != "--wipe"]

    # Default to DATA/ if no path provided
    target_dir = clean_args[1] if len(clean_args) >1 else "DATA"
    explict_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.isdir(target_dir):
        print(f"Error: Path {target_dir} is not an existing directory.")
        sys.exit(1)

    failed = run_universal_ingestion(target_dir, explicit_source_type=explict_type, wipe=wipe_requested)
    logfire.info("Universal ingestion Job completed.")

    if failed:
        sys.exit(1)
