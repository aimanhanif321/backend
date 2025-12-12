import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------
# CONFIG
# -------------------------------------
SITEMAP_URL = "https://aimanhanif321.github.io/AI-Humanoid-Robotics-Book/sitemap.xml"

co = cohere.Client(os.getenv("COHERE_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION_NAME = "humanoid_ai_book"

# -------------------------------------
# UTIL FUNCTIONS
# -------------------------------------

def get_all_urls(sitemap_url):
    """Extract all URLs from sitemap."""
    xml = requests.get(sitemap_url).text
    root = ET.fromstring(xml)

    urls = []
    for node in root.iter():
        if node.tag.endswith("loc"):
            urls.append(node.text.strip())

    print("\nFOUND URLS:")
    for u in urls:
        print(" -", u)

    return urls


def fetch_text(url):
    """Download HTML + extract clean readable text with BeautifulSoup."""
    print(f"Processing: {url}")

    try:
        html = requests.get(url, timeout=10).text
    except:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # Remove irrelevant elements
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.extract()

    text = soup.get_text(separator="\n").strip()

    # Normalize whitespace
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

    return text


def chunk_text(text, max_len=1200):
    """Split long text into clean chunks."""
    chunks = []
    while len(text) > max_len:
        split_pos = text.rfind(".", 0, max_len)
        if split_pos == -1:
            split_pos = max_len

        chunks.append(text[:split_pos])
        text = text[split_pos:]

    if text.strip():
        chunks.append(text.strip())

    return chunks


def embed(text):
    """Convert text to Cohere embedding."""
    resp = co.embed(
        model="embed-english-v3.0",
        texts=[text],
        input_type="search_document"
    )
    return resp.embeddings[0]


# -------------------------------------
# MAIN INGEST FUNCTION
# -------------------------------------
def ingest_book():
    urls = get_all_urls(SITEMAP_URL)

    # Create collection
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

    point_id = 1

    for url in urls:
        text = fetch_text(url)
        if not text:
            print("   ❌ No text extracted")
            continue

        chunks = chunk_text(text)

        for chunk in chunks:
            vector = embed(chunk)

            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(id=point_id, vector=vector, payload={"text": chunk})]
            )

            print(f"   Saved chunk {point_id}")
            point_id += 1

    print("\n✔️ Ingestion completed!")
    print(f"Total chunks stored: {point_id - 1}")


# RUN
if __name__ == "__main__":
    ingest_book()





