from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from rfc_rag.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, PROJ_ROOT

import json
import xml.etree.ElementTree as ET


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks of up to max_tokens words.
    """
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks


app = typer.Typer()


@app.command()
def main(
    # Input directory containing unzipped RFC XML files
    input_path: Path = RAW_DATA_DIR / "xmlsource-all",
    # Output JSONL file for chunked RFC data
    output_path: Path = PROCESSED_DATA_DIR / "rfc_chunks.jsonl",
):
    # Identify XML input directory or file
    xml_root = input_path
    if xml_root.is_file():
        xml_files = [xml_root]
    else:
        xml_files = sorted(xml_root.glob("*.xml"))

    all_chunks: list[dict] = []
    logger.info(f"Found {len(xml_files)} RFC XML files. Parsing and chunking...")
    for xml_file in tqdm(xml_files, desc="RFC files"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # Identifier from filename (e.g., 'rfc8650')
        identifier = xml_file.stem
        # Extract title (first title element in front)
        title_elem = root.find('.//title')
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
        # Process each section
        for sec in root.findall('.//section'):
            sec_name = sec.findtext('name', default='').strip()
            # Gather text from <t> elements (paragraph text)
            texts: list[str] = []
            for elem in sec.findall('.//t'):
                if elem.text:
                    texts.append(elem.text.strip())
            section_text = "\n".join(texts)
            # Chunk and collect entries
            for idx, chunk in enumerate(chunk_text(section_text)):
                entry = {
                    "rfc_id": identifier,
                    "title": title,
                    "section": sec_name,
                    "chunk_id": idx,
                    "text": chunk,
                    "metadata": {
                        "file": str(xml_file.relative_to(PROJ_ROOT)),
                        "section_name": sec_name,
                        "chunk_index": idx,
                    },
                }
                all_chunks.append(entry)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write as JSON lines for embedding
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in all_chunks:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.success(f"Wrote {len(all_chunks)} chunks to {output_path}")


if __name__ == "__main__":
    app()
