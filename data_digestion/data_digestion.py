import os
import subprocess
import argparse
import base64
import json
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# Lazy-loaded Qwen2-VL model/processor (shared HF cache with other scripts)
_VLM_MODEL = None
_VLM_PROCESSOR = None
_VLM_DEVICE = None


def _snapshot_is_complete(snapshot_path):
    index_path = os.path.join(snapshot_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return False
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    except Exception:
        return False
    weight_map = index.get("weight_map", {})
    if not weight_map:
        return False
    for shard_file in set(weight_map.values()):
        if not os.path.exists(os.path.join(snapshot_path, shard_file)):
            return False
    return True


def resolve_local_model_path(model_name):
    if not os.path.isdir(model_name):
        return model_name
    # If a snapshot directory is provided directly, use it.
    processor_files = {
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    }
    if any(os.path.exists(os.path.join(model_name, f)) for f in processor_files):
        return model_name
    # If this is a HF cache repo dir, pick the newest snapshot.
    snapshots_dir = os.path.join(model_name, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return model_name
    snapshot_paths = [
        os.path.join(snapshots_dir, d)
        for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    ]
    if not snapshot_paths:
        return model_name
    snapshot_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for snapshot_path in snapshot_paths:
        if _snapshot_is_complete(snapshot_path):
            return snapshot_path
    return snapshot_paths[0]


def get_vlm(model_name="Qwen/Qwen2-VL-7B-Instruct", device="auto", local_files_only=False):
    """
    Load Qwen2-VL model and processor lazily.
    device: 'auto' | 'cuda' | 'mps' | 'cpu'
    """
    global _VLM_MODEL, _VLM_PROCESSOR, _VLM_DEVICE
    if _VLM_MODEL is not None:
        return _VLM_MODEL, _VLM_PROCESSOR, _VLM_DEVICE
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    resolved_model = resolve_local_model_path(model_name)
    print(f"Loading VLM '{resolved_model}' on device '{device}' ...")
    _VLM_PROCESSOR = AutoProcessor.from_pretrained(
        resolved_model, local_files_only=local_files_only
    )
    _VLM_MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
        resolved_model,
        torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32,
        device_map=device if device in ("cuda", "mps") else None,
        local_files_only=local_files_only,
    ).eval()
    _VLM_DEVICE = device
    return _VLM_MODEL, _VLM_PROCESSOR, _VLM_DEVICE


class DataDigester:
    def __init__(self):
        pass

    def convert_to_pdf(self, input_path, output_pdf):
        print(f"Converting {input_path}...")
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".pdf":
            return input_path
        if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
            img = Image.open(input_path)
            img = img.convert("RGB")
            img.save(output_pdf, "PDF")
            return output_pdf
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pdf",
                input_path,
                "--outdir",
                os.path.dirname(output_pdf),
            ],
            check=True,
        )
        return output_pdf

    def segment_pdf(self, pdf_path):
        try:
            import fitz  # PyMuPDF
        except Exception as exc:
            raise RuntimeError(
                "Failed to import PyMuPDF (fitz). If you installed the "
                "'fitz' package, uninstall it and install 'pymupdf' instead."
            ) from exc
        doc = fitz.open(pdf_path)
        page_images = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_images.append(img)
        return page_images

    def extract_structured_content(self, images):
        raise NotImplementedError("Subclasses must implement extract_structured_content().")

    def store_outputs(
        self,
        outputs,
        output_folder,
        source_file,
        store_mode="json",
        encoder_fn=None,
        vector_db="chroma",
        collection_name="documents",
    ):
        if store_mode == "none":
            return
        records = []
        for page_number, content in enumerate(outputs, start=1):
            records.append(
                {
                    "file_name": source_file,
                    "page_number": page_number,
                    "content": content,
                }
            )
        if store_mode == "json":
            path = os.path.join(output_folder, "vlm_chunks.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")
            return
        if store_mode != "vector":
            raise ValueError(f"Unsupported store_mode: {store_mode}")
        if encoder_fn is None:
            embeddings = _encode_with_mmdocir_colbert([record["content"] for record in records])
        else:
            embeddings = encoder_fn([record["content"] for record in records])
        if len(embeddings) != len(records):
            raise ValueError("encoder_fn returned mismatched embedding count.")
        if vector_db == "chroma":
            try:
                import chromadb
            except Exception as exc:
                raise RuntimeError(
                    "chromadb is required for vector storage. "
                    "Install it with: pip install chromadb"
                ) from exc
            persist_dir = os.path.join(output_folder, "chroma_db")
            client = chromadb.PersistentClient(path=persist_dir)
            collection = client.get_or_create_collection(name=collection_name)
            ids = [
                f"{record['file_name']}::page={record['page_number']}"
                for record in records
            ]
            metadatas = [
                {"file_name": r["file_name"], "page_number": r["page_number"]}
                for r in records
            ]
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=[r["content"] for r in records],
                metadatas=metadatas,
            )
            return
        if vector_db == "qdrant":
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http.models import Distance, VectorParams
            except Exception as exc:
                raise RuntimeError(
                    "qdrant-client is required for vector storage. "
                    "Install it with: pip install qdrant-client"
                ) from exc
            qdrant_path = os.path.join(output_folder, "qdrant_db")
            client = QdrantClient(path=qdrant_path)
            vector_size = len(embeddings[0]) if embeddings else 0
            try:
                client.get_collection(collection_name=collection_name)
            except Exception:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
            points = []
            for idx, record in enumerate(records):
                points.append(
                    {
                        "id": f"{record['file_name']}::page={record['page_number']}",
                        "vector": embeddings[idx],
                        "payload": {
                            "file_name": record["file_name"],
                            "page_number": record["page_number"],
                            "content": record["content"],
                        },
                    }
                )
            client.upsert(collection_name=collection_name, points=points)
            return
        raise ValueError(f"Unsupported vector_db: {vector_db}")


class QwenVLMDataDigester(DataDigester):
    def __init__(self, model_path="Qwen/Qwen2-VL-7B-Instruct", device="auto", local_files_only=False):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.local_files_only = local_files_only

    def extract_structured_content(self, images):
        model, processor, dev = get_vlm(
            self.model_path, self.device, local_files_only=self.local_files_only
        )
        extracted_data = []
        prompt = (
            "Extract the entire page and convert it to a single Markdown document. "
            "Preserve structure, headings, lists, and tables (as Markdown tables). "
            "Do not invent content."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts text from images."},
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]},
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        for img in images:
            inputs = processor(text=[text], images=[img], return_tensors="pt").to(dev)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    num_beams=1,
                )
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if output_text.startswith(prompt):
                output_text = output_text[len(prompt):].strip()
            extracted_data.append(output_text.strip())
        return extracted_data


class OpenaiVLMDataDigester(DataDigester):
    def __init__(self, model_name="gpt-4o-mini"):
        super().__init__()
        self.model_name = model_name

    def extract_structured_content(self, images):
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(
                "openai package is required for OpenaiVLMDataDigester. "
                "Install it with: pip install openai"
            ) from exc
        client = OpenAI()
        extracted_data = []
        prompt = (
            "Extract the entire page and convert it to a single Markdown document. "
            "Preserve structure, headings, lists, and tables (as Markdown tables). "
            "Do not invent content."
        )
        for img in images:
            image_b64 = _pil_to_png_base64(img)
            response = client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_data": image_b64},
                        ],
                    }
                ],
            )
            extracted_data.append(response.output_text.strip())
        return extracted_data


def _pil_to_png_base64(img):
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_with_colbert(texts, model_name="colbert-ir/colbertv2.0", device="cpu"):
    """
    Default embedding function. Requires ColBERT to be installed.
    Provide your own encoder_fn to override.
    """
    try:
        from colbert.infra import ColBERTConfig, Run, RunConfig
        from colbert.modeling.checkpoint import Checkpoint
        from colbert.modeling.tokenization import QueryTokenizer
    except Exception as exc:
        raise RuntimeError(
            "ColBERT is not installed. Install it or provide encoder_fn. "
            "Example: pip install colbert-ai"
        ) from exc
    with Run().context(RunConfig(nranks=1, experiment="colbert-encode")):
        config = ColBERTConfig(checkpoint=model_name, query_maxlen=256, doc_maxlen=512)
        checkpoint = Checkpoint(config, checkpoint=model_name, device=device)
        tokenizer = QueryTokenizer(config)
        embeddings = []
        for text in texts:
            Q = tokenizer.tensorize([text])
            embedding = checkpoint.query(Q)
            embeddings.append(embedding.squeeze(0).mean(dim=0).detach().cpu().tolist())
    return embeddings



def main(
    input_folder,
    output_folder,
    digester,
    store_mode="none",
    encoder_fn=None,
    vector_db="chroma",
    collection_name="documents",
):
    for filename in os.listdir(input_folder):
        input_file = os.path.join(input_folder, filename)
        if os.path.isfile(input_file):
            # 1. Convert & Segment
            temp_pdf = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.pdf")
            pdf_path = digester.convert_to_pdf(input_file, temp_pdf)
            pages = digester.segment_pdf(pdf_path)
            
            # 2. Extract
            results = digester.extract_structured_content(pages)
            
            # Save results
            with open(os.path.join(output_folder, f"{filename}.txt"), "w") as f:
                f.write("\n\n".join(results))
            digester.store_outputs(
                results,
                output_folder,
                filename,
                store_mode=store_mode,
                encoder_fn=encoder_fn,
                vector_db=vector_db,
                collection_name=collection_name,
            )
            print(f"Finished processing {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Digestion")
    parser.add_argument("--input_folder", type=str, default="./input_data", help="Input folder")
    parser.add_argument("--output_folder", type=str, default="./output_data", help="Output folder")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="VLM model name")
    parser.add_argument("--device", type=str, default="mps", help="Device: auto|cuda|mps|cpu")
    parser.add_argument("--local_files_only", action="store_true", help="Use local HF cache only")
    parser.add_argument("--store_mode", type=str, default="none", help="none|json|vector")
    parser.add_argument("--vector_db", type=str, default="chroma", help="chroma|qdrant")
    parser.add_argument("--collection", type=str, default="documents", help="Vector DB collection name")
    parser.add_argument("input_pos", nargs="?", help="Positional input folder")
    parser.add_argument("output_pos", nargs="?", help="Positional output folder")
    args = parser.parse_args()
    input_folder = args.input_pos or args.input_folder
    output_folder = args.output_pos or args.output_folder
    digester = QwenVLMDataDigester(
        model_path=args.model_name,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    os.makedirs(output_folder, exist_ok=True)
    main(
        input_folder,
        output_folder,
        digester,
        store_mode=args.store_mode,
        encoder_fn=None,
        vector_db=args.vector_db,
        collection_name=args.collection,
    )
