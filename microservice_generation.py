explain below code """
Auto-generated microservice generator with automatic dependency detection
and LLM-driven interconnected microservice code generation.

Assumptions:
- You already have Qwen model and tokenizer initialized (local).
- Java monolith is present under `repo_path`.
- This script will create output/<microservice_name>/ with generated files.

Set environment variables as required (e.g. GITLAB_API_TOKEN) before running.
"""

import os
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

# -------------------------
# Your existing model setup
# (reuse your existing InitializeModel() + tokenizer)
# -------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss

# ---------- CONFIG ----------
repo_path = "./playing-with-java-microservices-monolith-example/src/main/java"
OUTPUT_DIR = "output"
model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
embedding_model_name = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
# LLM inference device - use "cuda" if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Input mapping of microservices (example: replace or reuse your input_json)
input_json = [
    {
        "name": "Catalog Service",
        "description": "Handles product-related operations.",
        "classes": [
            "com.targa.labs.myboutique.web.ProductResource",
            "com.targa.labs.myboutique.service.ProductService",
            "com.targa.labs.myboutique.repository.ProductRepository",
            "com.targa.labs.myboutique.web.dto.ProductDto"
        ]
    },
    {
        "name": "Customer Service",
        "description": "Handles customer-related operations.",
        "classes": [
            "com.targa.labs.myboutique.web.CustomerResource",
            "com.targa.labs.myboutique.service.CustomerService",
            "com.targa.labs.myboutique.repository.CustomerRepository",
            "com.targa.labs.myboutique.web.dto.CustomerDto"
        ]
    },
    {
      "name": "Common Utilities",
      "description": "Contains utility classes and constants used across multiple services.",
      "classes": [
        "com.targa.labs.myboutique.common.Web",
        "com.targa.labs.myboutique.domain.enumeration.CartStatus",
        "com.targa.labs.myboutique.domain.enumeration.OrderStatus",
        "com.targa.labs.myboutique.domain.enumeration.ProductStatus",
        "com.targa.labs.myboutique.domain.enumeration.PaymentStatus"
      ]
    }
]

# -------------------------
# Helper utilities
# -------------------------
def InitializeModel():
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        quantization_config=quant_config
    )
    return model

print("Loading models (this may take a while)...")
model = InitializeModel()
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = SentenceTransformer(embedding_model_name)
dim = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dim)
vector_db = []

# ---------- IO helpers ----------
def read_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# -------------------------
# Step 1: gather all java files and map class names -> file paths
# -------------------------
def collect_java_files(root: str) -> List[Path]:
    p = Path(root)
    return [f for f in p.rglob("*.java")]

all_java_files = collect_java_files(repo_path)
print(f"Found {len(all_java_files)} Java files.")

# Build a lookup: fully qualified class name -> file path
# We'll attempt to parse package + class name from file content
def extract_fqcn_from_file(path: Path) -> str:
    text = read_file(path)
    pkg_match = re.search(r'^\s*package\s+([a-zA-Z0-9_.]+)\s*;', text, re.MULTILINE)
    cls_match = re.search(r'public\s+(?:class|interface|enum)\s+([A-Za-z0-9_]+)', text)
    if pkg_match and cls_match:
        return f"{pkg_match.group(1)}.{cls_match.group(1)}"
    # fallback: try filename-based mapping
    return ".".join(path.with_suffix("").parts[-4:])  # rough fallback

fqcn_to_path: Dict[str, Path] = {}
for f in all_java_files:
    try:
        fqcn = extract_fqcn_from_file(f)
        fqcn_to_path[fqcn] = f
    except Exception:
        continue

print(f"Extracted {len(fqcn_to_path)} fully-qualified class names (approx).")

# -------------------------
# Step 2: map class -> microservice (based on provided input_json)
# -------------------------
# Build map microservice_name -> set of FQCNs
ms_to_classes: Dict[str, Set[str]] = {}
for ms in input_json:
    ms_name = ms["name"]
    ms_to_classes[ms_name] = set(ms.get("classes", []))

# If a class in input_json is not present in the repo, we still include it
# Build reverse map FQCN -> microservice
class_to_ms: Dict[str, str] = {}
for ms_name, classes in ms_to_classes.items():
    for c in classes:
        class_to_ms[c] = ms_name

# Also attempt to detect FQCNs appearing in repo that are not in input_json:
# map them to 'unknown' microservice (optionally assign later)
for fqcn in fqcn_to_path.keys():
    if fqcn not in class_to_ms:
        class_to_ms[fqcn] = "UNASSIGNED"

# -------------------------
# Step 3: dependency detection
#  - For each file, search for mentions of other FQCN class simple names or imports
#  - Record edges: msA -> msB if file belonging to msA references a class that belongs to msB
# -------------------------
def short_name(fqcn: str) -> str:
    return fqcn.split(".")[-1]

# Build file -> owning microservice (based on most likely class inside it)
file_to_ms: Dict[Path, str] = {}
for fqcn, path in fqcn_to_path.items():
    ms = class_to_ms.get(fqcn, "UNASSIGNED")
    file_to_ms[path] = ms

# Precompute mapping from simple class name -> fqcn list (many-to-one possible)
simple_to_fqcns: Dict[str, List[str]] = {}
for fqcn in fqcn_to_path.keys():
    s = short_name(fqcn)
    simple_to_fqcns.setdefault(s, []).append(fqcn)

# Detect dependencies
def detect_dependencies() -> Dict[str, Set[str]]:
    deps: Dict[str, Set[str]] = {ms: set() for ms in ms_to_classes.keys()}
    for path in all_java_files:
        owner_ms = file_to_ms.get(path, "UNASSIGNED")
        content = read_file(path)
        # Check import lines for cross-ms references
        imports = re.findall(r'^\s*import\s+([a-zA-Z0-9_.]+)\s*;', content, re.MULTILINE)
        for imp in imports:
            if imp in class_to_ms:
                target_ms = class_to_ms[imp]
                if target_ms != owner_ms and owner_ms in deps:
                    deps[owner_ms].add(target_ms)
        # Check usages by simple name
        for simple, fqcn_list in simple_to_fqcns.items():
            # skip if same file defines the class
            if re.search(r'\b' + re.escape(simple) + r'\b', content):
                for fqcn in fqcn_list:
                    target_ms = class_to_ms.get(fqcn, "UNASSIGNED")
                    if target_ms != owner_ms and owner_ms in deps:
                        deps[owner_ms].add(target_ms)
        # Heuristic: look for @Autowired or field declarations referencing services
        for m in re.findall(r'@Autowired\s+private\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+);', content):
            cls_name = m[0]
            if cls_name in simple_to_fqcns:
                for fqcn in simple_to_fqcns[cls_name]:
                    tms = class_to_ms.get(fqcn)
                    if tms and tms != owner_ms and owner_ms in deps:
                        deps[owner_ms].add(tms)
    return deps

dependency_graph = detect_dependencies()

print("Detected dependency graph:")
print(json.dumps({k: list(v) for k,v in dependency_graph.items()}, indent=2))

# Save the graph for inspection
Path(OUTPUT_DIR).mkdir(exist_ok=True)
write_file(Path(OUTPUT_DIR)/"interconnections.json", json.dumps({k:list(v) for k,v in dependency_graph.items()}, indent=2))

# -------------------------
# Step 4: create enhanced prompts and call LLM to generate interconnected microservices
# -------------------------
# Utility to run model generation (wrap your earlier generation code)
def generate_with_qwen(prompt: str, max_new_tokens: int = 4000) -> str:
    # Inspired by your earlier usage; using tokenizer/model directly on device
    # For large outputs, you may want to stream or chunk calls
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(DEVICE)
    # generate
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # as in your earlier approach, trim prompt tokens from outputs
    out_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # sometimes decoder returns full text including prompt - try to remove prompt prefix
    if out_text.startswith(prompt):
        out_text = out_text[len(prompt):].strip()
    return out_text

# Compose enhanced prompt for each microservice
def compose_prompt(microservice_name: str, class_files: List[str], deps_graph: Dict[str, Set[str]], common_dt os=None) -> str:  # type: ignore (common_dt placeholder)
    # classes content concatenation for context (limit size)
    context_snippets = []
    for path in class_files:
        content = read_file(Path(path))
        # take up to N chars per class to keep prompt manageable
        context_snippets.append(f"/* file: {Path(path).name} */\n" + content[:3000])
    context_text = "\n\n".join(context_snippets)
    deps_for_ms = deps_graph.get(microservice_name, set())
    deps_json = json.dumps({k: list(v) for k, v in deps_graph.items()}, indent=2)
    # instruction
    prompt = f"""
You are an expert Java Spring Boot engineer and code generator. You will generate a complete microservice project for "{microservice_name}".
Context: You were given the extracted monolithic source files (snippets below) which belong to this microservice. Recapture the logic, data access, controllers, DTOs, and service layer.

CRITICALLY: The system also includes other microservices and their dependency graph is:
{deps_json}

For this microservice ("{microservice_name}"):
1) Generate code files for controllers, services, repositories, DTOs and entities needed to preserve the original logic.
2) For each dependency of this microservice (i.e. dependencies -> {list(deps_for_ms)}), generate:
   a) a typed client interface (Feign or WebClient) to call the remote microservice. Include Java code for the client interface and an example usage in the service layer.
   b) if the interaction should be event-driven, also generate event classes and a publisher (Kafka) and describe the topic name and schema.
3) Generate a common module (if not present) name "common-dto" that contains shared DTOs/enums referenced by multiple services; if a DTO is unique keep it inside this microservice.
4) Provide a pom.xml for this microservice listing dependencies for Spring Web, Spring Data JPA, Spring Boot Starter, Feign (or WebClient), Kafka (if events used), and the common-dto module as dependency.
5) Add application.yml with placeholders for service discovery (Eureka/Consul) and show sample API Gateway route config for this microservice (path & upstream).
6) Add SQL patch if this microservice needs schema changes (create table statements).
7) Provide a README describing how to build, run, and how the service communicates with others (sync endpoints and topics).

Input (snippets from monolith for context):
{context_text}

Important constraints:
- Keep each generated file complete and compilable (imports + package statements).
- For clients, use REST paths that logically match typical APIs (e.g. /api/products/{id}).
- For shared DTOs, create package com.example.common.dto (or follow original package if available).
- If uncertain about exact types, choose reasonable defaults and document assumptions in README.

Output format:
Return a JSON object with keys:
{{
  "files": [{{"path":"relative/path/to/File.java","content":"..."} , ...]],
  "pom": "pom.xml content",
  "README": "README text",
  "notes": "brief notes about any assumptions"
}}

Do not output anything else (the caller will parse JSON). Now generate.
"""
    return prompt

# -------------------------
# Run generation for each microservice
# -------------------------
def generate_microservice(ms_name: str, ms_classes: List[str], deps_graph: Dict[str, Set[str]]):
    # map class names to file paths (if available)
    class_paths = []
    for cls in ms_classes:
        p = fqcn_to_path.get(cls)
        if p:
            class_paths.append(str(p))
    prompt = compose_prompt(ms_name, class_paths, deps_graph)
    print(f"[LLM] Generating microservice: {ms_name} (prompt length {len(prompt)} chars)")
    generated = generate_with_qwen(prompt, max_new_tokens=6000)
    # Try parse JSON from LLM (robust)
    try:
        # LLM sometimes outputs text before JSON; find first '{' and last '}'.
        first = generated.find("{")
        last = generated.rfind("}")
        json_text = generated[first:last+1]
        parsed = json.loads(json_text)
    except Exception as e:
        print("[ERROR] Failed to parse JSON from LLM output:", e)
        # save raw output for inspection
        write_file(Path(OUTPUT_DIR)/ms_name/"raw_llm_output.txt", generated)
        return False
    # persist files
    ms_out_dir = Path(OUTPUT_DIR)/ms_name.replace(" ", "_")
    ms_out_dir.mkdir(parents=True, exist_ok=True)
    for f in parsed.get("files", []):
        path = ms_out_dir / f["path"]
        write_file(path, f["content"])
    # pom + README
    if "pom" in parsed:
        write_file(ms_out_dir/"pom.xml", parsed["pom"])
    if "README" in parsed:
        write_file(ms_out_dir/"README.md", parsed["README"])
    # notes
    write_file(ms_out_dir/"_notes.txt", parsed.get("notes", ""))
    print(f"[OK] Wrote microservice files to {ms_out_dir}")
    return True

# -------------------------
# Orchestrate generation
# -------------------------
def run_generation_all():
    for ms in input_json:
        ms_name = ms["name"]
        ms_classes = ms.get("classes", [])
        success = generate_microservice(ms_name, ms_classes, dependency_graph)
        if not success:
            print(f"[WARN] generation failed for {ms_name}. See raw_llm_output.txt in output folder")

# -------------------------
# Save dependency graph and run
# -------------------------
if __name__ == "__main__":
    # save dependency graph
    write_file(Path(OUTPUT_DIR)/"dependency_graph.json", json.dumps({k:list(v) for k,v in dependency_graph.items()}, indent=2))
    run_generation_all()
    print("Done. Check the output/ directory for generated microservices and interconnections.")
