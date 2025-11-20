import os
import re
import json
import shutil
import subprocess
import zipfile
import asyncio
from io import BytesIO
from datetime import datetime
from typing import Optional, List

import httpx
import asyncpg
from fastapi import HTTPException, Depends, APIRouter, UploadFile, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langgraph.graph import StateGraph
from embedding import Embedding
from postgres import get_postgres
from qwencoderllm import analyse_file, identify_microservices

# ==========================================================
# === Global Config ========================================
# ==========================================================

router = APIRouter()
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://10.0.131.69:8011/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-instruct")
JAVA_API_URL = os.environ.get("JAVA_API_URL", "http://localhost:8080/generate/project")

upload_dir = os.path.join(os.getcwd(), "monolith_apps")
microservice_dir = os.path.join(os.getcwd(), "generated_microservices")
os.makedirs(upload_dir, exist_ok=True)
os.makedirs(microservice_dir, exist_ok=True)
embedding = Embedding()

# ==========================================================
# === LangGraph State Schema ===============================
# ==========================================================

class GraphState(BaseModel):
    name: str
    language: Optional[str] = None
    git_url: Optional[str] = None
    zip_file: Optional[object] = None
    source_path: Optional[str] = None
    embedding_done: Optional[bool] = None
    microservice_json: Optional[str] = None
    output_zip: Optional[str] = None


# ==========================================================
# === DB Helper ============================================
# ==========================================================

async def save_microservice_suggestion_db(
    monolith_name: str,
    microservice_suggestion: str,
    db_pool: asyncpg.Pool,
):
    query = """
        UPDATE monotomicro
        SET microservice_suggestion = $1,
            date_time = $2,
            status = $3,
            action = $4
        WHERE monolith_name = $5
    """
    async with db_pool.acquire() as conn:
        await conn.execute(
            query,
            microservice_suggestion,
            datetime.now(),
            "JSON saved",
            "Create microservices",
            monolith_name,
        )


# ==========================================================
# === LangGraph Agent Nodes ================================
# ==========================================================

async def add_monolith(state: GraphState) -> GraphState:
    db_pool = await get_postgres()
    name, lang = state.name, state.language
    git_url, zip_file = state.git_url, state.zip_file

    monolith_path = os.path.join(upload_dir, name)
    if os.path.exists(monolith_path):
        shutil.rmtree(monolith_path)

    # --- Clone from Git or extract from zip ---
    if git_url:
        subprocess.run(["git", "clone", git_url, monolith_path], check=True)
    elif zip_file:
        zip_path = os.path.join(upload_dir, zip_file.filename)
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(zip_file.file, f)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(monolith_path)
    else:
        raise HTTPException(status_code=400, detail="No source provided (git_url or zip_file).")

    # --- Insert database record ---
    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO monotomicro 
               (monolith_name, monolith_url, status, date_time, language, action)
               VALUES ($1, $2, $3, $4, $5, $6)""",
            name,
            git_url or "uploaded_zip",
            "Monolith uploaded",
            datetime.now(),
            lang,
            "Creating embeddings",
        )

    state.source_path = monolith_path
    return state


async def embed_code(state: GraphState) -> GraphState:
    """Generate embeddings for the uploaded monolith codebase."""
    await embedding.add_dotnet_codebase_embeddings(state.name, state.source_path, state.language)
    state.embedding_done = True
    return state


async def generate_microservice_suggestion(state: GraphState) -> GraphState:
    """Use Qwen model to suggest microservice boundaries."""
    db_pool = await get_postgres()
    summaries = []

    for root, _, files in os.walk(state.source_path):
        for file in files:
            if file.endswith("." + state.language):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                    context = embedding.GetRelevantContext(code)
                    result = await analyse_file(context, code)
                    await insert_summary(state.name, result)
                    summaries.append(result)

    combined_summary = "\n".join(summaries)
    microservice_json = await identify_microservices(combined_summary)

    async with db_pool.acquire() as conn:
        await conn.execute(
            """UPDATE monotomicro 
               SET microservice_suggestion=$1, status=$2, action=$3 
               WHERE monolith_name=$4""",
            microservice_json,
            "Microservice suggestions ready",
            "Preview JSON",
            state.name,
        )

    state.microservice_json = microservice_json
    return state

async def insert_summary(
    
    monolith_name: str,
    monolith_analysis: str
) -> str:
    query = """
        INSERT INTO monotomicro_summary (monolith_name, monolith_analysis)
        VALUES ($1, $2)
    """
    db_pool = await get_postgres()
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchrow(query, monolith_name, monolith_analysis)
            return "monolith_analysis updated!"
    except Exception as e:
        print(f"Error updating monolith_analysis: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during updating monolith_analysis"
        )

# ==========================================================
# === Microservice Code Generation Agent ===================
# ==========================================================

async def generate_microservice_code(state: GraphState) -> GraphState:
    """Generate microservice Java code using Qwen + Java API."""
    db_pool = await get_postgres()
    await save_microservice_suggestion_db(state.name, state.microservice_json, db_pool)

    microservice_data = json.loads(state.microservice_json)
    zip_response = await analyze_microservices(MicroserviceRequest(**microservice_data))
    state.output_zip = f"{microservice_dir}/{state.name}_microservices.zip"

    async with db_pool.acquire() as conn:
        await conn.execute(
            """UPDATE monotomicro SET status=$1, action=$2 WHERE monolith_name=$3""",
            "Generated code",
            "View code",
            state.name,
        )

    return state

import asyncio
import json
import re
import httpx
from io import BytesIO
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from typing import List
from datetime import datetime

# === Configuration (reused) ===
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://10.0.131.69:8010/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-instruct")
JAVA_API_URL = os.environ.get("JAVA_API_URL", "http://localhost:8080/generate/project")
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries


# ==========================================================
# === Helper Models ========================================
# ==========================================================

class DependencyLinks(BaseModel):
    internal: List[str] = []
    external: List[str] = []


class MicroserviceItem(BaseModel):
    name: str
    description: str
    dependencies: DependencyLinks
    classes: List[str] = []


class Interconnection(BaseModel):
    source: str
    target: str
    description: str


class MicroserviceRequest(BaseModel):
    microservices: List[MicroserviceItem]
    interconnections: Optional[List[Interconnection]] = None


# ==========================================================
# === Helper Function: Safe HTTP call to Qwen ===============
# ==========================================================

async def call_qwen_instruct_async(prompt: str, retry_count=MAX_RETRIES) -> str:
    """
    Sends a prompt to Qwen instruct API with retry & logging.
    """
    for attempt in range(1, retry_count + 1):
        try:
            async with httpx.AsyncClient(base_url=QWEN_BASE_URL, timeout=7200.0) as client:
                payload = {
                    "model": QWEN_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are an assistant that generates Java microservices JSON specs."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 4096,
                    "temperature": 0,
                    "top_p": 1,
                }
                response = await client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                result = data["choices"][0]["message"]["content"]
                print(f"[✅ QWEN SUCCESS] Attempt {attempt}/{retry_count}")
                return result
        except Exception as e:
            print(f"[⚠️ QWEN RETRY {attempt}/{retry_count}] Error: {e}")
            if attempt == retry_count:
                raise HTTPException(status_code=500, detail=f"QWEN API failed after {retry_count} retries.")
            await asyncio.sleep(RETRY_DELAY)


# ==========================================================
# === Core Function: Analyze Microservices =================
# ==========================================================

async def analyze_microservices(request_data: MicroserviceRequest):
    """
    1. Calls Qwen per microservice to generate detailed JSON
    2. Wraps them into a project-level JSON
    3. Calls Java API to build and return ZIP
    """

    # ---- Inner function: generate one microservice detail ----
    async def generate_detail(micro: MicroserviceItem, idx: int, total: int):
        print(f"[🧩 Step {idx}/{total}] Generating details for: {micro.name}")
        detail_prompt = f"""You are an assistant that generates only the microservice details (no project/package info).

From this info:
Name: {micro.name}
Description: {micro.description}
Dependencies (internal): {', '.join(micro.dependencies.internal)}
Dependencies (external): {', '.join(micro.dependencies.external)}
Classes: {chr(10).join(micro.classes)}

Fill ONLY this JSON template: fi

{{
  "microserviceName": "string",
  "groupId": "string",
  "artifactId": "string",
  "description": "string",
  "controllerModel": {{
    "controllerClassName": "string",
    "restControllerEndpoint": "string",
    "controllerMethods": [
      {{
        "methodName": "string",
        "returnType": "string",
        "methodArguments": [
          {{
            "argumentName": "string",
            "dataType": "string",
            "parameterAnnotation": "string"
          }}
        ],
        "httpMethod": "string",
        "endPoint": "string",
        "description": "string"
      }}
    ]
  }},
  "serviceClassName": "string",
  "entity": {{
    "entityName": "string",
    "entityFields": [
      {{
        "fieldName": "string",
        "dataType": "string"
      }}
    ]
  }},
  "portNumber": "string",
  "dataSourceDetails": {{
    "portNumber": "string",
    "databaseName": "string",
    "userName": "string",
    "password": "string"
  }},
  "dependentOn": "string"
}}

Your response must be ONLY the above JSON object, parseable by Python's json.loads().
"""
        llm_output = await call_qwen_instruct_async(detail_prompt)
        match = re.search(r"\{.*\}", llm_output, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                print(f"[✅ SUCCESS] {micro.name}")
                return parsed
            except Exception:
                print(f"[⚠️ JSON Parse Error] {micro.name}")
                return {"name": micro.name, "raw_output": llm_output}
        print(f"[❌ INVALID OUTPUT] {micro.name}")
        return {"name": micro.name, "raw_output": llm_output}

    # ---- Generate details concurrently ----
    total = len(request_data.microservices)
    print(f"[Starting Qwen Generation for {total} microservices]")
    tasks = [generate_detail(m, i + 1, total) for i, m in enumerate(request_data.microservices)]
    microservice_details_list = await asyncio.gather(*tasks)
    print(f"[All {total} microservices processed]")

    # ---- Combine details into project JSON ----
    wrapper_prompt = f"""You are an assistant that generates the top-level project JSON for Java microservices.

Here are all the microservice details:
{json.dumps(microservice_details_list, indent=2)}

Fill and return this JSON template with values inferred from the microservices:

{{
  "projectName": "string",
  "groupId": "string",
  "artifactId": "string",
  "microserviceDetailsList": [ ... ]
}}

Where microserviceDetailsList is exactly as shown above, with no change.
Your response must be ONLY the JSON object, parseable by Python's json.loads().
"""
    print("[ Building Project JSON via Qwen]")
    llm_output = await call_qwen_instruct_async(wrapper_prompt)
    match = re.search(r"\{.*\}", llm_output, re.DOTALL)
    project_payload = None
    if match:
        try:
            project_payload = json.loads(match.group(0))
            print("[Project JSON parsed successfully]")
        except Exception:
            project_payload = None

    if not project_payload or "projectName" not in project_payload or "microserviceDetailsList" not in project_payload:
        print("[Invalid project JSON from LLM]", llm_output)
        raise HTTPException(status_code=500, detail="LLM did not return a valid project JSON.")

    # ---- Call Java microservice generator ----
    async with httpx.AsyncClient(timeout=7200.0) as client:
        try:
            print("[⚙️ Sending to Java API...]")
            java_response = await client.post(JAVA_API_URL, json=project_payload)
            java_response.raise_for_status()
            zip_bytes = java_response.content
            print("[✅ Java API ZIP received]")
        except httpx.HTTPStatusError as exc:
            detail_msg = f"Java API returned {exc.response.status_code}: {exc.response.text[:300]}"
            raise HTTPException(status_code=exc.response.status_code, detail=detail_msg)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f"Java API request failed: {exc}")

    # ---- Return ZIP stream ----
    return StreamingResponse(
        BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=microservices.zip"}
    )

# ==========================================================
# === LangGraph Definition =================================
# ==========================================================

ingestion_graph = StateGraph(GraphState)
ingestion_graph.add_node("AddMonolithAgent", add_monolith)
ingestion_graph.add_node("EmbedCodeAgent", embed_code)
ingestion_graph.add_node("GenerateMicroserviceSuggestionAgent", generate_microservice_suggestion)
ingestion_graph.add_edge("AddMonolithAgent", "EmbedCodeAgent")
ingestion_graph.add_edge("EmbedCodeAgent", "GenerateMicroserviceSuggestionAgent")
ingestion_graph.set_entry_point("AddMonolithAgent")
ingestion_graph.set_finish_point("GenerateMicroserviceSuggestionAgent")

generation_graph = StateGraph(GraphState)
generation_graph.add_node("GenerateMicroserviceCodeAgent", generate_microservice_code)
generation_graph.set_entry_point("GenerateMicroserviceCodeAgent")
generation_graph.set_finish_point("GenerateMicroserviceCodeAgent")


# ==========================================================
# === Graph Runner Functions ===============================
# ==========================================================

async def run_ingestion_graph(name, language, git_url="", zip_file=None):
    state = GraphState(name=name, language=language, git_url=git_url, zip_file=zip_file)
    
    graph_app = ingestion_graph.compile()   
    await graph_app.ainvoke(state)          

    return state


async def run_generation_graph(name, microservice_json):
    state = GraphState(name=name, microservice_json=microservice_json)

    graph_app = generation_graph.compile()  
    await graph_app.ainvoke(state)          

    return state



# ==========================================================
# === FastAPI Routes =======================================
# ==========================================================

from fastapi import HTTPException, UploadFile, File, Form
from typing import Optional
from postgres import get_postgres

@router.post("/add_monolith/")
async def add_monolith_route(
    name: str = Form(...),
    language: str = Form(...),
    git_url: Optional[str] = Form(""),
    zip_file: Optional[UploadFile] = File(None),
):
    """
    Step 1: Adds monolith → creates embeddings → generates suggestions automatically.
    Handles:
      - `git_url` for cloning repo
      - `zip_file` if uploading source code ZIP
      - gracefully handles missing/empty zip_file
    """
    # Check for duplicate name
    db_pool = await get_postgres()
    name_exist = await check_if_name_already_added(name, db_pool)
    if name_exist:
        raise HTTPException(
            status_code=400, detail="Given name already exists in the database."
        )

    # Handle empty or missing file gracefully
    if zip_file and zip_file.filename.strip() == "":
        zip_file = None

    # Must have at least one source input
    if not git_url and not zip_file:
        raise HTTPException(
            status_code=400,
            detail="Either 'git_url' or 'zip_file' must be provided."
        )

    ctx = await run_ingestion_graph(name, language, git_url, zip_file)

    return {
        "message": "Monolithic application uploaded successfully."
    }




async def check_if_name_already_added(name: str, db_pool):
    query = "select monolith_name from monotomicro"
    async with db_pool.acquire() as conn:
        result = await conn.fetch(query)
        names = [record[0] for record in result]
        print (names)
        return name in names
 
@router.get("/get_monolith_repos/")
async def get_monolith_repos():
    """
    Fetch the list of monolithic application names.
    """
    db_pool = await get_postgres()

    query = "SELECT monolith_name FROM monotomicro ORDER BY date_time DESC"

    async with db_pool.acquire() as conn:
        result = await conn.fetch(query)

        if not result:
            raise HTTPException(status_code=404, detail="No monolithic repos found.")

        # Extract values
        names = [row["monolith_name"] for row in result]

        return {
            "repositories": names,
            "count": len(names),
            "message": "Monolithic repos retrieved successfully."
        }

@router.get("/get_monotomicro_status/")
async def get_monotomicro_status(name: str):
    """
    Fetch full monolith ingestion status including:
    monolith name, url, created date, status,
    uploaded zip file name and path.
    """
    db_pool = await get_postgres()

    query = """
        SELECT 
            monolith_name, 
            monolith_url, 
            date_time, 
            status,
            zip_file_name,
            zip_file_path
        FROM monotomicro
        WHERE monolith_name = $1
    """

    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(query, name)

        if not result:
            raise HTTPException(status_code=404, detail="Monolith name not found.")

        return {
            "monolith_name": result["monolith_name"],
            "monolith_url": result["monolith_url"],
            "created_date": result["date_time"],
            "status": result["status"],
            "zip_file_name": result["zip_file_name"],
            "zip_file_path": result["zip_file_path"],
            "message": "Monolith ingestion status fetched successfully."
        }


@router.get("/get_microservice_suggestion/")
async def get_microservice_suggestion_route(name: str):
    """
    Fetches the microservice suggestion JSON for a given monolith name.
    """
    db_pool = await get_postgres()
    query = "SELECT microservice_suggestion FROM monotomicro WHERE monolith_name = $1"
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(query, name)
        if result and result["microservice_suggestion"]:
            return {
                "suggestion": result["microservice_suggestion"],
                "message": "Microservice suggestion retrieved successfully."
            }
        else:
            raise HTTPException(status_code=404, detail="Microservice suggestion not found.")


@router.post("/save_microservice_suggestion/")
async def save_microservice_suggestion_route(name: str, microservice_suggestion: str):
    """
    Step 2: Generates microservice code from the suggestion JSON.
    """
    ctx = await run_generation_graph(name, microservice_suggestion)
    return {
        "message": "Microservices saved successfully."
    }

@router.get("/download_microservices/")
async def download_microservices_route(name: str):
    """
    Downloads the generated microservices ZIP for a given monolith name.
    """
    db_pool = await get_postgres()
    query = "SELECT output_zip FROM monotomicro WHERE monolith_name = $1"
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(query, name)
        if result and result["output_zip"]:
            zip_path = result["output_zip"]
            if os.path.exists(zip_path):
                return StreamingResponse(
                    open(zip_path, "rb"),
                    media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename={name}_microservices.zip"},
                    zip_path=zip_path
                )
            else:
                raise HTTPException(status_code=404, detail="Generated ZIP file not found.")
        else:
            raise HTTPException(status_code=404, detail="Microservices not generated yet.")
        
@router.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model": QWEN_MODEL, "qwen_base_url": QWEN_BASE_URL}
