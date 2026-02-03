import os
import sys
import re
import json
import shutil
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# Add current directory to Python path BEFORE importing local modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

import pandas as pd

# Make sure local modules can be imported


from graphExperiment import course_graph, get_course_prerequisites, get_major_course_codes


# =========================
# Globals
# =========================

pdf_retriever = None
course_retriever = None  # For individual course queries (processed_psu_courses.csv)
majors_retriever = None  # For major requirements queries (all_abington_majors_combined(5).csv)
policy_retriever = None  # For advising/registrar policy PDF queries
policy_vectorstore = None
llm1 = None
llm2 = None
has_transcript = False
stored_pdf_docs = None  # Store original PDF docs for deterministic extraction
latest_student_data = None  # Cache most recent extracted student data

# Absolute paths for Chroma persistence
LLM_DIR = os.path.abspath(os.path.dirname(__file__))
CHROMA_DATA_DIR = os.path.join(LLM_DIR, "chroma_data")
COLLECTION_METADATA_FILE = os.path.join(CHROMA_DATA_DIR, "collection_metadata.json")
POLICY_PDF_PATH = os.path.join(LLM_DIR, "PennState_Combined_Advising_and_Registrar_VERBATIM.pdf")

# =========================
# Helper Functions for Document Processing
# =========================

def get_document_hash(file_path: str) -> str:
    """Generate SHA256 hash of PDF file."""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_transcript_collection_name(user_id: str, doc_hash: str) -> str:
    """Generate collection name: transcript_userid_hash8"""
    return f"transcript_{user_id}_{doc_hash[:8]}"

def chunk_with_metadata(pdf_docs: List[Document]) -> List[Document]:
    """Chunk PDF with page and section metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=130
    )
    chunks = text_splitter.split_documents(pdf_docs)
    
    # Add metadata
    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 0)
        content = chunk.page_content
        
        # Detect section
        section = "general"
        if "Not Satisfied" in content or "Unmet" in content:
            section = "unmet_requirements"
        elif "In Progress" in content:
            section = "in_progress"
        elif "Course History" in content or re.search(r'\b[A-Z]{2,}\s+\d{3}\b', content):
            section = "course_history"
        elif "Major" in content and "Option" in content:
            section = "major_info"
        
        chunk.metadata.update({
            "page": page,
            "section": section,
            "chunk_id": f"page_{page}_chunk_{i}"
        })
    
    return chunks


def is_policy_question(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    policy_terms = [
        "advisor", "adviser", "advising", "advising center",
        "drop", "add", "late drop", "withdraw", "audit",
        "leave of absence", "declare major", "declare minor",
        "double major", "concurrent major", "change my major",
        "what-if report", "lionpath", "academic requirements",
        "entry to major", "early progress report", "internship",
        "aleks", "world campus", "registration", "bursar",
        "tuition penalty", "credits", "f grade", "gpa"
    ]
    return any(term in t for term in policy_terms)

def track_collection_creation(collection_name: str, user_id: str):
    """Track when collection was created for TTL cleanup."""
    metadata = {}
    if os.path.exists(COLLECTION_METADATA_FILE):
        try:
            with open(COLLECTION_METADATA_FILE, 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {}
    
    metadata[collection_name] = {
        "user_id": user_id,
        "created_at": datetime.now().isoformat()
    }
    
    with open(COLLECTION_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def cleanup_old_collections(ttl_days=7):
    """Remove transcript collections older than TTL."""
    if not os.path.exists(COLLECTION_METADATA_FILE):
        return
    
    try:
        with open(COLLECTION_METADATA_FILE, 'r') as f:
            metadata = json.load(f)
    except:
        return
    
    cutoff = datetime.now() - timedelta(days=ttl_days)
    to_remove = []
    
    for collection_name, info in metadata.items():
        try:
            created_at = datetime.fromisoformat(info['created_at'])
            if created_at < cutoff:
                to_remove.append(collection_name)
                # Delete collection directory
                collection_path = os.path.join(CHROMA_DATA_DIR, "transcripts", collection_name)
                if os.path.exists(collection_path):
                    shutil.rmtree(collection_path)
        except:
            continue
    
    # Update metadata
    for name in to_remove:
        del metadata[name]
    
    with open(COLLECTION_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if to_remove:
        print(f"🧹 Cleaned up {len(to_remove)} old collections")


class CachedEmbeddings:
    """Wrapper around OpenAIEmbeddings that caches results."""
    
    def __init__(self, embedding_fn, cache_dir=None):
        self.embedding_fn = embedding_fn
        if cache_dir is None:
            cache_dir = os.path.join(CHROMA_DATA_DIR, "embedding_cache")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{text_hash}.pkl")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with caching."""
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            cache_path = self._get_cache_path(text)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        embeddings.append((i, pickle.load(f)))
                except:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            new_embeddings = self.embedding_fn.embed_documents(texts_to_embed)
            for idx, text, emb in zip(indices_to_embed, texts_to_embed, new_embeddings):
                cache_path = self._get_cache_path(text)
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(emb, f)
                except:
                    pass
                embeddings.append((idx, emb))
        
        # Return in original order
        return [emb for _, emb in sorted(embeddings)]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with caching."""
        cache_path = self._get_cache_path(text)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        embedding = self.embedding_fn.embed_query(text)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except:
            pass
        return embedding

# =========================
# Environment / Setup
# =========================

def setup_environment(whatIfUpload=None, user_id='default'):
    """
    Initialize environment, LLMs, vectorstores for PDF + courses.
    If whatIfUpload is None or has no 'transcript', run in no-transcript mode.
    """
    global pdf_retriever, course_retriever, majors_retriever, llm1, llm2, has_transcript, stored_pdf_docs, latest_student_data

    has_transcript = False  # reset flag each call
    
    # Ensure Chroma data directory exists
    os.makedirs(CHROMA_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(CHROMA_DATA_DIR, "courses"), exist_ok=True)
    os.makedirs(os.path.join(CHROMA_DATA_DIR, "transcripts"), exist_ok=True)

    #  Environment Variables
    LANGCHAIN_PROJECT = "pr-notable-garbage-99"
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = 'REDACTED'
    os.environ['OPENAI_API_KEY'] = "REDACTED"

    if not os.getenv('LANGCHAIN_API_KEY'):
        print("⚠️ LANGCHAIN_API_KEY is not set in the environment.")
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️ OPENAI_API_KEY is not set in the environment.")

    # ---- Optional transcript / What-If report ----
    if whatIfUpload is not None and 'transcript' in whatIfUpload.files:
        transcript_file = whatIfUpload.files['transcript']
        filename = secure_filename(transcript_file.filename)
        temp_pdf_path = f"/tmp/{filename}"
        transcript_file.save(temp_pdf_path)

        # Generate document hash
        doc_hash = get_document_hash(temp_pdf_path)
        collection_name = get_transcript_collection_name(user_id, doc_hash)
        collection_path = os.path.join(CHROMA_DATA_DIR, "transcripts", collection_name)
        
        # Check if collection exists (cached)
        if os.path.exists(collection_path):
            try:
                cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                vectorstore_pdf = Chroma(
                    collection_name=collection_name,
                    persist_directory=os.path.join(CHROMA_DATA_DIR, "transcripts"),
                    embedding_function=cached_embeddings
                )
                pdf_retriever = vectorstore_pdf.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.5}
                )
                # For cached transcripts, reload PDF for deterministic extraction
                loader = PyPDFLoader(temp_pdf_path)
                stored_pdf_docs = redact_student_info(loader.load())
                try:
                    latest_student_data = extract_student_data_deterministic(stored_pdf_docs)
                except Exception as e:
                    print(f"⚠️ Failed to cache student data from cached transcript: {e}")
                has_transcript = True
                print(f"✅ Using cached transcript: {collection_name}")
            except Exception as e:
                print(f"⚠️ Error loading cached transcript: {e}, recreating...")
                # Fall through to create new
                vectorstore_pdf = None
        else:
            vectorstore_pdf = None
        
        # Create new collection if not cached
        if vectorstore_pdf is None:
            loader = PyPDFLoader(temp_pdf_path)
            pdf_docs = redact_student_info(loader.load())
            stored_pdf_docs = pdf_docs  # Store for deterministic extraction
            try:
                latest_student_data = extract_student_data_deterministic(pdf_docs)
            except Exception as e:
                print(f"⚠️ Failed to cache student data on upload: {e}")
            pdf_splits = chunk_with_metadata(pdf_docs)

            # Use cached embeddings
            cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
            vectorstore_pdf = Chroma.from_documents(
                pdf_splits,
                collection_name=collection_name,
                persist_directory=os.path.join(CHROMA_DATA_DIR, "transcripts"),
                embedding=cached_embeddings
            )

            # Track collection creation for TTL cleanup
            track_collection_creation(collection_name, user_id)

            pdf_retriever = vectorstore_pdf.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.5}
            )
            has_transcript = True
            print(f"✅ Transcript loaded and vectorstore built: {collection_name}")
    else:
        pdf_retriever = None
        stored_pdf_docs = None
        has_transcript = False
        print("ℹ️ No transcript provided. Running in no-transcript mode.")

    # ---- Load course-level data (processed_psu_courses.csv) for individual course queries ----
    courses_csv_path = os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv")
    if not os.path.exists(courses_csv_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "processed_psu_courses.csv")
        if os.path.exists(alt_path):
            courses_csv_path = alt_path
        else:
            print("⚠️ processed_psu_courses.csv not found")
            courses_csv_path = None
    
    if courses_csv_path and os.path.exists(courses_csv_path):
        df_courses = pd.read_csv(courses_csv_path)
        course_texts = []
        course_metadata = []

        for _, row in df_courses.iterrows():
            course_text = f"Course: {row.get('course_number', '')}. "
            course_text += f"Title: {row.get('course_name', '')}. "
            course_text += f"Description: {row.get('description', '')}. "
            prereq_str = str(row.get('prerequisite_course_numbers', ''))
            if prereq_str and prereq_str != 'nan':
                course_text += f"Prerequisites: {prereq_str}. "
            course_texts.append(course_text)
            
            course_code = str(row.get('course_number', '')).strip()
            course_metadata.append({
                "course_code": course_code,
                "department": course_code.split()[0] if ' ' in course_code else ''
            })
        
        chroma_courses_dir = os.path.join(CHROMA_DATA_DIR, "courses")
        try:
            if os.path.exists(chroma_courses_dir):
                try:
                    cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                    vectorstore_courses = Chroma(
                        persist_directory=chroma_courses_dir,
                        embedding_function=cached_embeddings
                    )
                    _ = vectorstore_courses.as_retriever(search_kwargs={"k": 1})
                    print("✅ Using existing course-level vectorstore.")
                except Exception as e:
                    print(f"⚠️ Course vectorstore corrupted, recreating... Error: {e}")
                    if os.path.exists(chroma_courses_dir):
                        shutil.rmtree(chroma_courses_dir)
                    cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                    vectorstore_courses = Chroma.from_texts(
                        texts=course_texts,
                        embedding=cached_embeddings,
                        metadatas=course_metadata,
                        persist_directory=chroma_courses_dir
                    )
                    vectorstore_courses.persist()
                    print("✅ Created new course-level vectorstore.")
            else:
                cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                vectorstore_courses = Chroma.from_texts(
                    texts=course_texts,
                    embedding=cached_embeddings,
                    metadatas=course_metadata,
                    persist_directory=chroma_courses_dir
                )
                vectorstore_courses.persist()
                print("✅ Created new course-level vectorstore.")
        except Exception as e:
            print(f"⚠️ Error with course vectorstore: {e}")
            if os.path.exists(chroma_courses_dir):
                shutil.rmtree(chroma_courses_dir)
            cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
            vectorstore_courses = Chroma.from_texts(
                texts=course_texts,
                embedding=cached_embeddings,
                metadatas=course_metadata,
                persist_directory=chroma_courses_dir
            )
            vectorstore_courses.persist()

        course_retriever = vectorstore_courses.as_retriever(search_kwargs={"k": 5})
        print("✅ Course-level vectorstore initialized.")
    else:
        course_retriever = None
        print("⚠️ Course-level retriever not available (processed_psu_courses.csv missing)")

    # ---- Load majors CSV & build Chroma vectorstore for major requirements queries ----
    majors_csv_path = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
    if not os.path.exists(majors_csv_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        if os.path.exists(alt_path):
            majors_csv_path = alt_path
        else:
            print("⚠️ all_abington_majors_combined(5).csv not found")
            majors_csv_path = None
    
    if majors_csv_path and os.path.exists(majors_csv_path):
        df_majors = pd.read_csv(majors_csv_path)
        majors_texts = []
        majors_metadata = []
        
        for _, row in df_majors.iterrows():
            text_data = ""
            metadata = {}
            for col in df_majors.columns:
                value = str(row[col]) if not pd.isna(row[col]) else ""
                text_data += f"{col}: {value}. "
                metadata[col] = str(value) if value else ""
            majors_texts.append(text_data.strip())
            majors_metadata.append(metadata)
        
        chroma_majors_dir = os.path.join(CHROMA_DATA_DIR, "majors")
        try:
            if os.path.exists(chroma_majors_dir):
                try:
                    cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                    vectorstore_majors = Chroma(
                        persist_directory=chroma_majors_dir,
                        embedding_function=cached_embeddings
                    )
                    _ = vectorstore_majors.as_retriever(search_kwargs={"k": 1})
                    print("✅ Using existing majors vectorstore.")
                except Exception as e:
                    print(f"⚠️ Majors vectorstore corrupted, recreating... Error: {e}")
                    if os.path.exists(chroma_majors_dir):
                        shutil.rmtree(chroma_majors_dir)
                    cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                    vectorstore_majors = Chroma.from_texts(
                        texts=majors_texts,
                        embedding=cached_embeddings,
                        metadatas=majors_metadata,
                        persist_directory=chroma_majors_dir
                    )
                    vectorstore_majors.persist()
                    print("✅ Created new majors vectorstore.")
            else:
                cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                vectorstore_majors = Chroma.from_texts(
                    texts=majors_texts,
                    embedding=cached_embeddings,
                    metadatas=majors_metadata,
                    persist_directory=chroma_majors_dir
                )
                vectorstore_majors.persist()
                print("✅ Created new majors vectorstore.")
        except Exception as e:
            print(f"⚠️ Error with majors vectorstore: {e}")
            if os.path.exists(chroma_majors_dir):
                shutil.rmtree(chroma_majors_dir)
            cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
            vectorstore_majors = Chroma.from_texts(
                texts=majors_texts,
                embedding=cached_embeddings,
                metadatas=majors_metadata,
                persist_directory=chroma_majors_dir
            )
            vectorstore_majors.persist()
        
        majors_retriever = vectorstore_majors.as_retriever(search_kwargs={"k": 3})
        print("✅ Majors vectorstore initialized.")
    else:
        majors_retriever = None
        print("⚠️ Majors retriever not available (all_abington_majors_combined(5).csv missing)")

    # ---- Load advising/registrar policy PDF into vectorstore ----
    global policy_retriever, policy_vectorstore
    policy_pdf_path = os.path.join(LLM_DIR, "PennState_Combined_Advising_and_Registrar_VERBATIM.pdf")
    if os.path.exists(policy_pdf_path):
        chroma_policy_dir = os.path.join(CHROMA_DATA_DIR, "policies")
        policy_collection = "advising_registrar_policies_v1"
        try:
            if os.path.exists(chroma_policy_dir):
                try:
                    cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                    policy_vectorstore = Chroma(
                        collection_name=policy_collection,
                        persist_directory=chroma_policy_dir,
                        embedding_function=cached_embeddings
                    )
                    _ = policy_vectorstore.as_retriever(search_kwargs={"k": 1})
                    print("✅ Using existing policy vectorstore.")
                except Exception as e:
                    print(f"⚠️ Policy vectorstore corrupted, recreating... Error: {e}")
                    if os.path.exists(chroma_policy_dir):
                        shutil.rmtree(chroma_policy_dir)
                    cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                    loader = PyPDFLoader(policy_pdf_path)
                    policy_docs = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
                    policy_chunks = splitter.split_documents(policy_docs)
                    for chunk in policy_chunks:
                        chunk.metadata["source"] = "advising_registrar_policy"
                    policy_vectorstore = Chroma.from_documents(
                        policy_chunks,
                        collection_name=policy_collection,
                        persist_directory=chroma_policy_dir,
                        embedding=cached_embeddings
                    )
                    policy_vectorstore.persist()
                    print("✅ Created new policy vectorstore.")
            else:
                cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                loader = PyPDFLoader(policy_pdf_path)
                policy_docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
                policy_chunks = splitter.split_documents(policy_docs)
                for chunk in policy_chunks:
                    chunk.metadata["source"] = "advising_registrar_policy"
                policy_vectorstore = Chroma.from_documents(
                    policy_chunks,
                    collection_name=policy_collection,
                    persist_directory=chroma_policy_dir,
                    embedding=cached_embeddings
                )
                policy_vectorstore.persist()
                print("✅ Created new policy vectorstore.")
        except Exception as e:
            print(f"⚠️ Error with policy vectorstore: {e}")
            policy_vectorstore = None
        policy_retriever = policy_vectorstore.as_retriever(search_kwargs={"k": 4}) if policy_vectorstore else None
        if policy_retriever:
            print("✅ Policy retriever initialized.")
    else:
        policy_retriever = None
        print("⚠️ Policy PDF not found for RAG.")

    # ---- LLM Instances ----
    llm1 = ChatOpenAI(model_name="gpt-4o", temperature=0)
    llm2 = ChatOpenAI(model_name="gpt-4o", temperature=0)
    print(" LLMs initialized.")


# =========================
# Helper Functions
# =========================

def redact_student_info(docs):
    """
    Roughly redact student name and ID from the PDF text.
    """
    redacted_docs = []

    for doc in docs:
        content = doc.page_content

        # Mask patterns like "Name: John Doe 123456789"
        content = re.sub(
            r"\b[A-Z][a-z]+\s[A-Z][a-z]+(?=:\s?\d{6,})",
            "[REDACTED NAME]",
            content
        )

        # Mask 9-digit ID
        content = re.sub(r"\b\d{9}\b", "[REDACTED ID]", content)

        # Optionally mask first occurrence of a "First Last" pattern
        content = re.sub(
            r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b",
            "[REDACTED NAME]",
            content,
            count=1
        )

        doc.page_content = content
        redacted_docs.append(doc)

    return redacted_docs


def extract_completed_and_ip_courses(text):
    pattern = re.findall(r"\b[A-Z]{2,}\s+[0-9]{3}\b", text)
    courses = {re.sub(r"\s+", " ", course.strip()) for course in pattern}
    return courses


def extract_courses_from_term_lines(full_text: str) -> Tuple[List[str], List[str]]:
    """
    Extract completed and in-progress courses from term-based lines like:
    "FA 2024 CMPSC 131 ... 3.00 A" or "... 3.00 IP"
    """
    term_pattern = re.compile(r"\b(FA|SP|SU)\s+\d{4}\s+([A-Z]{2,})\s+(\d{1,3}[A-Z]?)\b")
    grade_pattern = re.compile(r"\b(A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|F|IP|W|WF|LD)\b")

    completed = []
    in_progress = []
    completed_seen = set()
    in_progress_seen = set()

    for match in term_pattern.finditer(full_text):
        course_code = f"{match.group(2)} {match.group(3)}"
        context = full_text[match.end():match.end() + 80]
        grade_match = grade_pattern.search(context)
        grade = grade_match.group(1) if grade_match else ""

        if grade == "IP":
            if course_code not in in_progress_seen:
                in_progress.append(course_code)
                in_progress_seen.add(course_code)
            continue
        if grade in {"W", "WF", "LD", "F"}:
            continue
        if course_code not in completed_seen:
            completed.append(course_code)
            completed_seen.add(course_code)

    return completed, in_progress


def extract_total_credits_from_text(full_text: str) -> Optional[Tuple[float, float, float]]:
    """
    Extract total required/used/needed credits from summary lines.
    Prefer the line with the largest required credits (e.g., 120.00).
    """
    pattern = re.compile(
        r"Units:\s*([\d.]+)\s*required,\s*([\d.]+)\s*used,\s*([\d.]+)\s*needed",
        re.IGNORECASE
    )
    matches = []
    for match in pattern.finditer(full_text):
        try:
            required = float(match.group(1))
            used = float(match.group(2))
            needed = float(match.group(3))
            matches.append((required, used, needed))
        except Exception:
            continue

    if not matches:
        return None

    required, used, needed = max(matches, key=lambda t: (t[0], t[2]))
    return required, used, needed


def extract_student_data_deterministic(pdf_docs: List[Document]) -> dict:
    """
    Extract student data from full transcript text using deterministic parsing.
    More accurate and faster than RAG for structured data.
    """
    full_text = "\n".join(doc.page_content for doc in pdf_docs)
    
    # Extract major - try multiple patterns
    major = None
    
    # Pattern 1: "Major/Option: Computer Science, B.S. (Abington) Req Term: ..."
    # Stop at "Req Term" or end of line
    major_match = re.search(r'Major/Option[:\s]+([^\n]+?)(?:\s+Req\s+Term|$)', full_text, re.IGNORECASE)
    if major_match:
        major = major_match.group(1).strip()
        # Clean up - remove "Req Term" and anything after it
        major = re.sub(r'\s+Req\s+Term.*$', '', major, flags=re.IGNORECASE).strip()
        # Remove trailing punctuation/whitespace
        major = major.rstrip('.,;:')
    
    # Pattern 2: Look for major name with degree type pattern: "Name, B.S. (Location)"
    if not major or len(major) < 5 or 'Req Term' in major:
        # Look for pattern like "Computer Science, B.S. (Abington)"
        degree_pattern = r'([A-Z][a-zA-Z\s&]+(?:,\s*[A-Z][a-zA-Z\s&]+)*)\s*,\s*B\.?[AS]\.?\s*\([^)]+\)'
        match = re.search(degree_pattern, full_text)
        if match:
            major = match.group(1).strip()
            # Try to get the full string including degree
            full_match = re.search(rf'{re.escape(major)}[^\n]*B\.?[AS]\.?[^\n]*\([^)]+\)', full_text, re.IGNORECASE)
            if full_match:
                major = full_match.group(0).strip()
    
    # Pattern 3: If still not found, look for common major keywords
    if not major or len(major) < 5:
        # Look for major names near "Major/Option" label
        major_section = re.search(r'Major/Option[:\s]+([^\n]{10,100})', full_text, re.IGNORECASE)
        if major_section:
            section_text = major_section.group(1)
            # Try to extract just the major name part (before "Req Term" or other labels)
            clean_major = re.sub(r'\s+Req\s+Term.*$', '', section_text, flags=re.IGNORECASE).strip()
            if clean_major and len(clean_major) > 5:
                major = clean_major.split(',')[0].strip()  # Take first part before comma if exists
    
    # Extract completed and in-progress courses (term-based lines)
    completed_courses, in_progress = extract_courses_from_term_lines(full_text)

    # Extract total credits from summary lines
    totals = extract_total_credits_from_text(full_text)
    if totals:
        credits_required, credits_completed, credits_needed = totals
    else:
        credits_match = re.search(
            r'(\d+\.\d+)\s+units?\s+required.*?(\d+\.\d+)\s+used',
            full_text,
            re.IGNORECASE
        )
        credits_completed = float(credits_match.group(2)) if credits_match else 0.0
        credits_required = float(credits_match.group(1)) if credits_match else 120.0
        credits_needed = max(0.0, credits_required - credits_completed)
    
    # Extract unmet requirements (deterministic parsing)
    unmet_requirements = extract_not_satisfied_sections_deterministic(full_text)
    
    # Normalize major name if extracted
    if major:
        major = normalize_major_name(major)
    
    return {
        "major": major,
        "completed_courses": completed_courses,
        "in_progress_courses": in_progress,
        "credits_completed": credits_completed,
        "credits_required": credits_required,
        "credits_needed": credits_needed,
        "unmet_requirements": unmet_requirements
    }


def _clean_major_phrase(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    stopwords = {
        "major", "degree", "program", "student", "studying", "study", "in", "of",
        "ba", "bs", "b", "a", "s", "abington", "penn", "state", "university",
        "freshman", "sophomore", "junior", "senior", "first", "second", "third", "fourth", "year"
    }
    tokens = [t for t in text.split() if t not in stopwords and len(t) > 2]
    return " ".join(tokens)


def normalize_major_name(extracted_major: str) -> str:
    """
    Normalize extracted major name to match CSV format using fuzzy matching.
    """
    if not extracted_major or len(extracted_major.strip()) < 3:
        return None
    
    user_major = extracted_major.strip()
    user_clean = _clean_major_phrase(user_major)
    user_tokens = {w.lower() for w in re.split(r"[^A-Za-z]+", user_clean) if len(w) > 2}
    if not user_tokens and not user_clean:
        return None

    # Load majors CSV to get available majors
    majors_csv_path = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
    if not os.path.exists(majors_csv_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        if os.path.exists(alt_path):
            majors_csv_path = alt_path
        else:
            return extracted_major  # Return as-is if CSV not found
    
    try:
        df_majors = pd.read_csv(majors_csv_path)
        available_majors = df_majors['Major Name'].tolist()
        cleaned_majors = {major: _clean_major_phrase(major) for major in available_majors}
        
        # First, try exact match (case-insensitive)
        for major in available_majors:
            if user_major.lower() == major.lower().strip():
                return major

        # Cleaned exact match (removes degree/abington/etc.)
        for major, cleaned in cleaned_majors.items():
            if user_clean and user_clean == cleaned:
                return major

        # Token subset match: require at least 2 overlapping tokens
        best_token_match = None
        best_token_count = 0
        for major in available_majors:
            major_tokens = {w.lower() for w in re.split(r"[^A-Za-z]+", cleaned_majors.get(major, "")) if len(w) > 2}
            overlap = user_tokens.intersection(major_tokens) if user_tokens else set()
            if len(overlap) >= 2 and len(overlap) > best_token_count:
                best_token_match = major
                best_token_count = len(overlap)
        if best_token_match:
            return best_token_match

        # Single-token match for common majors (e.g., "business")
        if len(user_tokens) == 1:
            token = next(iter(user_tokens))
            best_single = None
            for major, cleaned in cleaned_majors.items():
                if token in cleaned:
                    # Prefer the shortest cleaned major containing the token
                    if not best_single or len(cleaned) < len(cleaned_majors.get(best_single, "")):
                        best_single = major
            if best_single:
                return best_single

        # Substring match (either direction)
        for major in available_majors:
            cleaned = cleaned_majors.get(major, "")
            if user_clean and (user_clean in cleaned or cleaned in user_clean):
                return major

        # Try fuzzy matching with higher threshold to avoid wrong majors
        try:
            from fuzzywuzzy import process
            best_match = process.extractOne(user_clean or user_major, available_majors)
            if best_match and best_match[1] >= 70:  # slightly more forgiving
                return best_match[0]
        except ImportError:
            try:
                import difflib
                candidates = difflib.get_close_matches(user_clean or user_major, available_majors, n=1, cutoff=0.6)
                if candidates:
                    return candidates[0]
            except Exception:
                pass

        # If nothing matches, return None to force the caller to ask for clarification
        return None
    except Exception as e:
        print(f"⚠️ Error normalizing major name: {e}")
        return None


def extract_not_satisfied_sections_deterministic(full_text: str) -> List[dict]:
    """Extract unmet requirements using deterministic parsing (faster than RAG)."""
    unmet = []
    
    # Find all "Not Satisfied" sections
    not_satisfied_pattern = r'Not Satisfied.*?Units:\s*([\d.]+)\s*required.*?([\d.]+)\s*used.*?([\d.]+)\s*needed'
    for match in re.finditer(not_satisfied_pattern, full_text, re.DOTALL | re.IGNORECASE):
        try:
            unmet.append({
                "category": "Requirement",
                "required": float(match.group(1)),
                "used": float(match.group(2)),
                "needed": float(match.group(3))
            })
        except:
            continue
    
    return unmet


def extract_not_satisfied_sections(docs):
    """
    Parse What-If report text to find 'Not Satisfied' / unmet requirement blocks.
    Expects docs to be a list of Documents (e.g., from pdf_retriever.invoke()).
    """
    all_text = "\n".join(doc.page_content for doc in docs)
    print(f"🔍 Analyzing text sample: {all_text[:500]}...")

    # Pattern 1: Standard "Not Satisfied" sections
    pattern1 = re.compile(
        r"(.*?)Not Satisfied.*?Units:\s*([\d.]+)\s*required,\s*([\d.]+)\s*used(?:,\s*([\d.]+)\s*needed)?",
        re.DOTALL | re.IGNORECASE
    )
    results1 = pattern1.findall(all_text)
    print(f"🔍 Pattern 1 matches: {len(results1)}")

    # Pattern 2: Course tables with "needed" credits
    pattern2 = re.compile(
        r"(Units:\s*[\d.]+\s*required,\s*[\d.]+\s*used,\s*([\d.]+)\s*needed)",
        re.IGNORECASE
    )
    results2 = pattern2.findall(all_text)
    print(f"🔍 Pattern 2 matches: {len(results2)}")

    # Pattern 3: More flexible pattern for course tables
    pattern3 = re.compile(
        r"(Units:\s*([\d.]+)\s*required,\s*([\d.]+)\s*used,\s*([\d.]+)\s*needed)",
        re.IGNORECASE
    )
    results3 = pattern3.findall(all_text)
    print(f"🔍 Pattern 3 matches: {len(results3)}")

    extracted = []

    # Process Pattern 1: Not Satisfied sections
    for match in results1:
        category, required, used, needed = match
        extracted.append({
            "category": category.strip(),
            "required": float(required),
            "used": float(used),
            "needed": float(needed) if needed else round(float(required) - float(used), 2),
            "type": "not_satisfied"
        })

    # Process Pattern 2 & 3 only if near "Not Satisfied" context
    for m in pattern2.finditer(all_text):
        full_match = m.group(1)
        needed = m.group(2)
        start_idx = m.start()
        context_start = max(0, start_idx - 500)
        context_text = all_text[context_start:start_idx]
        if re.search(r"Not\s+Satisfied|Complete\s+the\s+following|Unmet|Needed", context_text, re.IGNORECASE):
            if float(needed) > 0:
                extracted.append({
                    "category": "Course Requirements Table",
                    "required": 0,
                    "used": 0,
                    "needed": float(needed),
                    "type": "course_table",
                    "context": full_match.strip()
                })

    for m in pattern3.finditer(all_text):
        full_match = m.group(1)
        required = m.group(2)
        used = m.group(3)
        needed = m.group(4)
        start_idx = m.start()
        context_start = max(0, start_idx - 500)
        context_text = all_text[context_start:start_idx]
        if re.search(r"Not\s+Satisfied|Complete\s+the\s+following|Unmet|Needed", context_text, re.IGNORECASE):
            if float(needed) > 0:
                extracted.append({
                    "category": "Course Requirements Table (Detailed)",
                    "required": float(required),
                    "used": float(used),
                    "needed": float(needed),
                    "type": "course_table_detailed",
                    "context": full_match.strip()
                })

    return extracted


def route_query(user_question: str) -> str:
    """
    Route query to appropriate retriever and processing path.
    Returns: 'course', 'major', 'transcript', 'policy', or 'chat'
    """
    question_lower = user_question.lower()
    
    # Course code pattern (e.g., "CMPSC 131", "MATH 140")
    if re.search(r'\b[A-Z]{2,}\s+\d{3}[A-Z]?\b', user_question, re.IGNORECASE):
        return 'course'

    # Advising/registrar policy questions
    if is_policy_question(question_lower):
        return 'policy'
    
    # If user mentions a class year + a likely major keyword, treat as transcript/plan intent
    year_terms = ['freshman', 'sophomore', 'junior', 'senior', 'first-year', 'second-year', 'third-year', 'fourth-year']
    major_terms = ['computer science', 'biology', 'business', 'psychology', 'engineering', 'math', 'chemistry', 'physics', 'major', 'studying']
    if any(y in question_lower for y in year_terms) and any(m in question_lower for m in major_terms):
        return 'transcript'
    
    # Major requirements keywords
    if any(term in question_lower for term in ['major requirement', 'major courses', 'prescribed', 'option', 'what courses do i need for']):
        return 'major'
    
    # Transcript/planning keywords
    if any(term in question_lower for term in ['graduation plan', 'my transcript', 'my courses', 'what do i need', 'generate plan']):
        return 'transcript'
    
    # Default to general chat
    return 'chat'


def _course_level_num(code: str) -> int:
    m = re.search(r'\b(\d{3})\b', code or '')
    try:
        return int(m.group(1)) if m else 999
    except Exception:
        return 999

def _normalize_year_token(year: str) -> Optional[str]:
    """Normalize year tokens to freshman/sophomore/junior/senior."""
    if not year:
        return None
    year_lower = year.strip().lower()
    year_mapping = {
        'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
        'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
        'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
        'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
    }
    if year_lower in year_mapping:
        return year_mapping[year_lower]
    if year_lower in ['freshman', 'sophomore', 'junior', 'senior']:
        return year_lower
    return None

def _fuzzy_year_from_text(text: str) -> Optional[str]:
    """Handle common typos like 'freshamn' by fuzzy matching."""
    import difflib
    tokens = re.split(r'[^A-Za-z]+', text or "")
    choices = ['freshman', 'sophomore', 'junior', 'senior', 'firstyear', 'secondyear', 'thirdyear', 'fourthyear']
    for token in tokens:
        if len(token) < 5:
            continue
        token_lower = token.lower()
        match = difflib.get_close_matches(token_lower, choices, n=1, cutoff=0.78)
        if match:
            normalized = match[0]
            normalized = normalized.replace('firstyear', 'freshman').replace('secondyear', 'sophomore') \
                                   .replace('thirdyear', 'junior').replace('fourthyear', 'senior')
            return _normalize_year_token(normalized)
    return None

def _format_course_for_display(course: str) -> str:
    if course.startswith("GENED__"):
        return "Gen Ed Elective"
    if course.startswith("OPT__"):
        return "Option Elective"
    return course

def _format_plan_structure(plan_structure: dict, start_semester_number: int = 1) -> str:
    semesters = plan_structure.get("semesters", []) if plan_structure else []
    prereq_map = plan_structure.get("prerequisites", {}) if plan_structure else {}
    if not semesters:
        return "I couldn't generate a semester plan. Please try again or upload your transcript."
    lines = []
    for idx, sem in enumerate(sorted(semesters, key=lambda s: s.get("semester", 0))):
        display_semester = start_semester_number + idx
        lines.append(f"Semester {display_semester}:")
        courses = sem.get("courses", [])
        for course in courses:
            prereqs = prereq_map.get(course, []) or []
            prereq_text = ", ".join(prereqs) if prereqs else "None"
            display_course = _format_course_for_display(course)
            lines.append(f"{display_course} (prerequisites: {prereq_text})")
        lines.append(f"Total Credits: {sem.get('credits', len(courses) * 3)}")
        lines.append("---")
    return "\n".join(lines).rstrip("-\n")


def get_student_data_from_transcript() -> Optional[dict]:
    """Return deterministic student data if transcript is loaded."""
    global stored_pdf_docs, latest_student_data
    if stored_pdf_docs:
        try:
            latest_student_data = extract_student_data_deterministic(stored_pdf_docs)
            return latest_student_data
        except Exception as e:
            print(f"⚠️ Failed to extract student data deterministically: {e}")
    return latest_student_data

def process_psu_courses(user_question):
    """Process course-level queries using course_retriever."""
    global course_retriever
    if course_retriever is None:
        return "Course information not available."
    relevant_courses = course_retriever.invoke(user_question)
    if not relevant_courses:
        return "No relevant courses found."
    context_summary = "\n\n".join(doc.page_content for doc in relevant_courses)
    return context_summary


def extract_major_from_summary(summary_text):
    match = re.search(r"Major/Option\s*:\s*(.+)", summary_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_completed_courses_from_record(student_record):
    """
    Extract all completed and in-progress course codes from the LLM's
    structured student record summary.
    """
    patterns = [
        r'\b([A-Z]{2,}\s+\d{3}[A-Z]?)\b',  # e.g. "CMPSC 472" or "MATH 140A"
        r'\b([A-Z]{2,}\s+\d{3})\b',
        r'([A-Z]{2,}\s+\d{3}[A-Z]?)',
    ]

    courses = []
    for pattern in patterns:
        matches = re.findall(pattern, student_record)
        courses.extend(matches)

    seen = set()
    unique_courses = []
    for course in courses:
        normalized = course.replace(' ', ' ').strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_courses.append(normalized)

    print(f" Extracted {len(unique_courses)} courses: {unique_courses[:10]}...")
    return unique_courses


def validate_prerequisites_in_plan(course_list, major_graph, completed_courses=None):
    """
    Check that for each course in course_list, all prerequisite courses
    are either completed or appear earlier in the list.
    """
    if completed_courses is None:
        completed_courses = []

    invalid_courses = []
    all_available_courses = set(completed_courses + course_list)

    for course in course_list:
        prereqs = [
            u for u, v, d in major_graph.in_edges(course, data=True)
            if d.get("type") == "prereq"
        ]

        if prereqs:
            missing_prereqs = [pr for pr in prereqs if pr not in all_available_courses]

            if missing_prereqs:
                invalid_courses.append({
                    'course': course,
                    'missing_prereqs': missing_prereqs
                })

    return invalid_courses


def enhance_graduation_plan_with_prerequisites(graduation_plan, major_graph):
    """
    Scan the generated plan and, for each course, append a clean
    "prerequisites: ..." clause based on the course graph.
    """
    enhanced_plan = graduation_plan

    course_pattern = r'\b([A-Z]{2,}\s+\d{3})\b'
    courses_in_plan = re.findall(course_pattern, graduation_plan)

    for course in courses_in_plan:
        prereqs = [
            u for u, v, d in major_graph.in_edges(course, data=True)
            if d.get("type") == "prereq"
        ]

        if prereqs:
            course_section_pattern = rf'\*\*{re.escape(course)}[^*]*\*\*'
            course_match = re.search(course_section_pattern, enhanced_plan)

            if course_match:
                course_text = course_match.group()

                # Remove any existing prereq text
                course_text = re.sub(
                    r',\s*prerequisites:\s*None',
                    '',
                    course_text
                )
                course_text = re.sub(
                    r',\s*prerequisites:\s*[^,]*(?:,\s*prerequisites:\s*[^,]*)*',
                    '',
                    course_text
                )

                prereq_text = f", prerequisites: {', '.join(prereqs)}"
                enhanced_plan = enhanced_plan.replace(
                    course_match.group(),
                    course_text + prereq_text
                )

    return enhanced_plan


# =========================
# MAIN: Transcript-based plan
# =========================

def aggregate_and_generate_response(user_question=None):
    """
    Main function to generate graduation plan using deterministic extraction and planning.
    Requires that a transcript was uploaded and stored_pdf_docs is available.
    """
    global stored_pdf_docs, has_transcript
    
    if not has_transcript or stored_pdf_docs is None:
        return "No transcript found. Please upload a What-If Report, or tell me your major and year level (e.g., 'I'm a Computer Science freshman') to generate a graduation plan."

    print("\n🔹 Using Deterministic Extraction and Planning...")

    # Step 1: Deterministic extraction (no RAG)
    student_data = extract_student_data_deterministic(stored_pdf_docs)
    print(f"✅ Extracted major: {student_data.get('major')}")
    completed_courses = student_data.get("completed_courses", [])
    in_progress_courses = student_data.get("in_progress_courses", [])
    completed_and_ip = []
    for course in completed_courses + in_progress_courses:
        if course and course not in completed_and_ip:
            completed_and_ip.append(course)

    print(f"✅ Extracted {len(completed_courses)} completed courses")
    if in_progress_courses:
        print(f"✅ Extracted {len(in_progress_courses)} in-progress courses")
    print(f"✅ Credits completed: {student_data.get('credits_completed')}")
    print(f"✅ Credits needed: {student_data.get('credits_needed')}")

    # Step 2: Deterministic planning
    from graduationFlowchart import generate_semester_plan
    
    courses_csv = os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv")
    majors_csv = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")

    # Ensure CSV paths exist
    if not os.path.exists(courses_csv):
        alt_courses = os.path.join(os.path.dirname(__file__), "..", "LLM", "processed_psu_courses.csv")
        if os.path.exists(alt_courses):
            courses_csv = alt_courses
    
    if not os.path.exists(majors_csv):
        alt_majors = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        if os.path.exists(alt_majors):
            majors_csv = alt_majors
    
    # Normalize major name before planning
    extracted_major = student_data.get("major")
    if extracted_major:
        normalized_major = normalize_major_name(extracted_major)
        if normalized_major != extracted_major:
            print(f"🔄 Normalized major: '{extracted_major}' → '{normalized_major}'")
        student_data["major"] = normalized_major
    else:
        # Try to extract from PDF text using LLM as fallback
        print("⚠️ Could not extract major deterministically, trying LLM extraction...")
        try:
            transcript_context = "\n".join(doc.page_content[:500] for doc in stored_pdf_docs[:3])
            major_extraction = llm1.invoke([
                SystemMessage(content="Extract the student's major name from the transcript. Return ONLY the full major name as it appears in university records (e.g., 'Computer Science, B.S. (Abington)')."),
                HumanMessage(content=f"Extract major from: {transcript_context[:1000]}")
            ]).content.strip()
            if major_extraction:
                student_data["major"] = normalize_major_name(major_extraction)
        except:
            pass
    
    if not student_data.get("major"):
        return "I couldn't determine your major from the transcript. Please specify your major (e.g., 'Computer Science, B.S. (Abington)') or contact support."
    
    try:
        plan_structure = generate_semester_plan(
            major_name=student_data.get("major"),
            completed_courses=completed_and_ip,
            unmet_requirements=student_data.get("unmet_requirements", []),
            courses_csv_path=courses_csv,
            majors_csv_path=majors_csv,
            credits_needed=student_data.get("credits_needed")
        )
        print(f"✅ Generated plan with {len(plan_structure['semesters'])} semesters")
    except ValueError as e:
        error_msg = str(e)
        if "not found in majors CSV" in error_msg:
            # Try fuzzy matching to suggest correct major
            print(f"⚠️ Major '{student_data.get('major')}' not found, trying fuzzy match...")
            try:
                from fuzzywuzzy import process
                df_majors = pd.read_csv(majors_csv)
                available_majors = df_majors['Major Name'].tolist()
                best_match = process.extractOne(student_data.get("major"), available_majors)
                if best_match and best_match[1] >= 50:
                    suggested_major = best_match[0]
                    return f"I couldn't find the exact major '{student_data.get('major')}' in the system. Did you mean '{suggested_major}'? Please verify your major name or contact support."
            except:
                pass
            return f"I couldn't find the major '{student_data.get('major')}' in the system. Please verify your major name is correct, or contact support."
        else:
            print(f"⚠️ Error in deterministic planning: {e}")
            import traceback
            traceback.print_exc()
            return f"I encountered an error generating your plan: {str(e)}. Please try again or contact support."
    except Exception as e:
        print(f"⚠️ Error in deterministic planning: {e}")
        import traceback
        traceback.print_exc()
        return f"I encountered an error generating your plan: {str(e)}. Please try again or contact support."

    # Step 3: LLM only narrates/formats
    print("\n🔹 Formatting plan with LLM...")
    
    # Build prerequisites string for each course
    prereq_info = plan_structure.get('prerequisites', {})
    
    plan_narrative = llm2.invoke([
        SystemMessage(content="""You are an academic advisor. Format this graduation plan into clear, 
        organized text. For each course, show the course code and its prerequisites (if any) next to it.
        Do NOT include rationale or explanations - just list the courses with prerequisites.
        Format: Course Code (prerequisites: PREREQ1, PREREQ2) or Course Code (no prerequisites)"""),
        HumanMessage(content=f"""Format this graduation plan:
        
        Student Major: {student_data.get('major', 'Unknown')}
        Completed Courses: {len(student_data.get('completed_courses', []))}
        Credits Completed: {student_data.get('credits_completed', 0)}
        Credits Needed: {student_data.get('credits_needed', 0)}
        
        Plan Structure:
        {json.dumps(plan_structure, indent=2)}
        
        Prerequisites Map:
        {json.dumps(prereq_info, indent=2)}
        
        Create a semester-by-semester plan. For each semester:
        1. List all courses in that semester
        2. For each course, show: Course Code (prerequisites: X, Y) or Course Code (no prerequisites)
        3. Show total credits for the semester
        4. Do NOT include rationale or explanations
        5. Keep it concise and organized""")
    ]).content

    return plan_narrative
    course_list = re.findall(r'\b([A-Z]{2,}\s+\d{3})\b', enhanced_plan)
    validation_warnings = validate_prerequisites_in_plan(
        course_list,
        major_specific_graph,
        completed_courses
    )

    if validation_warnings:
        print("⚠️ Validation warnings (not shown to user):")
        for warning in validation_warnings:
            print(f"   - {warning['course']}: needs {', '.join(warning['missing_prereqs'])}")

    return enhanced_plan


def simple_extract_major_and_year(user_text):
    """Simple pattern-based extraction as fallback."""
    import re
    major = None
    year = None
    
    # Common major patterns
    major_patterns = [
        r'(?:studying|major in|major is|studies)\s+([^,\.\n]+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:major|degree)',
        r'(computer science|biology|business|engineering|psychology|mathematics|chemistry|physics|english|history|economics|accounting|finance|marketing|management)',
    ]
    
    # Year patterns
    year_patterns = [
        r'\b(freshman|sophomore|junior|senior|first-year|second-year|third-year|fourth-year|1st year|2nd year|3rd year|4th year)\b',
    ]
    
    text_lower = user_text.lower()
    
    # Extract year
    for pattern in year_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            year_str = match.group(1).lower()
            year_mapping = {
                'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
            }
            year = year_mapping.get(year_str, year_str)
            if year in ['freshman', 'sophomore', 'junior', 'senior']:
                break
    
    # Extract major - try to find in CSV
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        majors_df = pd.read_csv(csv_path)
        
        # Common major keywords
        major_keywords = {
            'computer science': ['computer science', 'cs', 'comp sci'],
            'biology': ['biology', 'bio'],
            'business': ['business', 'business administration'],
            'psychology': ['psychology', 'psych'],
            'mathematics': ['mathematics', 'math'],
        }
        
        for keyword, variations in major_keywords.items():
            if any(var in text_lower for var in variations):
                # Find matching major in CSV
                matching = majors_df[majors_df['Major Name'].str.contains(keyword, case=False, na=False)]
                if not matching.empty:
                    major = matching.iloc[0]['Major Name']
                    break
        
        # If not found, try direct search
        if not major:
            for major_name in majors_df['Major Name'].tolist():
                major_lower = major_name.lower()
                # Check if any significant word from major name is in user text
                major_words = [w for w in major_lower.split() if len(w) > 3]
                if any(word in text_lower for word in major_words):
                    major = major_name
                    break
    except Exception as e:
        print(f"⚠️ Error in simple extraction: {e}")
    
    return major, year

def extract_major_and_year_from_text(user_text, conversation_history):
    """Extract major and year level from user text using LLM."""
    try:
        # Get list of available majors from CSV
        csv_path = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
        if not os.path.exists(csv_path):
            # Try alternative path
            csv_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        majors_df = pd.read_csv(csv_path)
        available_majors = majors_df['Major Name'].tolist()[:50]  # Limit to first 50 for prompt size
        
        # Build context from conversation history
        history_context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-4:]  # Last 4 messages
            history_context = "\n\nPrevious conversation:\n"
            for msg in recent_history:
                role = "Student" if msg.get('role') == 'user' else "AI"
                history_context += f"{role}: {msg.get('content', '')}\n"
        
        # Use LLM to extract major and year
        # Show some common majors first
        common_majors = [m for m in available_majors if any(x in m.lower() for x in ['computer', 'biology', 'business', 'psychology', 'mathematics'])][:10]
        other_majors = [m for m in available_majors if m not in common_majors][:10]
        
        extraction_prompt = f"""Extract the student's major and year level from their message.

Common majors: {', '.join(common_majors)}
Other majors available: {', '.join(other_majors)}

Year levels: freshman, sophomore, junior, senior (or first-year, second-year, etc.)

Student message: "{user_text}"
{history_context}

CRITICAL INSTRUCTIONS:
1. Look carefully for year level words: "freshman", "sophomore", "junior", "senior", "first-year", "second-year", etc.
2. Even if the student says "I am a freshman" or "I am a computer science major freshman", extract BOTH the major AND the year.
3. The year might appear anywhere in the message or conversation history.

IMPORTANT: Respond ONLY in this exact format (one line, no other text):
MAJOR: [exact major name from list, or closest match]
YEAR: [freshman OR sophomore OR junior OR senior]

Examples:
- "I'm a computer science freshman" → MAJOR: Computer Science, B.S. (Abington) YEAR: freshman
- "I am a computer science. major freshman, give me a graduation plan" → MAJOR: Computer Science, B.S. (Abington) YEAR: freshman
- "I am a freshman studying computer science" → MAJOR: Computer Science, B.S. (Abington) YEAR: freshman
- "biology sophomore" → MAJOR: Biology, B.S. (Abington) YEAR: sophomore
- "business junior" → MAJOR: Business, B.A. (Abington) YEAR: junior

If you cannot determine, use:
MAJOR: UNKNOWN
YEAR: UNKNOWN"""
        
        extraction_result = llm1.invoke([
            SystemMessage(content="You are a precise information extractor. Extract major and year level from student messages."),
            HumanMessage(content=extraction_prompt)
        ]).content
        
        # Parse the result - be more flexible with parsing
        major = None
        year = None
        
        # Try multiple parsing strategies
        extraction_result_lower = extraction_result.lower()
        
        # Strategy 1: Look for explicit MAJOR: and YEAR: labels
        for line in extraction_result.split('\n'):
            line_upper = line.upper()
            if 'MAJOR:' in line_upper:
                major = line.split(':', 1)[1].strip()
                if major == 'UNKNOWN':
                    major = None
            elif 'YEAR:' in line_upper:
                year = line.split(':', 1)[1].strip().lower()
                if year == 'UNKNOWN':
                    year = None
                # Normalize year terms
                year_mapping = {
                    'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                    'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                    'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                    'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
                }
                year = year_mapping.get(year, year)
        
        # Strategy 2: If year not found, try direct pattern matching in the LLM response
        if not year:
            year_patterns = [
                r'\b(freshman|sophomore|junior|senior)\b',
                r'\b(first-year|second-year|third-year|fourth-year)\b',
                r'\b(1st year|2nd year|3rd year|4th year)\b'
            ]
            for pattern in year_patterns:
                match = re.search(pattern, extraction_result_lower, re.IGNORECASE)
                if match:
                    year_str = match.group(1).lower()
                    year_mapping = {
                        'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                        'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                        'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                        'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
                    }
                    year = year_mapping.get(year_str, year_str)
                    if year in ['freshman', 'sophomore', 'junior', 'senior']:
                        break
        
        # Strategy 3: If still no year, check the original user text and conversation history
        if not year:
            # Check current user message
            year_patterns = [
                r'\b(freshman|sophomore|junior|senior)\b',
                r'\b(first-year|second-year|third-year|fourth-year)\b',
                r'\b(1st year|2nd year|3rd year|4th year)\b'
            ]
            for pattern in year_patterns:
                match = re.search(pattern, user_text.lower(), re.IGNORECASE)
                if match:
                    year_str = match.group(1).lower()
                    year_mapping = {
                        'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                        'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                        'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                        'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
                    }
                    year = year_mapping.get(year_str, year_str)
                    if year in ['freshman', 'sophomore', 'junior', 'senior']:
                        break
            
            # Check conversation history if still not found
            if not year and conversation_history:
                for msg in reversed(conversation_history[-4:]):
                    if msg.get('role') == 'user':
                        msg_text = msg.get('content', '').lower()
                        for pattern in year_patterns:
                            match = re.search(pattern, msg_text, re.IGNORECASE)
                            if match:
                                year_str = match.group(1).lower()
                                year_mapping = {
                                    'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                                    'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                                    'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                                    'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
                                }
                                year = year_mapping.get(year_str, year_str)
                                if year in ['freshman', 'sophomore', 'junior', 'senior']:
                                    break
                        if year:
                            break

        # Strategy 4: Fuzzy year matching for typos (e.g., "freshamn")
        if not year:
            year = _fuzzy_year_from_text(user_text)
        
        # Try to match major name with available majors using fuzzy matching if needed
        if major and major != 'UNKNOWN':
            # Check if major is in the list or find close match
            df = majors_df
            matching_major = df[df['Major Name'].str.contains(major, case=False, na=False)]
            if not matching_major.empty:
                major = matching_major.iloc[0]['Major Name']
            else:
                # Try fuzzy matching on first few words
                try:
                    from fuzzywuzzy import process
                    major_names = majors_df['Major Name'].tolist()
                    best_match = process.extractOne(major, major_names)
                    if best_match and best_match[1] >= 70:  # 70% similarity threshold
                        major = best_match[0]
                    else:
                        major = None
                except ImportError:
                    # Fallback: try partial string matching
                    for major_name in majors_df['Major Name'].tolist():
                        if major.lower() in major_name.lower() or major_name.lower() in major.lower():
                            major = major_name
                            break
                    else:
                        major = None
        
        # Final fallback: simple deterministic extraction from user text
        if not major or not year:
            simple_major, simple_year = simple_extract_major_and_year(user_text)
            major = major or simple_major
            year = year or simple_year

        # Fuzzy year matching for misspellings in user text
        if not year:
            year = _fuzzy_year_from_text(user_text)

        # Normalize major to a CSV-backed canonical name
        if major:
            normalized_major = normalize_major_name(major)
            if normalized_major:
                major = normalized_major
        
        print(f"🔍 Extracted - Major: {major}, Year: {year}")
        print(f"🔍 LLM raw response: {extraction_result[:200]}...")  # Debug: show first 200 chars of LLM response
        return major, year
        
    except Exception as e:
        print(f"⚠️ Error extracting major/year: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_conversational_response(user_question, conversation_history):
    """Generate a conversational response using conversation history for context."""
    print("\n🔹 Generating Conversational Response...")
    
    global llm1, llm2, has_transcript, stored_pdf_docs, latest_student_data
    global course_retriever, majors_retriever, pdf_retriever, policy_retriever

    if llm1 is None or llm2 is None:
        print("⚠️ LLMs not initialized. Initializing now...")
        try:
            setup_environment(None)
        except Exception as e:
            print(f"⚠️ Error initializing LLMs: {e}")
            from langchain_openai import ChatOpenAI
            if not os.getenv('OPENAI_API_KEY'):
                os.environ['OPENAI_API_KEY'] = "REDACTED"
            if not os.getenv('LANGCHAIN_API_KEY'):
                os.environ['LANGCHAIN_API_KEY'] = 'REDACTED'
            llm1 = ChatOpenAI(model_name="gpt-4o", temperature=0)
            llm2 = ChatOpenAI(model_name="gpt-4o", temperature=0)

    query_type = route_query(user_question)
    student_data = get_student_data_from_transcript()

    # Quick course prerequisite path
    if query_type == 'course':
        course_match = re.search(r'\b([A-Z]{2,})\s+(\d{3}[A-Z]?)\b', user_question, re.IGNORECASE)
        if course_match:
            course_code = f"{course_match.group(1).upper()} {course_match.group(2)}"
            try:
                try:
                    from graphExperiment import course_graph as cg
                    if cg:
                        prereq_text = get_course_prerequisites(course_code, cg)
                        return prereq_text
                except ImportError:
                    pass
            except Exception as e:
                print(f"⚠️ Could not fetch prerequisites: {e}")
        course_info = process_psu_courses(user_question)
        if course_info and "No relevant courses found" not in course_info:
            return course_info
        return "I couldn't find that course. Please check the code (e.g., 'CMPSC 131')."

    # Policy questions directly answered from the policy PDF
    if query_type == 'policy':
        if policy_retriever:
            try:
                policy_docs = policy_retriever.invoke(user_question)
                if policy_docs:
                    policy_context = "\n\n".join(doc.page_content[:900] for doc in policy_docs[:4])
                    response = llm2.invoke([
                        SystemMessage(content="Answer the student's question using ONLY the provided policy text. If the answer isn't present, say you couldn't find it in the policy."),
                        HumanMessage(content=f"Policy context:\n{policy_context}\n\nQuestion: {user_question}")
                    ]).content
                    return response
            except Exception as e:
                print(f"⚠️ Could not retrieve policy information: {e}")
        return "I couldn't find that policy information. Please rephrase your question."

    # Transcript-based planning or transcript Q&A
    if query_type == 'transcript':
        plan_keywords = ['graduation plan', 'generate plan', 'create plan', 'semester plan', 'course plan', 'plan', 'schedule', 'next semester']
        wants_plan = any(k in user_question.lower() for k in plan_keywords)

        has_transcript_available = stored_pdf_docs is not None and has_transcript

        if wants_plan:
            if has_transcript_available:
                print("📋 Detected plan request with transcript, using deterministic pipeline...")
                return aggregate_and_generate_response(user_question)

            print("📋 Plan request without transcript, falling back to major/year extraction...")
            try:
                major, year = extract_major_and_year_from_text(user_question, conversation_history)
                if not major or not year:
                    simple_major, simple_year = simple_extract_major_and_year(user_question)
                    major = major or simple_major
                    year = year or simple_year
                if not year:
                    year = _fuzzy_year_from_text(user_question)
                if major and year:
                    return generate_plan_from_major_and_year(major, year)
            except Exception as e:
                print(f"⚠️ Error generating plan without transcript: {e}")
            return "I need your major and year (e.g., 'Computer Science freshman') or a transcript upload to build a plan."

        if student_data:
            credits_needed = student_data.get("credits_needed", "unknown")
            credits_completed = student_data.get("credits_completed", "unknown")
            unmet = student_data.get("unmet_requirements", [])
            unmet_summary = "\n".join(
                [f"- {item.get('category', 'Requirement')}: {item.get('needed', 0)} credits needed"]
                for item in unmet[:5]
            ) if unmet else "No unmet requirements found."
            return (
                f"From your transcript:\n"
                f"- Major: {student_data.get('major', 'Unknown')}\n"
                f"- Credits completed: {credits_completed}\n"
                f"- Credits remaining: {credits_needed}\n"
                f"- Top unmet requirements:\n{unmet_summary}\n"
                f"If you want a detailed plan, ask me to generate a graduation plan."
            )
        return "I couldn't read a transcript. Please upload one so I can answer transcript-specific questions."

    # Major requirements lookups
    if query_type == 'major':
        major_info = process_psu_courses(user_question)
        if major_info and "No relevant courses found" not in major_info:
            return major_info
        return "I couldn't find that major's requirements. Please provide the exact major name."

    # Otherwise, use conversational approach with context or short recommendations
    try:
        if any(k in user_question.lower() for k in ['next semester', 'upcoming semester', 'what should i take next', 'recommend courses', 'recommend me courses']):
            if not student_data:
                if stored_pdf_docs:
                    try:
                        student_data = extract_student_data_deterministic(stored_pdf_docs)
                        latest_student_data = student_data
                    except Exception as e:
                        print(f"⚠️ Failed to extract student data: {e}")
                        student_data = latest_student_data
                else:
                    student_data = latest_student_data

            if student_data:
                completed = set(student_data.get("completed_courses", []))
                in_progress = set(student_data.get("in_progress_courses", []))
                all_done = completed.union(in_progress)
                major_raw = student_data.get("major", "Unknown") or "Unknown"
                major = normalize_major_name(major_raw) or major_raw

                remaining_courses = []
                try:
                    major_course_codes = get_major_course_codes(major) or []
                    if not major_course_codes:
                        majors_csv = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
                        if os.path.exists(majors_csv):
                            dfm = pd.read_csv(majors_csv)
                            candidates = dfm[dfm['Major Name'].str.contains(major, case=False, na=False)]
                            if not candidates.empty:
                                try:
                                    major_course_codes = get_major_course_codes(candidates.iloc[0]['Major Name']) or []
                                except Exception:
                                    major_course_codes = []
                    for c in major_course_codes:
                        if c not in all_done:
                            remaining_courses.append(c)
                except Exception as e:
                    print(f"⚠️ Could not load major course codes: {e}")
                    try:
                        from graduationFlowchart import GraduationFlowchart
                        majors_csv = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
                        if os.path.exists(majors_csv):
                            flowchart = GraduationFlowchart(
                                os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv"),
                                majors_csv
                            )
                            row = flowchart._get_major_row(major)
                            if row is not None:
                                from graduationFlowchart import extract_all_courses, resolve_common_or_groups
                                prescribed = resolve_common_or_groups(extract_all_courses(str(row.get("Prescribed Courses", "") or "")))
                                for c in prescribed:
                                    if c not in all_done:
                                        remaining_courses.append(c)
                    except Exception as e2:
                        print(f"⚠️ Fallback also failed: {e2}")

                remaining_courses = sorted(set(remaining_courses), key=_course_level_num)
                suggested = remaining_courses[:5] if remaining_courses else []
                if not suggested:
                    try:
                        from graduationFlowchart import generate_semester_plan
                        courses_csv = os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv")
                        majors_csv = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
                        completed_and_ip = list(all_done)
                        plan = generate_semester_plan(
                            major_name=major or "Unknown",
                            completed_courses=completed_and_ip,
                            unmet_requirements=student_data.get("unmet_requirements", []),
                            courses_csv_path=courses_csv,
                            majors_csv_path=majors_csv,
                            credits_needed=student_data.get("credits_needed")
                        )
                        if plan.get("semesters"):
                            next_sem = sorted(plan["semesters"], key=lambda s: s.get("semester", 0))[0]
                            suggested = next_sem.get("courses", [])[:5]
                    except Exception as e:
                        print(f"⚠️ Could not build next-semester suggestion from plan: {e}")

                if suggested:
                    return (
                        f"Here are recommended courses for your next semester (based on your transcript and major {major}):\n"
                        + "\n".join([f"- {c}" for c in suggested])
                        + "\nIf you'd like a full graduation plan, ask me to generate one."
                    )
                return "I couldn't confidently pick next-semester courses from your record. Please ask me to generate a full graduation plan."

            if has_transcript and stored_pdf_docs:
                return "I'm processing your transcript. Please try asking again in a moment, or ask me to generate a full graduation plan."
            return "I need your transcript to recommend next-semester courses. Please upload it or tell me your major/year."

        student_record = ""
        major = ""
        completed_courses = []
        try:
            if student_data:
                major = student_data.get("major", "") or ""
                completed_courses = student_data.get("completed_courses", [])
                student_record = json.dumps(student_data, indent=2)
        except Exception as e:
            print(f"⚠️ Could not retrieve student record: {e}")
        
        conversation_context = ""
        if conversation_history and len(conversation_history) > 1:
            recent_history = conversation_history[-6:]
            conversation_context = "\n\nPrevious conversation:\n"
            for msg in recent_history:
                role = "Student" if msg.get('role') == 'user' else "AI Advisor"
                conversation_context += f"{role}: {msg.get('content', '')}\n"
        
        system_content = """You are a friendly and helpful AI Academic Advisor. You help students with:
- Course planning and scheduling
- Prerequisite questions
- Major requirements
- Graduation planning
- General academic advice

Be conversational, helpful, and empathetic. Use the conversation history to maintain context and provide personalized advice. If the student asks about their graduation plan, you can generate one. If they ask general questions, answer them naturally using the available course and student information."""
        
        if student_record:
            system_content += f"\n\nStudent Information:\n{student_record}\n"
            if major:
                system_content += f"Major: {major}\n"
            if completed_courses:
                system_content += f"Completed Courses: {', '.join(completed_courses[:10])}\n"
        
        course_context = ""
        policy_context = ""
        if policy_retriever and is_policy_question(user_question):
            try:
                policy_docs = policy_retriever.invoke(user_question)
                if policy_docs:
                    policy_context = "\n\nRelevant Advising/Registrar Policy:\n" + "\n\n".join(
                        doc.page_content[:600] for doc in policy_docs[:3]
                    )
            except Exception as e:
                print(f"⚠️ Could not retrieve policy information: {e}")

        try:
            if query_type == 'course':
                if course_retriever:
                    relevant_docs = course_retriever.invoke(user_question)
                    if relevant_docs:
                        course_context = "\n\nRelevant Course Information:\n" + "\n\n".join(
                            doc.page_content[:500] for doc in relevant_docs[:3]
                        )
            elif query_type == 'major':
                if majors_retriever:
                    relevant_docs = majors_retriever.invoke(user_question)
                    if relevant_docs:
                        course_context = "\n\nRelevant Major Requirements:\n" + "\n\n".join(
                            doc.page_content[:500] for doc in relevant_docs[:3]
                        )
            elif query_type == 'transcript':
                if pdf_retriever:
                    relevant_docs = pdf_retriever.invoke(user_question)
                    if relevant_docs:
                        course_context = "\n\nRelevant Transcript Information:\n" + "\n\n".join(
                            doc.page_content[:500] for doc in relevant_docs[:3]
                        )
            else:
                if course_retriever:
                    course_docs = course_retriever.invoke(user_question)
                    if course_docs:
                        course_context += "\n\nRelevant Course Information:\n" + "\n\n".join(
                            doc.page_content[:500] for doc in course_docs[:2]
                        )
                if majors_retriever:
                    major_docs = majors_retriever.invoke(user_question)
                    if major_docs:
                        course_context += "\n\nRelevant Major Requirements:\n" + "\n\n".join(
                            doc.page_content[:500] for doc in major_docs[:2]
                        )
            if policy_context:
                course_context += policy_context
        except Exception as e:
            print(f"⚠️ Could not retrieve course information: {e}")
        
        full_prompt = f"{conversation_context}\n\nCurrent Question: {user_question}"
        if course_context:
            full_prompt += course_context
        
        response = llm2.invoke([
            SystemMessage(content=system_content),
            HumanMessage(content=full_prompt)
        ]).content
        
        return response
        
    except Exception as e:
        import traceback
        print(f"Error in conversational response: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return "I apologize, but I encountered an error processing your question. Please try rephrasing it or ask me to generate a graduation plan if you haven't already uploaded your transcript."


# =========================
# NEW: Major + Year (no transcript)
# =========================

def generate_plan_from_major_and_year(major, year_level, target_grad_in_years=4):
    """
    Generate a graduation plan when the student does NOT upload a transcript.
    Uses major, year_level, course graph, and CSV-based major requirements.
    """
    global llm1, llm2, course_graph
    
    # Ensure LLMs are initialized
    if llm1 is None or llm2 is None:
        print("⚠️ LLMs not initialized. Initializing now...")
        try:
            setup_environment(None)  # Initialize without transcript
        except Exception as e:
            print(f"⚠️ Error in setup_environment: {e}")
            # Manual initialization as fallback
            from langchain_openai import ChatOpenAI
            if not os.getenv('OPENAI_API_KEY'):
                os.environ['OPENAI_API_KEY'] = "REDACTED"
            if not os.getenv('LANGCHAIN_API_KEY'):
                os.environ['LANGCHAIN_API_KEY'] = 'REDACTED'
            llm1 = ChatOpenAI(model_name="gpt-4o", temperature=0)
            llm2 = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # Ensure course_graph is available
    if 'course_graph' not in globals() or course_graph is None:
        print("⚠️ Course graph not available. Importing...")
        try:
            from graphExperiment import course_graph as cg
            course_graph = cg
        except Exception as e:
            print(f"⚠️ Could not import course_graph: {e}")
            course_graph = None
    
    # Normalize year input (handles typos and variants)
    year_level = _normalize_year_token(year_level) or year_level
    if year_level not in ['freshman', 'sophomore', 'junior', 'senior']:
        year_level = _fuzzy_year_from_text(year_level or "") or "freshman"

    # Calculate semesters completed and remaining
    typical_total_credits = 120
    year_lower = year_level.lower()

    if year_lower == "freshman":
        assumed_completed = 0
        semesters_completed = 0
    elif year_lower == "sophomore":
        assumed_completed = 30
        semesters_completed = 2
    elif year_lower == "junior":
        assumed_completed = 60
        semesters_completed = 4
    elif year_lower == "senior":
        assumed_completed = 90
        semesters_completed = 6
    else:
        assumed_completed = 0
        semesters_completed = 0

    total_credits_needed = max(0, typical_total_credits - assumed_completed)
    semesters_remaining = 8 - semesters_completed
    credits_per_semester = total_credits_needed / semesters_remaining if semesters_remaining > 0 else 15
    
    print(f"📊 Student status: {year_level}, {assumed_completed} credits completed, {total_credits_needed} credits needed")
    print(f"📊 Semesters remaining: {semesters_remaining} ({credits_per_semester:.1f} credits per semester)")

    # Normalize major and load CSVs
    major_norm = normalize_major_name(major)
    if not major_norm:
        return f"I couldn't match the major '{major}' to a known program. Please provide the exact major name as listed in the catalog (e.g., 'Business, B.A. (Abington)')."
    courses_csv = os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv")
    majors_csv = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
    if not os.path.exists(courses_csv):
        alt_courses = os.path.join(os.path.dirname(__file__), "..", "LLM", "processed_psu_courses.csv")
        if os.path.exists(alt_courses):
            courses_csv = alt_courses
    if not os.path.exists(majors_csv):
        alt_majors = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        if os.path.exists(alt_majors):
            majors_csv = alt_majors

    # Deterministic schedule using graduationFlowchart
    from graduationFlowchart import generate_semester_plan
    try:
        # First build a full plan to infer completed courses for upperclassmen
        full_plan = generate_semester_plan(
            major_name=major_norm,
            completed_courses=[],
            unmet_requirements=[],
            courses_csv_path=courses_csv,
            majors_csv_path=majors_csv,
            credits_needed=typical_total_credits
        )
        completed_courses = []
        if semesters_completed > 0:
            for sem in full_plan.get("semesters", []):
                if sem.get("semester", 0) <= semesters_completed:
                    completed_courses.extend(sem.get("courses", []))

        plan_structure = generate_semester_plan(
            major_name=major_norm,
            completed_courses=completed_courses,
            unmet_requirements=[],
            courses_csv_path=courses_csv,
            majors_csv_path=majors_csv,
            credits_needed=total_credits_needed
        )
        print(f"✅ Generated deterministic plan with {len(plan_structure.get('semesters', []))} semesters")
    except Exception as e:
        print(f"⚠️ Error generating deterministic plan (no transcript): {e}")
        # Try fuzzy match on major name as a fallback
        try:
            dfm = pd.read_csv(majors_csv)
            try:
                from fuzzywuzzy import process
                best_match = process.extractOne(major_norm, dfm['Major Name'].tolist())
                if best_match and best_match[1] >= 60:
                    print(f"🔄 Retrying with fuzzy matched major: {best_match[0]}")
                    plan_structure = generate_semester_plan(
                        major_name=best_match[0],
                        completed_courses=[],
                        unmet_requirements=[],
                        courses_csv_path=courses_csv,
                        majors_csv_path=majors_csv,
                        credits_needed=total_credits_needed
                    )
                else:
                    return "I couldn't generate a plan without your transcript right now. Please upload your transcript or try again."
            except Exception:
                # Fallback to stdlib fuzzy matching if fuzzywuzzy is unavailable
                import difflib
                candidates = difflib.get_close_matches(
                    major_norm,
                    dfm['Major Name'].tolist(),
                    n=1,
                    cutoff=0.6
                )
                if candidates:
                    print(f"🔄 Retrying with difflib matched major: {candidates[0]}")
                    plan_structure = generate_semester_plan(
                        major_name=candidates[0],
                        completed_courses=[],
                        unmet_requirements=[],
                        courses_csv_path=courses_csv,
                        majors_csv_path=majors_csv,
                        credits_needed=total_credits_needed
                    )
                else:
                    return "I couldn't generate a plan without your transcript right now. Please upload your transcript or try again."
        except Exception as e2:
            print(f"⚠️ Fuzzy fallback failed: {e2}")
            return "I couldn't generate a plan without your transcript right now. Please upload your transcript or try again."

    # Deterministic formatting (avoid LLM hallucinations)
    start_semester_number = semesters_completed + 1 if semesters_completed > 0 else 1
    return _format_plan_structure(plan_structure, start_semester_number=start_semester_number)


# =========================
# Flask App / Endpoint
# =========================

app = Flask(__name__)

@app.route("/generate_plan", methods=["POST"])
def generate_plan_route():
    """
    Unified endpoint:
    - If 'transcript' file is present → transcript-based plan.
    - Else → expects JSON or form with 'major' and 'year' to generate plan.
    """
    # If there's a transcript file, use transcript-based mode
    if 'transcript' in request.files:
        setup_environment(request)
        plan = aggregate_and_generate_response(
            user_question="Generate my graduation plan from my transcript."
        )
        return jsonify({"plan": plan})

    # Otherwise, expect major + year
    data = request.get_json(silent=True) or request.form
    major = data.get("major")
    year_level = data.get("year")

    if not major or not year_level:
        return jsonify({
            "error": "Please provide 'major' and 'year' (e.g., 'Computer Science, BS (CMPAB_BS)' and 'junior')."
        }), 400

    setup_environment(None)
    plan = generate_plan_from_major_and_year(
        major=major,
        year_level=year_level
    )
    return jsonify({"plan": plan})


if __name__ == "__main__":
    # For local testing
    # Make sure OPENAI_API_KEY and LANGCHAIN_API_KEY are set in your environment first.
    app.run(host="0.0.0.0", port=5000, debug=True)
