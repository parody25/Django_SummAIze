import os
import uuid
import json
from docx import Document
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from concurrent.futures import ThreadPoolExecutor
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from .models import Application, PDFDocument
# from nemoguardrails import LLMRails, RailsConfig
import datetime
from django.http import JsonResponse
from docx import Document
from playwright.async_api import async_playwright
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import tempfile
from urllib.parse import urljoin

from .helpers import validate_pdf, process_question
from .helpers import parse_document
from .helpers import create_credit_application, get_credit_application, update_credit_application_section
from .helpers import generate_doc_response

#from llama_index.prompts import PromptTemplate as PT
from llama_index.core.schema import ImageDocument, TextNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from typing import List
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaOpenAIEmbedding
#from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import (
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)

from .scraperMod2 import Website, summarize_text
from .scraperModule import CompanyInfoScraper
from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import shutil
from .safety import is_text_safe
import hashlib
from django.core.cache import cache
from .RedisCacheService import RedisCacheService
from .ContentSafetyService import ContentSafetyService
from django.conf import settings
from .FallBackService import FallBackService

IMAGES_DOWNLOAD_PATH = os.path.join(os.path.dirname(__file__), 'constant', 'images')
#RISK_ANALYSIS_REPORT = os.path.join(os.path.dirname(__file__), 'constant', 'CBD_PoC_Credit_Bureau_Report.pdf')
#CRM_DOC_PATH = os.path.join(os.path.dirname(__file__), 'constant', 'CBD PoC_CRM.docx')
RISK_ANALYSIS_REPORT = os.path.join(os.path.dirname(__file__), 'constant', 'Lincoln Electric_Credit Score Report.pdf')
CRM_DOC_PATH = os.path.join(os.path.dirname(__file__), 'constant', 'Lincoln Electric_CRM.docx')
CRM_DATA_CACHE = None
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

open_API=os.getenv('OPENAI_API_KEY')
llama_cloud_api = os.getenv('LLAMA_CLOUD_API')
#debug_handler = LlamaDebugHandler()
#callback_manager = CallbackManager(handlers=[debug_handler])


# Initialize embeddings
text_embeddings = OpenAIEmbeddings()


# Advanced Embeddings - LlamaIndex
embed_model = LlamaOpenAIEmbedding(model="text-embedding-3-small")
llm = LlamaOpenAI(model="gpt-4o", api_key=open_API)
Settings.llm = llm
Settings.embed_model = embed_model
safety_service = ContentSafetyService(
    endpoint=settings.AZURE_CONTENT_SAFETY_ENDPOINT,
    key=settings.AZURE_CONTENT_SAFETY_KEY
)

@csrf_exempt
def separate_embedding_upload_pdfs(request):
    if request.method == 'POST':
        application_id = str(uuid.uuid4())
        create_credit_application(application_id)
        print('Process started for Application id ', application_id)
        
        uploaded_files = request.FILES.getlist('pdfs')
        application, created = Application.objects.get_or_create(application_id=application_id)

        valid_pdfs = []
        invalid_pdfs = []

        valid_pdfs, invalid_pdfs = validate_pdf(uploaded_files)
        valid_pdfs = [file_tuple[0] for file_tuple in valid_pdfs]

        if invalid_pdfs:
            print(f"Invalid files detected: {invalid_pdfs}")

        if not valid_pdfs:
            return JsonResponse({"error": "No valid files found"}, status=400)

        print(f"{len(valid_pdfs)} files validated successfully.")

        document_embeddings_info = []
        for file in valid_pdfs:
            file.seek(0)
            file_name = os.path.basename(file.name)
            document_name, file_extension = os.path.splitext(file_name)
            document_id = f"{application_id}_{document_name}"

            document_info = {
                "document_name": file_name,
                "embedding_file": f"{document_id}_embeddings.pkl",
                "document_id": document_id
            }

            if file_extension == '.pdf' or '.docx' or '.xlsx':
                try:
                    file_name = os.path.basename(file_name)
                    #index_path = os.path.join(INDEX_DIR,file_name)
                    app_embedding_dir = f"embeddings/{application_id}"
                    os.makedirs(app_embedding_dir, exist_ok=True)
                    embeddings_file = f"{app_embedding_dir}/{file_name}_embedding.pkl"
                    if os.path.exists(embeddings_file):
                        print("Embedding already present{embeddings_file}")
                    else:
                        global_embedding_file = f"embeddings/global/{file_name}_embedding.pkl"
                        if os.path.exists(global_embedding_file):
                            shutil.copytree(global_embedding_file, embeddings_file)
                            print(f"Embedding for this file {file_name} is already available, copying the same")
                        else:
                            print(f"No existing index found. Creating new index for {file_name}")
                            llama_parser = LlamaParse(result_type="markdown", api_key=llama_cloud_api)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                                temp_file.write(file.read())
                                temp_file_path = temp_file.name

                            documents = llama_parser.load_data(temp_file_path)
                            node_parser = MarkdownElementNodeParser(llm=llm,num_workers=15)
                            nodes = node_parser.get_nodes_from_documents(documents)
                            base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
                            index = VectorStoreIndex(base_nodes+objects)
                            index.storage_context.persist(embeddings_file)
                            shutil.copy(embeddings_file, global_embedding_file)
                            print(f"New embedding created and saved globally at {global_embedding_file}")
                        pdf_doc = PDFDocument.objects.create(
                            application=application,
                            pdf_name=file_name,
                            time_uploaded=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                        )
                        document_info["pdf_document_id"] = pdf_doc.id
                        document_embeddings_info.append(document_info)
                        print(f"Saved embeddings for {file_name} at {embeddings_file}")
                except Exception as e:
                    print(f"Failed to create embeddings for {file_name}: {str(e)}")
                    continue
        if document_embeddings_info:
            # print(collateral_qa_population(application_id))
            # print(management_qa_population(application_id))
            return JsonResponse({
                "status": "success",
                "message": "Documents processed successfully",
                "application_id": application_id,
                "documents": document_embeddings_info
            }, status=200)
        else:
            return JsonResponse({"error": "Failed to create embeddings for any documents"}, status=400)
        
    return JsonResponse({"error": "Invalid request method"}, status=405)
                                                  

def load_crm_data():
    """Load and parse the CRM document once at startup"""
    global CRM_DATA_CACHE
    if CRM_DATA_CACHE is None:
        try:
            doc = Document(CRM_DOC_PATH)
            CRM_DATA_CACHE = parse_document(doc)
            print("CRM data loaded successfully")
        except Exception as e:
            print(f"Error loading CRM data: {str(e)}")
            CRM_DATA_CACHE = {}
    return CRM_DATA_CACHE

@csrf_exempt
def get_crm_value(request):
    if request.method == 'GET':
        try:
            application_id = request.GET.get('application_id', '')
            if not application_id:
                return JsonResponse({"status": "error", "message": "Missing application_id"}, status=400)
        
            crm_data = load_crm_data()
            
            response_data = {
                "borrower_name": crm_data.get("Account Information", {}).get("Account Name", ""),
                "date_of_application": crm_data.get("Account Information", {}).get("Last modified date", ""),
                "type_of_business": crm_data.get("Account Information", {}).get("Industry", ""),
                "borrower_risk_rating": crm_data.get("Account Information", {}).get("Rating", ""),
                "new_or_existing": crm_data.get("Account Information", {}).get("Customer Type", ""),
                "naics_code": crm_data.get("Account Information", {}).get("SIC Code", ""),
                "borrower_address": crm_data.get("Account Information", {}).get("Address", ""),
                "telephone": crm_data.get("Account Information", {}).get("Phone Number", ""),
                "email_address": crm_data.get("Account Information", {}).get("Email", ""),
                "fax_number": crm_data.get("Account Information", {}).get("Fax", ""),
                "branch_number": crm_data.get("Account Information", {}).get("Branch Number", ""),
                "account_number": crm_data.get("Account Information", {}).get("Account Number", ""),
                "related_borrowings": crm_data.get("Related Borrowings", [])
            }

            # CREDIT_APPLICATION_STORE[application_id].borrower_details = BorrowerDetails.model_validate(response_data)

            # update_application(application_id, "borrower_details", response_data)
            
            # updated_borrower_details = CREDIT_APPLICATION_STORE[application_id].borrower_details.model_dump()

            try:
                update_credit_application_section(application_id, "borrower_details", response_data)
            except ValueError:
                print("Credit application not found against application id")
            except AttributeError:
                print("Section not found")




            
            return JsonResponse({
                "status": "success",
                "data": response_data
            }, status=200)
            
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=500)
    
    return JsonResponse({
        "status": "error",
        "message": "Only GET requests are allowed"
    }, status=405)


@csrf_exempt
def get_section(request):
    if request.method != 'POST':
        return JsonResponse({
            "status" : "error",
            "message" : f"Invalid request method",
            "data": None}, status=405)
    try:
        data = json.loads(request.body)
        application_id = data.get("application_id")
        section_name = data.get("section")

        if not application_id or not section_name:
            return JsonResponse({
            "status" : "error",
            "message" : f"Missing required parameters: application_id and section",
            "data": None}, status=400)

        # Retrieve the CreditApplication instance from the global store
        credit_application = get_credit_application(application_id)
        if credit_application is None:
            return JsonResponse({
            "status" : "error",
            "message" : f"CreditApplication instance not found for the given application_id",
            "data": None}, status=404)
            
        
        # Check if the requested section exists within the CreditApplication
        if not hasattr(credit_application, section_name):
            return JsonResponse({
            "status" : "error",
            "message" : f"Section '{section_name}' not found in CreditApplication",
            "data": None}, status=400)
            

        # Extract the section's Pydantic model
        section_object = getattr(credit_application, section_name)
        # Convert the Pydantic model to a dictionary (with default values)
        section_data = section_object.model_dump()

        # populated_section = populate_section(application_id, section_name, section_data)


        return JsonResponse({
            "status" : "success",
            "message" : f"{section_name} retrieved succesfully",
            "data": section_data,
            "application_id": application_id}, status=200)
    
    
    
    except json.JSONDecodeError:
        return JsonResponse({
            "status" : "error",
            "message" : "Invalid JSON format",
            "data": None}, status=400)
    except Exception as e:
        return JsonResponse({
            "status" : "error",
            "message" : "Error",
            "data": None}, status=500)

@csrf_exempt
def web_scrapping(request):
    if request.method == 'POST':
        try:
            # Get parameters from request
            url = request.GET.get('url')
            application_id = request.GET.get('application_id')
            
            if not url or not application_id:
                return JsonResponse({"error": "Both url and application_id are required"}, status=400)

            print(f'User Input URL: {url}')
            print(f'Application ID: {application_id}')

            # Initialize CREDIT_APPLICATION_STORE if not exists
            # if application_id not in CREDIT_APPLICATION_STORE:
            #     CREDIT_APPLICATION_STORE[application_id] = CreditApplication()

            # Caching logic
            url_hash = hashlib.md5(url.strip().lower().encode()).hexdigest()
            cache_key = f"web_scrapping:{url_hash}"
            cached_summary = cache.get(cache_key)

            if cached_summary:
                print(f"[CACHE HIT] URL Hash: {url_hash}")
                section_data = {
                    "borrower_profile": cached_summary
                }
            #     update_credit_application_section(application_id, "borrower_history_and_background", section_data)

                return JsonResponse({
                    "status": "cached",
                    "summary": cached_summary,
                    "application_id": application_id
                })

            scrapper = CompanyInfoScraper()
            translatedData = scrapper.scrape_company_info(start_url = url) 
            arabic_text = ""
            full_text = ""
            for page in translatedData:
                print("\n" + "=" * 100)
                print(f"Context: Scraping company page for: {page.get('url')}")
                # print("\nSUMMARY (Paragraphs):")
                for para in page.get("summarized_content", []):
                    arabic_text += para + ' '
                    # print(f"- {para}")
                # print("\nTRANSLATED CONTENT (Paragraphs):")
                for para in page.get("translated_content", []):
                    full_text += para + ' '
                    # print(f"- {full_text}")

            print("\n ARABIC SUMMARY (Paragraphs):")
            print(arabic_text)
            print("\nTRANSLATED CONTENT (Paragraphs):")
            print(full_text)

                # full_text = ' '.join([text.get_text(strip=True) for text in soup.find_all(['p', 'h1', 'h2', 'h3'])])
                
                # Generate summary
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes company information always in english language as output."},
                    {"role": "user", "content": f"""
                    Create a detailed company profile (200 words) covering:
                    1. Company name
                    2. Industry
                    3. Headquarters and locations
                    4. Products/services
                    5. Founding year
                    6. Company size (revenue, employees, subsidiaries)
                    
                    Text to analyze: {full_text}
                    """}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            summary = completion.choices[0].message.content.strip()
            cache.set(cache_key, summary, timeout=10000)
            print(f"[CACHE SET IN TRY] URL Hash: {url_hash}")            
            #Store result
            #CREDIT_APPLICATION_STORE[application_id].borrower_history_and_background.borrower_profile = summary
            section_data = {
                "borrower_profile" : summary
            }

            update_credit_application_section(application_id, "borrower_history_and_background", section_data)

            return JsonResponse({
                "status": "success",
                "summary": summary,
                "application_id": application_id
            })
                # except:
                #     return JsonResponse({"error": f"Failed to process content: {str(e)}"}, status=500)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Only POST requests allowed"}, status=405)

@csrf_exempt
def edit_section(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Invalid request method"}, status=405)
    try:
        data = json.loads(request.body)
        application_id = data.get("application_id")
        section_name = data.get("section")
        section_data = data.get("section_data")

        if not application_id or not section_name or not section_data:
            return JsonResponse(
                {"error": "Missing required parameters: application_id and section data"},
                status=400
            )
        
        try:
            update_credit_application_section(application_id, section_name, section_data)

            return JsonResponse({"status": "success", "message": f"Section '{section_name}' updated successfully"}, status=200)
        
        except:
            return JsonResponse({"error": "Invalid data or application_id"}, status=404)
    
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

COLLATERAL_QUESTIONS = [
"Create a concise summary paragraph from the real estate valuation report with details on property name and address, city, property description and type, neighborhood, assessed value and marketing time estimated"
]

@csrf_exempt
def collateral_qa(request):
    if request.method == 'POST':
        try:
            # Parse request data
            data = json.loads(request.body)
            application_id = data.get('application_id')
            
            if not application_id:
                return JsonResponse({"error": "application_id is required"}, status=400)

            embedding_dir = os.path.join("embeddings", application_id)
            if not os.path.exists(embedding_dir):
                return JsonResponse({"error": "Application embeddings directory not found"}, status=404)

            appraisal_files = [f for f in os.listdir(embedding_dir) 
                if 'appraisal' in f.lower() and f.endswith('.pkl')
            ]
            if not appraisal_files:
                return JsonResponse({"error": "No appraisal report embeddings found"}, status=404)
            
            embedding_file = appraisal_files[0]
            embedding_path = os.path.join(embedding_dir, embedding_file)

            if not os.path.exists(embedding_path):
                return JsonResponse({"error": "Collateral embeddings not found"}, status=404)
                
            #vectorstore = FAISS.load_local(
            #    embedding_path,
             #   embeddings=text_embeddings,
             #   allow_dangerous_deserialization=True
            #)
            if os.path.exists(embedding_path):
                print(f"Loading existing Faiss index: {embedding_path}")
                #vector_store = FaissVectorStore.from_persist_dir(embedding_path)
                #storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=embedding_path)
                storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
                vector_db = load_index_from_storage(storage_context=storage_context)
            
            print(f"Vectorstore loaded for the application ID {embedding_path}")

            llm = ChatOpenAI(model="gpt-4o", temperature=0)

            question_response_map = {}

            with ThreadPoolExecutor() as executor:
                responses = executor.map(lambda q: process_question(q, vector_db, llm), COLLATERAL_QUESTIONS)
            for question, response in responses:
                question_response_map[question] = response.response

            print("Final question-response map:", question_response_map)

            final_response = question_response_map[COLLATERAL_QUESTIONS[0]]

            section_data = {
                "real_estate_security" : final_response  
            }

            try:
                update_credit_application_section(application_id, "security", section_data)
            except:
                print(f"Unable to fetch credit_application object for this application id")
            

            return JsonResponse({
                "status": "success",
                "application_id": application_id,
                "data": final_response
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Only POST requests allowed"}, status=405)



@csrf_exempt
def management_qa(request):
    if request.method == 'POST':
        try:
            # Parse request data
            data = json.loads(request.body)
            application_id = data.get('application_id')
            
            if not application_id:
                return JsonResponse({"error": "application_id is required"}, status=400)

            embedding_dir = os.path.join("embeddings", application_id)
            if not os.path.exists(embedding_dir):
                return JsonResponse({"error": "Application embeddings directory not found"}, status=404)

            appraisal_files = [f for f in os.listdir(embedding_dir) 
                if 'annual' in f.lower() and f.endswith('.pkl')
            ]
            if not appraisal_files:
                return JsonResponse({"error": "No annual report embeddings found"}, status=404)
            
            embedding_file = appraisal_files[0]
            embedding_path = os.path.join(embedding_dir, embedding_file)

            if not os.path.exists(embedding_path):
                return JsonResponse({"error": "Annual report embeddings not found"}, status=404)
                
            #vectorstore = FAISS.load_local(
            #    embedding_path,
             #   embeddings=text_embeddings,
             #   allow_dangerous_deserialization=True
            #)
            if os.path.exists(embedding_path):
                print(f"Loading existing Faiss index: {embedding_path}")
                #vector_store = FaissVectorStore.from_persist_dir(embedding_path)
                #storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=embedding_path)
                storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
                vector_db = load_index_from_storage(storage_context=storage_context)
            
            print(f"Vectorstore loaded for the application ID {embedding_path}")

            #llm = ChatOpenAI(model="gpt-4o", temperature=0)

            question_response_map = {}
            with open("./prompts/Management_Analysis_prompt.txt", "r") as file:
                prompt_Management_Analysis = file.read()
            retriever = vector_db.as_retriever(similarity_top_k=30)
            retrievied_docs = retriever.retrieve('''Retrieve:

The full Corporate Governance or Corporate Governance Report section.

The Board of Directors list or table, including any related metadata such as tenure, roles, and committee assignments.''')
            corporate_governance_data = [node.text for node in retrievied_docs]

            prompt_text = prompt_Management_Analysis.format(
        corporate_governance_data=corporate_governance_data)
            
            response = llm.complete(prompt_text)

            #query_engine = vector_db.as_query_engine(similariy_top_k=25)
            #response = query_engine.query(Management_Analysis)

            #with ThreadPoolExecutor() as executor:
             #   responses = executor.map(lambda q: process_question(q, vector_db, llm), MANAGEMENT_QUESTIONS)
            #for question, response in responses:
             #   question_response_map[question] = response.response

            #print("Final question-response map:", question_response_map)

            final_response = response.text

            section_data = {
                "board_of_directors_profile" : final_response
            }

            try:
                update_credit_application_section(application_id, "management_analysis", section_data)
            except:
                print(f"Unable to fetch credit_application object for this application id")

            return JsonResponse({
                "status": "success",
                "application_id": application_id,
                "data": final_response
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Only POST requests allowed"}, status=405)

def collateral_qa_population(application_id):
    embedding_dir = os.path.join("embeddings", application_id)
    if not os.path.exists(embedding_dir):
        return JsonResponse({"error": "Application embeddings directory not found"}, status=404)

    appraisal_files = [f for f in os.listdir(embedding_dir) 
        if 'appraisal' in f.lower() and f.endswith('.pkl')
    ]
    if not appraisal_files:
        return JsonResponse({"error": "No appraisal report embeddings found"}, status=404)
    
    embedding_file = appraisal_files[0]
    embedding_path = os.path.join(embedding_dir, embedding_file)

    if not os.path.exists(embedding_path):
        return JsonResponse({"error": "Collateral embeddings not found"}, status=404)
        
    
    if os.path.exists(embedding_path):
        print(f"Loading existing Faiss index: {embedding_path}")
        storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
        vector_db = load_index_from_storage(storage_context=storage_context)
    
    print(f"Vectorstore loaded for the application ID {embedding_path}")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    question_response_map = {}

    with ThreadPoolExecutor() as executor:
        responses = executor.map(lambda q: process_question(q, vector_db, llm), COLLATERAL_QUESTIONS)

        
    for question, response in responses:
        question_response_map[question] = response.response


    print("Final question-response map:", question_response_map)

    section_data = {
        "real_estate_security" : question_response_map[COLLATERAL_QUESTIONS[0]]
    }

    try:
        update_credit_application_section(application_id, "security", section_data)
        # CREDIT_APPLICATION_STORE[application_id].security.real_estate_security = question_response_map[COLLATERAL_QUESTIONS[0]]
    except KeyError as e:
        print(f"❌ KeyError: {e} - Invalid key in COLLATERAL_QUESTIONS or question_response_map")

    return JsonResponse({
        "status": "success",
        "application_id": application_id,
        "collateral_details": question_response_map
    })

MANAGEMENT_QUESTIONS = [
    '''You are an AI Assistant that summarized data to create a brief profile of the key management people from the annual report of a company.
Step 1: Refer to the section on Corporate Governance/Board of Directors/Management Analysis in the Annual Report uploaded
Step 2: For each person’s description in the Board of Directors, select information on:
Name
Position/designation in company
Responsibilities handled in current position
Highest educational qualification
Number of years or work experience and previous work experience
Business acumen and reputation
Step 3: Write a concise and coherent summary of each person in the board of directors that covers information identified in Step 2. Each profile must be a paragraph of not more than 4-5 lines. ''']

def management_qa_population(application_id):
    embedding_dir = os.path.join("embeddings", application_id)
    if not os.path.exists(embedding_dir):
        return JsonResponse({"error": "Application embeddings directory not found"}, status=404)

    appraisal_files = [f for f in os.listdir(embedding_dir) 
        if 'annual' in f.lower() and f.endswith('.pkl')
    ]
    if not appraisal_files:
        return JsonResponse({"error": "No annual report embeddings found"}, status=404)
    
    embedding_file = appraisal_files[0]
    embedding_path = os.path.join(embedding_dir, embedding_file)

    if not os.path.exists(embedding_path):
        return JsonResponse({"error": "Annual report embeddings not found"}, status=404)
        
    
    if os.path.exists(embedding_path):
        print(f"Loading existing Faiss index: {embedding_path}")
        storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
        vector_db = load_index_from_storage(storage_context=storage_context)
    
    print(f"Vectorstore loaded for the application ID {embedding_path}")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    question_response_map = {}

    with ThreadPoolExecutor() as executor:
        responses = executor.map(lambda q: process_question(q, vector_db, llm), MANAGEMENT_QUESTIONS)

        
    for question, response in responses:
        question_response_map[question] = response.response


    print("Final question-response map:", question_response_map)

    section_data = {
        "board_of_directors_profile" : question_response_map[MANAGEMENT_QUESTIONS[0]]
    }

    try:
        update_credit_application_section(application_id, "management_analysis", section_data)
        # CREDIT_APPLICATION_STORE[application_id].management_analysis.board_of_directors_profile = question_response_map[MANAGEMENT_QUESTIONS[0]]
    except KeyError as e:
        print(f"❌ KeyError: {e} - Invalid key in MANAGEMENT_QUESTIONS or question_response_map")

    return JsonResponse({
        "status": "success",
        "application_id": application_id,
        "management_analysis_details": question_response_map
    })

@csrf_exempt
def collateral_chat(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            application_id = data.get('application_id')
            question = data.get('question')
            
            if not application_id or not question:
                return JsonResponse({"error": "application_id and question are required"}, status=400)

            safety_result = safety_service.is_text_safe(question)
            if not safety_result.get("safe"):
                return JsonResponse({
                    "error": "Question violates content policy",
                    "violations": safety_result.get("details"),
                    "status": "blocked"
                }, status=400)
            # Check cache
            cache_key = RedisCacheService.generate_cache_key(
                "collateral_chat", 
                application_id, 
                question.strip().lower()
            )
            cached_answer = RedisCacheService.get_cached_data(cache_key)
            
            if cached_answer:
                print(f"[CACHE HIT] AppID: {application_id}")
                return JsonResponse({
                    "question": question,
                    "answer": cached_answer,
                    "status": "cached",
                    "application_id": application_id,
                    "safety_checks": {
                        "input": safety_result,
                        "output": is_text_safe(cached_answer)
                    }
                })
            # # CACHING: Generate a unique key for this question+app
            # hash_q = hashlib.md5(question.strip().lower().encode()).hexdigest()
            # cache_key = f"collateral_chat:{application_id}:{hash_q}"
            # cached_answer = cache.get(cache_key)
            # if cached_answer:
            #     print(f"[CACHE HIT] AppID: {application_id}, QuestionHash: {hash_q}")
            #     return JsonResponse({
            #         "question": question,
            #         "answer": cached_answer,
            #         "status": "cached",
            #         "application_id": application_id,
            #         "safety_checks": {
            #             "input": safety_result,
            #             "output": is_text_safe(cached_answer)
            #         }
            #     })

            embedding_dir = os.path.join("embeddings", application_id)
            if not os.path.exists(embedding_dir):
                return JsonResponse({"error": "Application embeddings directory not found"}, status=404)

            appraisal_files = [f for f in os.listdir(embedding_dir) 
                if 'appraisal' in f.lower() and f.endswith('.pkl')
            ]
            if not appraisal_files:
                return JsonResponse({"error": "No appraisal report embeddings found"}, status=404)
            
            embedding_file = appraisal_files[0]
            embedding_path = os.path.join(embedding_dir, embedding_file)
            
            if not os.path.exists(embedding_path):
                return JsonResponse({"error": "Collateral embeddings not found"}, status=404)           
            if os.path.exists(embedding_path):
                print(f"Loading existing Faiss index: {embedding_path}")
                storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
                vector_db = load_index_from_storage(storage_context=storage_context)
            
            print(f"Vectorstore loaded for the application ID {embedding_path}")
            fallback = FallBackService()

            # 3. Process the single question using the same process_question function
            # llm = ChatOpenAI(model="gpt-4o", temperature=0)
            # _, response = process_question(question, vector_db, llm)
            # answer = response.response if hasattr(response, 'response') else response
            
            def generate_response():
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                _, response = process_question(question, vector_db, llm)
                return response.response if hasattr(response, 'response') else response

            try:
                answer = fallback.execute(generate_response)
            except Exception as e:
                return JsonResponse({"error": "LLM processing failed after retries", "details": str(e)}, status=500)
            
            output_safety = safety_service.is_text_safe(answer)
            if not output_safety.get("safe", False):
                answer = "Response redacted due to policy violation."

            # CACHING: Save the answer in Redis for 10 minutes
            RedisCacheService.set_cached_data(cache_key, answer, timeout=600)
            print(f"[CACHE SET] AppID: {application_id}")
            #cache.set(cache_key, answer, timeout=600)
            #print(f"[CACHE SET] AppID: {application_id}, QuestionHash: {hash_q}")
            
            return JsonResponse({
                "question": question,
                "answer": answer,
                "status": "success",
                "application_id": application_id,
                "safety_checks": {
                    "input": safety_result,
                    "output": output_safety
                }
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Only POST requests allowed"}, status=405)

VECTOR_DB_CACHE = {}
CHAT_ENGINE_CACHE = {}
memory = ChatMemoryBuffer.from_defaults(token_limit=10000)
@csrf_exempt
def management_chat(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            application_id = data.get('application_id')
            question = data.get('question')
            if not application_id or not question:
                return JsonResponse({"error": "application_id and question are required"}, status=400)
            if application_id in VECTOR_DB_CACHE:
                vector_db = VECTOR_DB_CACHE[application_id]
                chat_engine = CHAT_ENGINE_CACHE[application_id]
                print(f"Using cached vector DB for {application_id}")
 
            else:
 
                embedding_dir = os.path.join("embeddings", application_id)
                if not os.path.exists(embedding_dir):
                    return JsonResponse({"error": "Application embeddings directory not found"}, status=404)
 
                management_files = [f for f in os.listdir(embedding_dir) 
                    if 'annual' in f.lower() and f.endswith('.pkl')
                ]
                if not management_files:
                    return JsonResponse({"error": "No annual report embeddings found"}, status=404)
                embedding_file = management_files[0]
                embedding_path = os.path.join(embedding_dir, embedding_file)
                if not os.path.exists(embedding_path):
                    return JsonResponse({"error": "Annual Report embeddings not found"}, status=404)           
                if os.path.exists(embedding_path):
                    print(f"Loading existing Faiss index: {embedding_path}")
                    storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
                    vector_db = load_index_from_storage(storage_context=storage_context)
                    VECTOR_DB_CACHE[application_id] = vector_db
                    chat_engine = vector_db.as_chat_engine(similarity_top_k=25,
                    chat_mode="context",
                    memory=memory,
                    system_prompt=(
                    "You are an AI assistant who answers the user questions based on the context provided from Annual Report"
                    ),
                    )
                    CHAT_ENGINE_CACHE[application_id] = chat_engine
 
            
                print(f"Vectorstore loaded Newly and NewChat Engine for the application ID {embedding_path}")

 
            # 3. Process the single question using the same process_question function
            #llm = ChatOpenAI(model="gpt-4o", temperature=0)
            #chat_engine = vector_db.as_chat_engine(similarity_top_k=25,
            #chat_mode="context",
            #memory=memory,
            #system_prompt=(
            #"You are an AI assistant who answers the user questions based on the context provided from Annual Report"
            #),
        #)
            response = chat_engine.chat(question)
            #_, response = process_question(question, vector_db, llm)
 
            return JsonResponse({
                "question": question,
                "answer": response.response if hasattr(response, 'response') else response,
                "status": "success",
                "application_id": application_id
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Only POST requests allowed"}, status=405)

TEMPLATE_PATH = f"assets\\placeholder_template.docx"
DOCS_DIR = f"docs"

@csrf_exempt
def download_doc(request):
    application_id = request.GET.get("application_id")
    date = request.GET.get("date")
    credit_application_name = request.GET.get("credit_application_name")
    if not application_id:
        return JsonResponse({"error": "application_id is required"}, status=400)
    
    update_credit_application_section(application_id, "date", date)
    update_credit_application_section(application_id, "credit_application_name", credit_application_name)
    try:
        # Generate document from the credit_application object
        doc_path = generate_doc_response(application_id)
        return FileResponse(open(doc_path, 'rb'), as_attachment=True)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    

@csrf_exempt
def get_financial_ratios(request):
    if request.method != 'GET':
        return JsonResponse({"error": "Only GET requests allowed"}, status=405)
    
    application_id = request.GET.get("application_id")
    if not application_id:
        return JsonResponse({"error": "application_id is required"}, status=400)

    with open("constant\\financial_ratios.json", "r") as f:
        financial_ratios = json.load(f)
    return JsonResponse({
            "status" : "success",
            "application_id" : application_id,
            "data": financial_ratios}, status=200)

def load_ratio_config():
    """Load ratio configuration from JSON file"""
    try:
        with open("constant/ratio_config_rulengine.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise Exception("Ratio configuration file not found")
    except json.JSONDecodeError:
        raise Exception("Invalid ratio configuration format")

def check_breach(value, threshold, rule_type):
    """Check if a value breaches its threshold"""
    if rule_type == "minimum":
        return value < threshold
    elif rule_type == "maximum":
        return value > threshold
    return False

def generate_analysis(ratio_name, year, value, threshold, ratio_config):
    """Generate analysis only for breached ratios"""
    config = ratio_config.get(ratio_name, {})
    rule_type = config.get("rule_type")
    
    if not rule_type:
        return None
    
    is_breached = check_breach(value, threshold, rule_type)
    
    if not is_breached:
        return None
    
    action = "fallen below" if rule_type == "minimum" else "exceeded"
    
    return {
        "ratio": ratio_name,
        "year": year,
        "value": value,
        "threshold": threshold,
        "breach_message": f"The {ratio_name} has {action} the threshold of {threshold} in {year}",
        "questions": config.get("questions", []),
        "rule_type": rule_type
    }

@csrf_exempt
def financial_ratio_rule_engine(request):
    if request.method != 'GET':
        return JsonResponse({"error": "Only GET requests allowed"}, status=405)
    
    application_id = request.GET.get("application_id")
    if not application_id:
        return JsonResponse({"error": "application_id is required"}, status=400)

    try:
        # Load both data files
        with open("constant/financial_ratios.json", "r") as f:
            financial_ratios = json.load(f)
        
        ratio_config = load_ratio_config()
        
        # Process only breached ratios
        breached_ratios = []
        
        for ratio_name, yearly_data in financial_ratios.items():
            for data_point in yearly_data:
                analysis = generate_analysis(
                    ratio_name=ratio_name,
                    year=data_point["year"],
                    value=data_point["value"],
                    threshold=data_point["threshold"],
                    ratio_config=ratio_config
                )
                if analysis:  # Only include if analysis exists (breach occurred)
                    breached_ratios.append(analysis)
        
        return JsonResponse({
            "status": "success",
            "application_id": application_id,
            "threshold_analysis": {
                "breaches_found": len(breached_ratios) > 0,
                "total_breaches": len(breached_ratios),
                "breached_ratios": breached_ratios
            }
        }, status=200)
    
    except Exception as e:
        return JsonResponse({
            "error": str(e),
            "type": type(e).__name__
        }, status=500)

@csrf_exempt
def get_financial_analysis(request):
    if request.method != 'GET':
        return JsonResponse({"error": "Only GET requests allowed"}, status=405)
    
    application_id = request.GET.get("application_id")
    if not application_id:
        return JsonResponse({"error": "application_id is required"}, status=400)
    
    with open("constant\\financial_ratios.json", "r") as f:
        financial_ratios = json.load(f)

    with open("./prompts/risk_ratio_prompts.txt", "r") as file:
        template = file.read()

    # Retrieving the Financial statements from the Annual Report 
    embedding_dir = os.path.join("embeddings", application_id)
    if not os.path.exists(embedding_dir):
        return JsonResponse({"error": "Application embeddings directory not found"}, status=404)
    
    appraisal_files = [f for f in os.listdir(embedding_dir) 
                if 'annual' in f.lower() and f.endswith('.pkl')
            ]
    if not appraisal_files:
                return JsonResponse({"error": "No annual report embeddings found"}, status=404)
    embedding_file = appraisal_files[0]
    embedding_path = os.path.join(embedding_dir, embedding_file)
    
    if not os.path.exists(embedding_path):
        return JsonResponse({"error": "Annual report embeddings not found"}, status=404)
    
    if os.path.exists(embedding_path):
        print(f"Loading existing Faiss index: {embedding_path}")
        storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
        vector_index_db = load_index_from_storage(storage_context=storage_context)

    print(f"Vectorstore loaded for the application ID {embedding_path}")

    retriever = vector_index_db.as_retriever(similarity_top_k=30)

    #For Retrieving the Financial Statments Tabular Data
    with open("./prompts/Financial_statements_retrieveing_prompts.txt", "r") as file:
                prompt_template1 = file.read()

    retrievied_docs = retriever.retrieve(prompt_template1)
    financial_Tabular_data = [node.text for node in retrievied_docs]

    #For Retrieving the Detailed Disclosures of Consolidated Financial Statements
    with open("./prompts/Financial_statements_detail_disclosure_prompts.txt", "r") as file:
                prompt_template1 = file.read()
    retrievied_docs = retriever.retrieve(prompt_template1)
    financial_disclosure_data = [node.text for node in retrievied_docs]
    
    #Financial Ratio's 
    with open("constant\\financial_ratios.json", "r") as f:
        financial_ratios = json.load(f)

    #Financial Ratio's - Formula 
    with open("constant\\financial_ratio_formula.json", "r") as f:
        financial_formula = json.load(f)

    # Generating the Commentary 
    with open("./prompts/Financial_commentary.txt", "r") as file:
                prompt_template2 = file.read()
    #prompt_template = PromptTemplate(prompt_template2)

    documents = {
        "financial_statement_data": financial_Tabular_data,
        "notes_disclosures": financial_disclosure_data,
        "financial_formulas": financial_formula
    }

    prompt_text = prompt_template2.format(
        financial_statement_data=documents["financial_statement_data"],
        notes_disclosures=documents["notes_disclosures"],
        financial_formulas=documents["financial_formulas"]
    )
    
    #Final Response
    response = llm.complete(prompt_text)

    #QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    #llm = ChatOpenAI(model="gpt-4o",api_key=open_API,temperature=0)
    #chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)
    #response = chain.run({"Risk_Ratio": financial_ratios})

    financial_analysis = response.text if hasattr(response, 'text') else response

    section_data = {
        "ratios" : "",
        "analysis" : financial_analysis
    }

    update_credit_application_section(application_id, "financial_analysis", section_data)

    return JsonResponse({
            "status" : "success",
            "application_id" : application_id,
            "data": financial_analysis}, status=200)


#Risk Report Analysis 
@csrf_exempt
def risk_analysis(request):
    if request.method == 'GET':
        try:
            application_id = request.GET.get('application_id', '')
            if not application_id:
                return JsonResponse({"status": "error", "message": "Missing application_id"}, status=400)
            
            # Processing the Risk Analysis PDF 
            #RISK_ANALYSIS_REPORT
            parser = LlamaParse(api_key=llama_cloud_api,result_type="markdown",)
            json_objs = parser.get_json_result(RISK_ANALYSIS_REPORT)
            json_list = json_objs[0]["pages"]
            text_nodes = [TextNode(text=page["text"], metadata={"page": page["page"]}) for page in json_list]
            # Processing the Image Separately from the Nodes
            #with tempfile.TemporaryDirectory() as temp_dir:
            #print(temp_dir)
            image_dicts = parser.get_images(json_objs, download_path=IMAGES_DOWNLOAD_PATH)
            image_documents = [ImageDocument(image_path=image_dict["path"]) for image_dict in image_dicts]
            openai_mm_llm = OpenAIMultiModal(model="gpt-4o", api_key=open_API,max_new_tokens=1000)
            with open("./prompts/risk_report_prompts.txt", "r") as file:
                template = file.read()
            financial_context = [node.text for node in text_nodes]
            prompt = template.format(report_context=financial_context)
            print(prompt)
            response = openai_mm_llm.complete(
                   prompt=prompt,
                image_documents=image_documents)
            
            risk_rating_response = response.text

            try:
                update_credit_application_section(application_id, "risk_analysis", {
                    "risk_rating" : risk_rating_response
                })
            except:
                print("Unable to write section to credit application db")
            return JsonResponse({
                "status": "success",
                "data": risk_rating_response
            }, status=200)
        
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=500)
    return JsonResponse({
        "status": "error",
        "message": "Only GET requests are allowed"
    }, status=405)


@csrf_exempt
def get_justification(request):
    if request.method != 'GET':
        return JsonResponse({"error": "Only GET requests allowed"}, status=405)
    
    application_id = request.GET.get("application_id")
    if not application_id:
        return JsonResponse({"error": "application_id is required"}, status=400)

    try:
        credit_application = get_credit_application(application_id)
    except:
        return JsonResponse({
            "status" : "error",
            "message" : "Credit application object not found in db",
            "data" : None
        }, status=400)
    
    collateral_data = credit_application.security.real_estate_security
    financial_analysis_data = credit_application.financial_analysis.analysis
    risk_score_data = credit_application.risk_analysis.risk_rating
    

    with open("./prompts/justification_prompts.txt", "r") as file:
        template = file.read()

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o",api_key=open_API,temperature=0)
    chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)
    response = chain.run({
        "risk_score_data": risk_score_data,
        "collateral_data" : collateral_data,
        "financial_analysis_data" : financial_analysis_data
        })

    justification_response = response.response if hasattr(response, 'response') else response

    try:
        update_credit_application_section(application_id, "conclusion_and_recommendation", {
            "justification_for_loan" : justification_response
        })
    except:
        print("Unable to write credit_application object")

    return JsonResponse({
            "status" : "success",
            "application_id" : application_id,
            "data": justification_response}, status=200)

import redis

def test_redis_connection(request):
    try:
        r = redis.Redis(
            host='141.147.131.163',
            port=1379,
            password="eYVX7EwVmmxKPCDmwMtyKVge8oLd2t96",
            db=0
        )
        r.set("healthcheck", "ok", ex=10)  # expires in 10 seconds
        value = r.get("healthcheck").decode("utf-8")
        return JsonResponse({"status": "success", "redis_value": value})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})

