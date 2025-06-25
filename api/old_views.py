import os
import uuid
import json
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Application, PDFDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
import datetime
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor
from langchain.embeddings.openai import OpenAIEmbeddings

from .helpers import load_embeddings, save_embeddings, validate_pdf, process_question, generate_pdf_response
from .helpers import generate_doc_response

@csrf_exempt
def upload_pdfs(request):
    if request.method == 'POST':
        application_id = str(uuid.uuid4())
        print('Process started for Application id ', application_id)
        pdfs = request.FILES.getlist('pdfs')

        application, created = Application.objects.get_or_create(application_id=application_id)

        # Validate and process PDFs
        valid_pdfs, invalid_pdfs = validate_pdf(pdfs)

        if invalid_pdfs:
            print("Invalid PDF files detected. Please upload PDFs with valid content.")

        if valid_pdfs:
            print(f"{len(valid_pdfs)} PDF validated successfully.")

            # Add valid PDF names to the database
            for pdf, _ in valid_pdfs:
                PDFDocument.objects.create(
                    application=application,
                    pdf_name=os.path.basename(pdf.name),
                    time_uploaded=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                )

            # Improved text extraction and chunking with metadata
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=500,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            chunks_with_metadata = []
            for pdf, _ in valid_pdfs:
                pdf.seek(0)
                pdf_reader = PdfReader(pdf)
                pdf_name = os.path.basename(pdf.name)
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        # Clean the extracted text
                        cleaned_text = ' '.join(page_text.split())
                        cleaned_text = cleaned_text.replace('-\n', '').replace('\n', ' ')
                        
                        # Split each page's text into chunks but keep metadata
                        page_chunks = text_splitter.split_text(cleaned_text)
                        for chunk_num, chunk in enumerate(page_chunks, start=1):
                            metadata = {
                                "source": pdf_name,
                                "page": page_num,
                                "chunk": chunk_num
                            }
                            chunks_with_metadata.append((chunk, metadata))

            if chunks_with_metadata:
                # Load existing embeddings or create new ones with metadata
                user_embeddings = load_embeddings(application_id, text_embeddings)
                if user_embeddings is None:
                    print("Creating new embeddings...")
                    vectorstore = FAISS.from_texts(
                        texts=[chunk for chunk, _ in chunks_with_metadata],
                        metadatas=[metadata for _, metadata in chunks_with_metadata],
                        embedding=text_embeddings
                    )
                else:
                    print("Appending to existing embeddings...")
                    vectorstore = user_embeddings
                    vectorstore.add_texts(
                        texts=[chunk for chunk, _ in chunks_with_metadata],
                        metadatas=[metadata for _, metadata in chunks_with_metadata]
                    )

                save_embeddings(application_id, vectorstore)

                # Load predefined questions from constant.json
                try:
                    with open("constant/constant.json", "r") as f:
                        questions_data = json.load(f)
                        questions = [q for section in questions_data["data"] for q in section["questions"]]
                        questions = [q if isinstance(q, str) else list(q.values())[0] for q in questions]
                except Exception as e:
                    return JsonResponse({"error": f"Failed to load questions: {str(e)}"}, status=500)

                # Process questions with LLM
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                question_response_map = {}

                with ThreadPoolExecutor() as executor:
                    responses = executor.map(lambda q: process_question(q, vectorstore, llm), questions)
                for question, response in responses:
                    question_response_map[question] = response

                print("Final question-response map:", question_response_map)

                # Generate and save PDF response
                pdf_filename = f"pdfs/{application_id}_response.pdf"
                generate_pdf_response(question_response_map, pdf_filename, application_id)
                print("PDF File created")

                try:
                    with open("constant\\template_questions.json", "r") as f:
                        template_data = json.load(f)
                except Exception as e:
                    return JsonResponse({"error": f"Failed to load questions: {str(e)}"}, status=500)
                
                for key, question in template_data.items():
                    if question:
                        q, response = process_question(question, vectorstore, llm)
                        template_data[key] = response

                generate_doc_response(template_data, application_id)

                return JsonResponse({
                    "message": "PDFs processed successfully",
                    "application_id": application_id,
                    "pdf_file": pdf_filename
                }, status=200)
            else:
                return JsonResponse({"error": "No text found in the uploaded PDF files. Please upload PDFs with content."}, status=400)
        else:
            return JsonResponse({"error": "No valid PDFs found"}, status=400)
    return JsonResponse({"error": "Invalid request method"}, status=405)

text_embeddings = OpenAIEmbeddings()

def download_pdf(request):
    application_id = request.GET.get("application_id")
    if not application_id:
        return JsonResponse({"error": "application_id is required"}, status=400)
    pdf_path = f"pdfs/{application_id}_final.pdf"
    #pdf_path = "pdfs/report.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(open(pdf_path, 'rb'), as_attachment=True)
    return JsonResponse({"error": "PDF not found"}, status=404)

def download_doc(request):
    application_id = request.GET.get("application_id")
    if not application_id:
        return JsonResponse({"error": "application_id is required"}, status=400)
    doc_path = f"docs//{application_id}_final.doc"
    if os.path.exists(doc_path):
        return FileResponse(open(doc_path, 'rb'), as_attachment=True)
    return JsonResponse({"error": "Doc not found"}, status=404)

@csrf_exempt
def chat_with_pdf(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            application_id = data.get('application_id')
            questions = data.get('questions', [])

            if not application_id or not questions:
                return JsonResponse({"error": "Missing application_id or questions"}, status=400)

            print(f"Asking Questions for Application ID: {application_id}")
            print(f"Questions: {questions}")

            # Load the vectorstore
            vectorstore = load_embeddings(application_id, text_embeddings)
            print("Vectorstore loaded for the application ID")

            # Load LLM
            llm = ChatOpenAI(model="gpt-4o")

            # Process questions using multithreading
            question_response_map = {}
            with ThreadPoolExecutor() as executor:
                results = executor.map(lambda q: (q, process_question(q, vectorstore, llm)), questions)

            # Store results in dictionary
            for question, response in results:
                question_response_map[question] = response

            print("Final question-response map:", question_response_map)

            # Return response JSON
            return JsonResponse({"responses": question_response_map}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        except Exception as e:
            print(f"Error processing questions: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)