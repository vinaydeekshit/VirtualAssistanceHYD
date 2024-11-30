
from flask import Flask, render_template, jsonify, request, session
from flask_caching import Cache
from langchain_community.llms import DeepInfra
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import numpy as np
import sys
import re
import os
import random
import time
import requests
from langdetect import detect, LangDetectException
from concurrent.futures import ThreadPoolExecutor
from pdfminer.high_level import extract_text
import spacy
from sentence_transformers import SentenceTransformer, util
from spacy import cli

# Initialize Flask app and cache
app = Flask(__name__, static_url_path='/static')
app.secret_key = 'your_secret_key'
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Initialize DeepInfra API
os.environ["DEEPINFRA_API_TOKEN"] = 'fCUq30zmzPgZJMKx2Z8kUB7HB2cgC374'
llm = DeepInfra(model_id="meta-llama/Meta-Llama-3-70B-Instruct")
llm.model_kwargs = {
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "top_p": 0.9,
}

# Initialize sentence transformer model (smaller and faster model)
retrieval_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # Faster model for embeddings

# Load SpaCy model lazily to reduce memory usage
nlp = None

def load_spacy_model():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy model 'en_core_web_sm' not found. Installing...")
            cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF using pdfminer.six
def extract_text_from_pdf(pdf_path):
    start_time = time.time()
    text = extract_text(pdf_path)
    end_time = time.time()

    extraction_time = end_time - start_time
    print(f"Text extraction took {extraction_time:.2f} seconds.")

    return text.strip()

# Function to split text into chunks
def split_into_chunks(text, chunk_size=1024):
    load_spacy_model()  # Ensure SpaCy is loaded
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in nlp(text).sents:
        sentence_text = sentence.text.strip()
        if current_length + len(sentence_text) > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence_text]
            current_length = len(sentence_text)
        else:
            current_chunk.append(sentence_text)
            current_length += len(sentence_text)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to process chunks in parallel to speed up embedding generation
def process_chunks_in_parallel(chunks):
    chunk_batches = [chunks[i:i + 10] for i in range(0, len(chunks), 10)]

    def process_batch(batch):
        return retrieval_model.encode(batch, convert_to_tensor=True)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_batch, chunk_batches))

    return [item for sublist in results for item in sublist]

def retrieve_relevant_context_parallel(chunks, query, top_k=5):
    query_embedding = retrieval_model.encode([query], convert_to_tensor=True)
    chunk_embeddings = retrieval_model.encode(chunks, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    scores = scores.tolist()
    top_results = sorted(zip(range(len(scores)), scores), key=lambda x: x[1], reverse=True)[:top_k]

    relevant_chunks = [chunks[idx] for idx, score in top_results]
    return relevant_chunks

import re

def sanitize_answer(answer):
    # Remove any special characters (e.g., backticks, extra spaces, etc.)
    sanitized_answer = re.sub(r'[`~!@#$%^&*()_=+\[\]{}|\\:;"\'<>,.?/]', '', answer)
    # You can also strip any leading or trailing whitespace
    return sanitized_answer.strip()

# Function to detect the language of the question
def detect_language(question):
    try:
        # Detect the language of the input question
        return detect(question)
    except LangDetectException:
        return 'en'  # Default to English if language detection fails

# Modify the generate_response_with_deepinfra function to consider language
def generate_response_with_deepinfra(context, question, language='en'):
    prompt = f"""
    Based on the following context, provide a clear, concise, and direct response to the question in no more than 2-3 sentences. 
    If the question is generic or asked repeatedly, the answer should still be brief (2-3 lines), ensuring consistency in the response for such questions. 
    The answer should be polite and informative, but avoid unnecessary details. 
    If no relevant context is found, provide general information related to GMR Hyderabad Airport. 
    The answer must be limited to 2-3 lines, and should not be more than that.
    If the question is not have any proper context and information regard then don't provide any random answer just say i can't understand like that text 
    If the question is asked in Hindi, Telugu, or Urdu, provide the answer in the respective language.

    Context: {context}

    Question: {question}

    Answer:
    """

    prompt_template = PromptTemplate(input_variables=["context", "question"], template=prompt)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(context=context, question=question)

    # Translate the response to the detected language if needed
    if language == 'hi':
        # Translate to Hindi (use an external library or API for translation)
        # For now, you can just simulate the translation with a placeholder
        response = f"= {response}"
    elif language == 'te':
        # Translate to Telugu
        response = f" {response}"
    elif language == 'ur':
        # Translate to Urdu
        response = f"{response}"
    
    # Sanitize the response before returning it
    return sanitize_answer(response)

@app.route('/qa/query', methods=['POST'])
def get_answer():
    try:
        # Get the JSON data sent from the client
        data = request.get_json()

        # Extract the 'question' from the received JSON
        question = data.get('question')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Detect the language of the question
        language = detect_language(question)

        # Process the question and generate the answer (use your existing logic)
        pdf_path = r"AirportData.pdf"  # Change this to your PDF path
        context = ""

        # If it's a generic question, return a short answer directly
        if is_generic_question(question):
            answer = generate_response_with_deepinfra(context, question, language)
        else:
            document_text = extract_text_from_pdf(pdf_path)
            chunks = split_into_chunks(document_text)
            context = retrieve_relevant_context_parallel(chunks, question)
            # Generate the response using DeepInfra
            answer = generate_response_with_deepinfra(context, question, language)

        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Utility function to determine if a question is generic
def is_generic_question(question):
    generic_keywords = ["hello", "hi", "help", "thank you", "goodbye"]
    return any(keyword in question.lower() for keyword in generic_keywords)


@app.route('/')
def index():
    return render_template('initial.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/wifi')
def wifi():
    return render_template('wifi.html')

@app.route('/navigation/<gateNumber>')
@app.route('/navigation/')
def navigation(gateNumber=None):
    return render_template('navigation.html', gateNumber=gateNumber)

@app.route('/index')
def main_page():
    return render_template('index.html')











@app.route('/generate_wifi_coupon', methods=['POST'])
def generate_wifi_coupon():
    url = "http://wifi.i-on.in:85/Wairport/rest/generateCoupon"
    headers = {
        "AuthKey": "685e968a14eaeeade097555e514cf2c1"
    }
    payload = request.json
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            if response_data['status'] == "1":
                return jsonify({"success": True, "couponCode": response_data['couponCode']})
            else:
                return jsonify({"success": False, "message": response_data['message']})
        else:
            return jsonify({"success": False, "message": "Failed to connect to the WiFi API.", "status_code": response.status_code})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

def detect_language(question):
    try:
        return detect(question)
    except LangDetectException:
        return 'en'  
    
def check_phrases(user_input):
    input_lower = user_input.lower()
    
   
    if any(phrase in input_lower for phrase in greeting_phrases):
        return random.choice(greeting_responses)
    
   
    if any(phrase in input_lower for phrase in thanking_phrases):
        return random.choice(thanking_responses)

    # Check for goodbye
    if any(phrase in input_lower for phrase in bye_phrases):
        return random.choice(bye_responses)
    
    return None



@app.route('/flight/status', methods=['POST'])
def flight_status():
    data = request.json
    flight_date = data.get('flightDate')
    flight_number = data.get('flightNumber')

    if not flight_date or not flight_number:
        return jsonify({"error": "Flight date and flight number are required"}), 400

    api_url = "http://61.95.132.84:8082/VirtualAgentApi/api"

    def get_flight_data(flight_type):
        payload = {
            "flightDate": flight_date,
            "flightNumber": flight_number,
            "type": flight_type
        }
        print(f"Requesting {flight_type} data with payload: {payload}")
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                if response.text.strip():  
                    print(f"Received response: {response.json()}")  
                    return response.json()
                else:
                    print(f"Empty response received for {flight_type}")
            else:
                print(f"Error: Status code {response.status_code}")
        except Exception as e:
            print(f"Exception during API call: {e}")
        return None

   
    departure_data = get_flight_data("Dep")
    arrival_data = get_flight_data("Arr")

   
    if not departure_data and not arrival_data:
        return jsonify({
            "message": "Flight Not Found. Please contact our airport staff for additional information."
        }), 404

    key_map = {
        "EB_FLNO1": "Flight Number",
        "ETAI": "Estimated Time Of Arrival",
        "FLTI": "Flight Type",
        "ORG3": "Origin",
        "STOA": "Schedule Time Of Arrival",
        "DES3": "Destination",
        "LAND": "Actual Time Of Arrival",
        "ETDI": "Estimated Time Of Departure",
        "STOD": "Schedule Time Of Departure",
        "GATE1": "Boarding Gate",
        "GATE2": "Boarding Gate 2",
        "FirstBag": "FirstBag",
        "LastBag": "LastBag"
    }

    def map_flight_data(flight_data, default_origin=None, default_destination=None):
        mapped_data = {}
        for old_key, new_key in key_map.items():
            if old_key in flight_data and flight_data[old_key]:
                mapped_data[new_key] = flight_data[old_key]
        if default_origin and "Origin" not in mapped_data:
            mapped_data["Origin"] = default_origin
        if default_destination and "Destination" not in mapped_data:
            mapped_data["Destination"] = default_destination
        return mapped_data

    result = {}
    if departure_data:
        result["Departure"] = map_flight_data(departure_data, default_origin="Hyderabad")
    if arrival_data:
        result["Arrival"] = map_flight_data(arrival_data, default_destination="Hyderabad")

    return jsonify(result)



@app.route('/validate_passenger', methods=['POST'])
def validate_passenger():
    data = request.json
    pnr = data.get('pnr')
    current_date = data.get('currentDate')

    api_url = "http://61.95.132.84:8082/VirtualAgentApi2/PassengerValidation"
    payload = {
        "pnr": pnr,
        "currentDate": current_date
    }

    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            validation_data = response.json()
            if validation_data.get('passengerValid') == "True":
               
                wifi_payload = {
                    "pnr": pnr,
                    "identityNo": "ABCD3443",
                    "location": "Delhi",
                    "userType": "Domestic",
                    "identityType": "Passport",
                    "serialNo": "007",
                    "expiryDate": "30-05-2022"
                }
                wifi_response = requests.post("http://wifi.i-on.in:85/Wairport/rest/generateCoupon", json=wifi_payload, headers={"AuthKey": "685e968a14eaeeade097555e514cf2c1"})
                if wifi_response.status_code == 200:
                    wifi_data = wifi_response.json()
                    if wifi_data['status'] == "1":
                        return jsonify({"success": True, "couponCode": wifi_data['couponCode']})
                    else:
                        return jsonify({"success": False, "message": wifi_data['message']})
                else:
                    return jsonify({"success": False, "message": "Failed to connect to the WiFi API.", "status_code": wifi_response.status_code})
            else:
                return jsonify({"success": False, "message": "Passenger is not valid for WiFi coupon."})
        else:
            return jsonify({"message": "Failed to validate passenger"}), response.status_code
    except Exception as e:
        return jsonify({"message": "An error occurred while validating passenger."}), 500
    
@app.route('/validate_flight', methods=['POST'])
def validate_flight():
    data = request.json
    flight_no = data.get('flightNo')
    current_date = data.get('currentDate')

    api_url = "http://61.95.132.84:8082/VirtualAgentApi/api"

    payload_arr = {
        "flightDate": current_date,
        "flightNumber": flight_no,
        "type": "ARR"
    }

    payload_dep = {
        "flightDate": current_date,
        "flightNumber": flight_no,
        "type": "DEP"
    }

    def validate_with_payload(payload):
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                flight_data = response.json()
                return flight_data if flight_data else None
            else:
                return None
        except Exception as e:
            return None

    flight_data_arr = validate_with_payload(payload_arr)
    flight_data_dep = validate_with_payload(payload_dep)

    if flight_data_arr or flight_data_dep:
        wifi_payload = {
            "pnr": flight_no,
            "identityNo": "ABCD3443",
            "location": "Delhi",
            "userType": "International",
            "identityType": "Passport",
            "serialNo": "007",
            "expiryDate": "30-05-2022"
        }
        try:
            wifi_response = requests.post("http://wifi.i-on.in:85/Wairport/rest/generateCoupon", json=wifi_payload, headers={"AuthKey": "685e968a14eaeeade097555e514cf2c1"})
            if wifi_response.status_code == 200:
                wifi_data = wifi_response.json()
                if wifi_data['status'] == "1":
                    return jsonify({"success": True, "couponCode": wifi_data['couponCode']})
                else:
                    return jsonify({"success": False, "message": wifi_data['message']})
            else:
                return jsonify({"success": False, "message": "Failed to connect to the WiFi API.", "status_code": wifi_response.status_code})
        except Exception as e:
            return jsonify({"success": False, "message": f"An error occurred while generating WiFi coupon: {e}"})
    else:
        return jsonify({"success": False, "message": "Flight is not valid for WiFi coupon."})


@app.route('/get_qms_data', methods=['GET'])
def get_qms_data():
    api_url = "http://61.95.132.84:8082/HialQmsApi/GetQmsData"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            qms_data = response.json()
            

            terminal_entries = [record for record in qms_data["records"] if record["LocationType"] == "TerminalEntry"]
            sha_entries = [record for record in qms_data["records"] if record["LocationType"] == "SHA"]
            
            return jsonify({
                "TerminalEntry": terminal_entries,
                "SHA": sha_entries
            })
        else:
            return jsonify({"message": "Failed to fetch QMS data", "status_code": response.status_code}), response.status_code
    except Exception as e:
        return jsonify({"message": "An error occurred while fetching QMS data", "error": str(e)}), 500

    

@app.route('/get_all_cabs_live_statics', methods=['GET'])
def get_all_cabs_live_statics():
    url = 'http://61.95.132.84:8082/CabMasterApi/CabsLiveStatics'
    response = requests.get(url)
    
    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({"error": "Failed to fetch data"}), response.status_code

@app.route('/store_boarding_gate_status', methods=['POST'])
def store_boarding_gate_status():
    data = request.json
    flight_number = data.get('flno')

    url = 'http://61.95.132.84:8082/BoardingGateStatusApi/GetBoardingPercent'
    payload = {"flno": flight_number}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  
        
        return jsonify(response.json())
    
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/get_pushpak_service', methods=['POST'])
def get_pushpak_service():
    try:
        url = " http://61.95.132.84:8082/PushpakTimingsApi/PushpakService"
        
        response = requests.post(url)
    
        if response.status_code == 200:
            data = response.json()
            return jsonify(data) 
        else:
            return jsonify({"message": "Failed to retrieve data", "status_code": response.status_code})
    except Exception as e:
        return jsonify({"message": "An error occurred while processing the request.", "error": str(e)})




@app.route('/get_flight_info', methods=['POST'])
def get_flight_info():
    try:
       
        flight_number = request.json.get('flightNumber')
        flight_type = request.json.get('type')
        flight_date = request.json.get('flightDate')

    
        if not flight_number or not flight_type or not flight_date:
            return jsonify({"message": "Missing required fields"}), 400

        url = "http://61.95.132.84:8082/VirtualAgentApi/api"

      
        payload = {
            "flightNumber": flight_number,
            "type": flight_type,
            "flightDate": flight_date
        }

      
        response = requests.post(url, json=payload)

      
        if response.status_code == 200:
            data = response.json()  
            return jsonify(data)  
        else:
            return jsonify({
                "message": "Failed to retrieve flight data", 
                "status_code": response.status_code
            }), response.status_code 

    except Exception as e:
        
        return jsonify({
            "message": "An error occurred while processing the request.",
            "error": str(e)
        }), 500  
    
if __name__ == '__main__':
    app.run(debug=True)

