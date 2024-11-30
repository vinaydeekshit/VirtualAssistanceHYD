from flask import Flask, render_template, jsonify, request, session
from flask_caching import Cache
import difflib
import requests
from langchain_community.llms import DeepInfra
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from qa_data import qa_data

app = Flask(__name__)
app.secret_key = 'your_secret_key'
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

os.environ["DEEPINFRA_API_TOKEN"] = 'fCUq30zmzPgZJMKx2Z8kUB7HB2cgC374'
llm = DeepInfra(model_id="meta-llama/Meta-Llama-3-70B-Instruct")
llm.model_kwargs = {
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "max_new_tokens": 100,  
    "top_p": 0.9,
}

template = """Conversation:
{conversation}

Current Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["conversation", "question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

def find_matching_question(user_question):
    user_question_processed = user_question.lower().strip()
    matching_question_id = None
    max_score = 0.75
    best_score = 0
    for question_id, data in qa_data.items():
        question_processed = data["question"].lower().strip()
        score = difflib.SequenceMatcher(None, user_question_processed, question_processed).ratio()
        if score > best_score:
            best_score = score
            matching_question_id = question_id
    if best_score < max_score:
        matching_question_id = None
    return matching_question_id

@app.route('/')
def index():
    return render_template('page2.html')

@app.route('/wifi')
def wifi():
    return render_template('wifi.html')

@app.route('/index')
def main_page():
    session.clear()
    return render_template('index.html')

@app.route('/qa', methods=['GET'])
def get_qa():
    return jsonify({"message": "This endpoint provides the QA data"})

def is_response_complete(response):
    return response.endswith('.') or response.endswith('!') or response.endswith('?')

def filter_repetitive_content(response):
    sentences = response.split('. ')
    unique_sentences = []
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)
    return '. '.join(unique_sentences) + '.'

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

@cache.cached(timeout=300)  # Cache for 5 minutes
@app.route('/qa/query', methods=['POST'])
def query_qa():
    data = request.json
    user_question = data.get('question', '').strip()

    if not user_question:
        return jsonify({"message": "Please provide a valid question."})

    try:
        matching_question_id = find_matching_question(user_question)
        if matching_question_id:
            agent_answer = qa_data[matching_question_id]['answer']
            return jsonify({"answer": agent_answer})

        if 'conversation' not in session:
            session['conversation'] = []

        session['conversation'].append(f"User: {user_question}")

        conversation_history = "\n".join(session['conversation'])

        response = llm_chain.run(conversation=conversation_history, question=user_question)
        final_response = filter_repetitive_content(response)

        if final_response:
            session['conversation'].append(f"Agent: {final_response}")
            return jsonify({"answer": final_response})
        else:
            return jsonify({"message": "Sorry, I couldn't find an answer to your question."})
    except Exception as e:
        return jsonify({"message": "An error occurred. Please try again later."})

@app.route('/flight/status', methods=['POST'])
def flight_status():
    data = request.json
    flight_date = data.get('flightDate')
    flight_number = data.get('flightNumber')
    flight_type = data.get('type')

    api_url = "http://61.95.132.84:8082/VirtualAgentApi/api"
    payload = {
        "flightDate": flight_date,
        "flightNumber": flight_number,
        "type": flight_type
    }
    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            flight_data = response.json()

            mapped_data = {}
            key_map = {
                "EB_FLNO1": "FLIGHT NUMBER",
                "ETAI": "ESTIMATED TIME OF ARRIVAL",
                "FLTI": "FLIGHT TYPE",
                "ORG3": "ORIGIN",
                "STOA": "SCHEDULED TIME OF FLIGHT",
                "DES3": "DESTINATION",
                "ETDI": "ESTIMATED TIME OF DEPARTURE",
                "STOD": "SCHEDULE TIME OF FLIGHT",
                "GATE1": "GATE1",
                "GATE2": "GATE2",
                "FirstBag": "FirstBag",
                "LastBag": "LastBag"
            }

            for old_key, new_key in key_map.items():
                if old_key in flight_data and flight_data[old_key]:
                    if "TIME" in new_key or new_key in ["FirstBag", "LastBag"]:
                        flight_data[old_key] = flight_data[old_key][:16]
                    mapped_data[new_key] = flight_data[old_key]

            return jsonify(mapped_data)
        else:
            return jsonify({"Failed to fetch flight status"}), response.status_code
    except Exception as e:
        return jsonify({"Flight Not Found. please contact our airport staff for additional information"}), 500

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

if __name__ == '__main__':
    app.run(debug=True)