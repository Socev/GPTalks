from flask import Flask, render_template, request, jsonify, redirect, url_for, session, make_response, g, Response
import openai
import pymysql
import bcrypt
import re
import tiktoken
import os
from dotenv import load_dotenv
from datetime import timedelta
import uuid
import json
import time


load_dotenv()

mysqlset_host = os.getenv("MYSQL_HOST")
mysqlset_user = os.getenv("MYSQL_USER")
mysqlset_password = os.getenv("MYSQL_PASSWORD")
mysqlset_db = os.getenv("MYSQL_DB")
openai_api_key = os.getenv("OPENAI_API_KEY")
app_secret_key = os.getenv("APP_SECRET_KEY")

openai.api_key = openai_api_key

app = Flask(__name__)

app.secret_key = app_secret_key
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=90)

default_prompts_str = os.environ.get("DEFAULT_PROMPTS")
DEFAULT_PROMPTS = json.loads(default_prompts_str)


def is_valid_password(password):
    if len(password) < 8:
        return False
    if not re.search(r'[a-z]', password) or not re.search(r'[A-Z]', password):
        return False
    if re.search(r'[^a-zA-Z0-9]', password):
        return False
    return True

@app.before_request
def load_user_status():
    if 'user_id' in session:
        g.status = "Logged in"
        g.user_id = session['user_id']
        print("User ID in session:", g.user_id)  # Add this line
    else:
        g.status = "Not logged in"
        g.user_id = ""



@app.route('/')
def index():
    username = None
    user_id = None

    if 'user_id' in session:
        user_id = session['user_id']
        conn = pymysql.connect(
            host=mysqlset_host,
            user=mysqlset_user,
            password=mysqlset_password,
            db=mysqlset_db
        )

        cursor = conn.cursor()
        cursor.execute("SELECT username FROM ice_users WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()

        if result:
            username = result[0]

    return render_template('index.html', status=g.status, username=username)

@app.route('/logout')
def logout():
    if 'user_id' in session:
        session.pop('user_id')
        g.status = "Logged out"
    else:
        g.status = "Not logged in"

    return redirect(url_for('index'))


@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    cursor = conn.cursor()
    cursor.execute("SELECT user_id, username, email FROM ice_users WHERE user_id = %s", (user_id,))
    result = cursor.fetchone()

    if result:
        user = {
            "user_id": result[0],
            "username": result[1],
            "email": result[2],
        }
        return render_template('profile.html', user=user)
    else:
        return "User not found", 404


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        email = request.form['email']
        password = request.form['password'].encode('utf-8')

        conn = pymysql.connect(
            host=mysqlset_host,
            user=mysqlset_user,
            password=mysqlset_password,
            db=mysqlset_db
        )

        cursor = conn.cursor()
        cursor.execute("SELECT user_id, password FROM ice_users WHERE email = %s", (email,))
        result = cursor.fetchone()

        if result and bcrypt.checkpw(password, result[1].encode('utf-8')):
            session['user_id'] = result[0]
            session.permanent = True
            response = make_response(jsonify(success=True))
            response.headers['Content-Type'] = 'application/json'
            return response
        else:
            response = make_response(jsonify(success=False, message="Invalid email or password"), 400)
            response.headers['Content-Type'] = 'application/json'
            return response


from flask import request, jsonify

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'GET':
        return render_template('registration.html')
    elif request.method == 'POST':
        print("POST received")
        username = request.json['username']
        email = request.json['email']
        password = request.json['password']
        validation_answer = request.json['validation_answer']

        # Check the validation question answer
        if validation_answer.strip().lower() != 'schaap':
            response = make_response(jsonify(success=False, message="Incorrect validation question answer."), 400)
            response.headers['Content-Type'] = 'application/json'
            return response

        if not is_valid_password(password):
            response = make_response(jsonify(success=False, message="Invalid password. It must be at least 8 characters long and contain both upper and lowercase letters. No special characters are allowed."), 400)
            response.headers['Content-Type'] = 'application/json'
            return response
			
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        conn = pymysql.connect(
            host=mysqlset_host,
            user=mysqlset_user,
            password=mysqlset_password,
            db=mysqlset_db
        )

        cursor = conn.cursor()
        try:
            # Generate a UUID (version 4)
            user_id = uuid.uuid4()
            cursor.execute("INSERT INTO ice_users (user_id, username, email, password) VALUES (%s, %s, %s, %s)", (user_id, username, email, hashed_password))
            conn.commit()
        except pymysql.err.IntegrityError:
            print("Email already exists")
            response = make_response(jsonify(success=False, message="Email already exists."), 400)
            response.headers['Content-Type'] = 'application/json'
            return response
        finally:
            cursor.close()

    try:
        insert_default_prompts(user_id)  # Insert default prompts for the new user
    except Exception as e:
        print("Error during inserting default prompts:", e)
        response = make_response(jsonify(success=False, message="Error inserting default prompts."), 500)
        response.headers['Content-Type'] = 'application/json'
        return response
    finally:
        conn.close()

        response = make_response(jsonify(success=True))
        response.headers['Content-Type'] = 'application/json'
        return response



@app.route('/completion', methods=['POST'])
def completion_api():
    data = request.get_json()
    print("Data received from client:", data)
    selected_model = data['model']
    max_tokens = data['max_tokens']
    temperature = data['temperature']
    conversation = data['conversation']

    # Determine the max tokens for the conversation array based on the model
    if selected_model == "gpt-3.5-turbo":
        max_tokens_for_array = 4000 - max_tokens
    elif selected_model == "gpt-4":
        max_tokens_for_array = 8000 - max_tokens
    else:
        # Handle other models or raise an error if necessary
        raise ValueError(f"Unknown model: {selected_model}")

    print("Original conversation:", conversation)
    trimmed_conversation = trim_conversation_array(conversation, max_tokens_for_array)

    def generate_response():
        retries = 3
        wait_time = 10  # Waiting time in seconds

        while retries > 0:
            try:
                completion = openai.ChatCompletion.create(
                    model=selected_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                    messages=trimmed_conversation,
                    stream=True,
                )

                print(f"Trimmed conversation: {trimmed_conversation}")

                for line in completion:
                    if 'content' in line['choices'][0]['delta']:
                        content = line['choices'][0]['delta']['content']
                        yield f"{content}"
                break

            except openai.error.RateLimitError as e:
                print(f"RateLimitError: {e}")
                retries -= 1
                if retries > 0:
                    print(f"Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    yield "The model is currently overloaded with requests. Please try again later."

    response = Response(generate_response(), mimetype='text/event-stream')
    response.headers['X-Accel-Buffering'] = 'no'
    return response



def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def trim_conversation_array(conversation_array, max_tokens_for_array):
    conversation_string = " ".join([msg["content"] for msg in conversation_array])
    total_tokens = num_tokens_from_string(conversation_string)
    print("Total tokens: " + str(total_tokens))
    print("Max tokens: " + str(max_tokens_for_array))

    while total_tokens > max_tokens_for_array:
        # Remove the first message in the array
        conversation_array.pop(0)

        conversation_string = " ".join([msg["content"] for msg in conversation_array])
        total_tokens = num_tokens_from_string(conversation_string)
        print("Total tokens after trim: " + str(total_tokens))

    return conversation_array


@app.route('/api/prompts/get_prompts', methods=['GET'])
def get_prompts():
    user_id = session.get("user_id")

    if user_id is None:
        return jsonify({"error": "Not authenticated"}), 401

    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    cursor = conn.cursor()
    cursor.execute("SELECT prompt_id, user_id, description, prompt_text, prompt_order FROM ice_prompts WHERE user_id = %s", (user_id,))
    results = cursor.fetchall()

    prompts = [
        {
            "prompt_id": result[0],
            "user_id": result[1],
            "description": result[2],
            "prompt_text": result[3],
            "prompt_order": result[4]
        }
        for result in results
    ]

    cursor.close()
    conn.close()

    return jsonify(prompts=prompts)




def insert_default_prompts(user_id):
    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    cursor = conn.cursor()
    try:
        for idx, prompt in enumerate(DEFAULT_PROMPTS):
            cursor.execute("INSERT INTO ice_prompts (user_id, description, prompt_text, prompt_order) VALUES (%s, %s, %s, %s)", (user_id, prompt['description'], prompt['prompt_text'], idx + 1))
            conn.commit()
    finally:
        cursor.close()
        conn.close()


@app.route('/api/prompts/update_prompt', methods=['POST'])
def update_prompt():
    data = request.get_json()
    prompt_id = data.get('prompt_id')
    description = data.get('description')
    prompt_text = data.get('prompt_text')
    prompt_order = data.get('prompt_order')


    user_id = session.get("user_id")
    
    if user_id is None:
        return jsonify({"error": "Not authenticated"}), 401    

    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    cursor = conn.cursor()
    cursor.execute("UPDATE ice_prompts SET description = %s, prompt_text = %s, prompt_order = %s WHERE prompt_id = %s", (description, prompt_text, prompt_order, prompt_id))
    conn.commit()

    cursor.close()
    conn.close()

    return jsonify(success=True)


@app.route('/api/prompts/add_prompt', methods=['POST'])
def add_prompt():
#    user_id = request.form.get('user_id')

    user_id = session.get("user_id")
    
    if user_id is None:
        return jsonify({"error": "Not authenticated"}), 401
	
    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    # Fetch the maximum prompt_order for the given user_id
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(MAX(prompt_order), 0) FROM ice_prompts WHERE user_id = %s", (user_id,))
    max_prompt_order = cursor.fetchone()[0]

    # Insert the new prompt with an incremented prompt_order
    cursor.execute("INSERT INTO ice_prompts (user_id, description, prompt_text, prompt_order) VALUES (%s, '', '', %s)", (user_id, max_prompt_order + 1))
    new_prompt_id = cursor.lastrowid
    conn.commit()

    # Return the new prompt
    cursor.execute("SELECT prompt_id, description, prompt_text, prompt_order FROM ice_prompts WHERE prompt_id = %s", (new_prompt_id,))
    result = cursor.fetchone()

    new_prompt = {
        "prompt_id": result[0],
        "description": result[1],
        "prompt_text": result[2],
        "prompt_order": result[3]
    }

    cursor.close()
    conn.close()

    return jsonify(new_prompt=new_prompt)


@app.route('/api/prompts/delete_prompt', methods=['POST'])
def delete_prompt():
    data = request.get_json()
    prompt_id = data.get('prompt_id')
    user_id = session.get("user_id")
    
    if user_id is None:
        return jsonify({"error": "Not authenticated"}), 401

    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    cursor = conn.cursor()
    cursor.execute("DELETE FROM ice_prompts WHERE prompt_id = %s AND user_id = %s", (prompt_id, user_id))
    conn.commit()

    cursor.close()
    conn.close()

    return jsonify(success=True)


@app.route('/api/chats/storenew', methods=['POST'])
def store_chat():
    conversation = request.json.get('conversation')
    content = json.dumps(conversation)
    print("Conversation:", conversation)
    
    print("content:", content)

    user_id = session.get("user_id")

    if user_id is None:
        return jsonify({"error": "Not authenticated"}), 401

    last_user_message = ""

    # Find the last user message
    for message in conversation[::-1]:
        if message["role"] == "user":
            last_user_message = message["content"]
            break

    if not last_user_message:
        last_user_message = "no input"

    inputarray = [{"role": "user", "content": f'Describe the goal or statement of the following text in 2 words: {last_user_message}'}]

    label = generate_chattitle(inputarray)


    label = label.rstrip(".")

    chat_order = 0
    

    
    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    cursor = conn.cursor()
    cursor.execute("INSERT INTO ice_chats (user_id, label, chat_order, content) VALUES (%s, %s, %s, %s)", (user_id, label, chat_order, content))
    conn.commit()

    chat_id = cursor.lastrowid

    cursor.close()
    conn.close()

    return jsonify({"status": "success", "chat_id": chat_id})



def generate_chattitle(inputarray):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=10,
        n=1,
        stop=None,
        messages=inputarray,
    )

    response = completion.choices[0].message['content']

    # Add debugging information
    print("Input array:", inputarray)
    print("Generated response:", response)

    return response.strip()



@app.route('/api/chats/update', methods=['POST'])
def update_chat():
    chat_id = request.form.get('chat_id')
    content = request.form.get('content')
    user_id = session.get("user_id")
    
    if user_id is None:
        return jsonify({"error": "Not authenticated"}), 401

    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    cursor = conn.cursor()
    # Update both content and datestamp
    cursor.execute("UPDATE ice_chats SET content=%s, datestamp=NOW() WHERE chat_id=%s AND user_id=%s", (content, chat_id, user_id))
    conn.commit()

    affected_rows = cursor.rowcount
    cursor.close()
    conn.close()

    if affected_rows == 0:
        # The chat_id and user_id combination was not found, and no update occurred.
        return jsonify({"status": "failure", "message": "Invalid chat_id or user_id"})
    else:
        return jsonify({"status": "success"})


@app.route('/api/chats/delete', methods=['POST'])
def delete_chat():
    chat_id = request.form.get('chat_id')
    user_id = session.get("user_id")
    
    if user_id is None:
        return jsonify({"error": "Not authenticated"}), 401

    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    cursor = conn.cursor()
    cursor.execute("DELETE FROM ice_chats WHERE chat_id=%s AND user_id=%s", (chat_id, user_id))
    conn.commit()

    affected_rows = cursor.rowcount
    cursor.close()
    conn.close()

    if affected_rows == 0:
        # The chat_id and user_id combination was not found, and no delete occurred.
        return jsonify({"status": "failure", "message": "Invalid chat_id or user_id"})
    else:
        return jsonify({"status": "success"})

@app.route('/api/chats/loadall', methods=['GET'])
def get_user_chats():
    user_id = session.get("user_id")
    
    if user_id is None:
        return jsonify({"error": "Not authenticated"}), 401

    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    cursor = conn.cursor()
    cursor.execute("SELECT chat_id, user_id, label, chat_order, datestamp FROM ice_chats WHERE user_id = %s", (user_id,))
    result = cursor.fetchall()

    chats = []
    for chat in result:
        chats.append({
            'chat_id': chat[0],
            'user_id': chat[1],
            'label': chat[2],
            'chat_order': chat[3],
            'datestamp': chat[4]
        })

    cursor.close()
    conn.close()

    return jsonify(chats)


@app.route('/api/chats/loadone', methods=['GET'])
def get_chat():
    chat_id = request.args.get('chat_id')

    user_id = session.get("user_id")
    
    if user_id is None:
        return jsonify({"error": "Not authenticated"}), 401

    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ice_chats WHERE chat_id = %s AND user_id = %s", (chat_id, user_id))
    result = cursor.fetchone()

    if result:
        chat = {
            'chat_id': result[0],
            'user_id': result[1],
            'label': result[2],
            'chat_order': result[3],
            'content': result[4],
            'datestamp': result[5]
        }
    else:
        chat = None

    cursor.close()
    conn.close()

    return jsonify(chat)


@app.route('/api/chats/updatetitle', methods=['POST'])
def update_chat_title():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"status": "failure", "message": "User not logged in"})

    chat_id = request.form.get('chat_id')
    label = request.form.get('label')
    #if the label is empty, or only spaces, save "<no label>" as label
    if not label.strip():
        label = "<no label>"

    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    cursor = conn.cursor()
    cursor.execute("UPDATE ice_chats SET label=%s WHERE chat_id=%s AND user_id=%s", (label, chat_id, user_id))
    conn.commit()

    affected_rows = cursor.rowcount
    cursor.close()
    conn.close()

    if affected_rows == 0:
        # The chat_id and user_id combination was not found, and no update occurred.
        return jsonify({"status": "failure", "message": "Invalid chat_id or user_id"})
    else:
        return jsonify({"status": "success"})



def create_users_table():
    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ice_users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL UNIQUE,
            username VARCHAR(255) NOT NULL,
            email VARCHAR(255) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

def create_prompts_table():
    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ice_prompts (
            prompt_id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL,
            description VARCHAR(255) NOT NULL,
            prompt_text VARCHAR(500) NOT NULL,
            prompt_order INT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES ice_users(user_id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

def create_chats_table():
    conn = pymysql.connect(
        host=mysqlset_host,
        user=mysqlset_user,
        password=mysqlset_password,
        db=mysqlset_db
    )

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ice_chats (
            chat_id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL,
            label VARCHAR(255) NOT NULL,
            chat_order INT NOT NULL,
            content TEXT NOT NULL,
            datestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES ice_users(user_id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == '__main__':
    create_users_table()
    create_prompts_table()
    create_chats_table()
    app.run(debug=True)
