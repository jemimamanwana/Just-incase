import os
import json
from datetime import datetime

from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
import mysql.connector
import bcrypt

# ML Engine import
try:
    from ml_engine import calculate_all_risks, calculate_hereditary_risk, init_models
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print('[WARNING] ml_engine not available — ML features disabled')

# Chatbot Engine import (local TF-IDF, no API needed)
try:
    from chatbot_engine import init_chatbot, chat_respond, get_simple_response
    CHATBOT_AVAILABLE = True
except ImportError:
    CHATBOT_AVAILABLE = False
    print('[WARNING] chatbot_engine not available — using rule-based responses only')

    # Inline fallback for simple responses when chatbot_engine can't load
    def get_simple_response(user_message, first_name=''):
        msg = user_message.lower().strip()
        name = first_name if first_name else 'there'
        if msg in ('hi', 'hello', 'hey', 'hii', 'hiii', 'heya', 'howdy', 'sup', 'yo'):
            return f"Hello {name}! I'm your GeneShield AI health companion. I can help you understand your cancer risk scores, explain your family history, recommend screenings, or answer health questions. What would you like to know?"
        if msg in ('good morning', 'morning'):
            return f"Good morning {name}! How can I help you with your health today?"
        if msg in ('good afternoon', 'afternoon'):
            return f"Good afternoon {name}! How can I assist you with your health today?"
        if msg in ('good evening', 'evening'):
            return f"Good evening {name}! What health questions can I help you with tonight?"
        if msg in ('how are you', 'how are you doing', 'how are u', 'how r u', "what's up", 'whats up'):
            return f"I'm doing great, thank you for asking {name}! I'm here and ready to help you with any health questions."
        if msg in ('thank you', 'thanks', 'thank u', 'thanx', 'ty', 'thx'):
            return f"You're welcome {name}! I'm always here to help."
        if msg in ('bye', 'goodbye', 'see you', 'see ya', 'later', 'take care'):
            return f"Goodbye {name}! Take care of yourself. I'll be here whenever you need me."
        if msg in ('who are you', 'what are you', 'what do you do'):
            return f"I'm GeneShield AI, your personal hereditary cancer risk companion. I use medical data to help you understand your cancer risk based on family history, lifestyle, and health data."
        if msg in ('help', 'help me', 'what can you do', 'options', 'menu'):
            return f"I can help with: 1) Cancer risk scores, 2) Family history analysis, 3) Screening recommendations, 4) Prevention tips, 5) General health questions. Just type your question {name}!"
        if msg in ('yes', 'yeah', 'yep', 'sure', 'ok', 'okay'):
            return f"Great! What would you like to know {name}?"
        if msg in ('no', 'nah', 'nope'):
            return f"No problem {name}! I'm here whenever you're ready."
        return None

# ============================================
# CONFIG
# ============================================

app = Flask(__name__, static_folder='.', static_url_path='')
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'geneshield-dev-secret-change-in-production')

CORS(app, supports_credentials=True)

DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', '53704468mom'),
    'database': os.environ.get('DB_NAME', 'geneshield_db'),
}

## No external API needed — chatbot runs locally via chatbot_engine.py

# ============================================
# DATABASE HELPERS
# ============================================

def get_db():
    return mysql.connector.connect(**DB_CONFIG)


def init_db():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS health_profiles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT UNIQUE NOT NULL,
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            dob DATE NULL,
            gender VARCHAR(50),
            phone VARCHAR(30),
            district VARCHAR(100),
            height FLOAT,
            weight FLOAT,
            blood_type VARCHAR(10),
            current_conditions JSON,
            medications TEXT,
            father_history JSON,
            mother_history JSON,
            grand_history JSON,
            exercise VARCHAR(50),
            diet VARCHAR(50),
            smoke VARCHAR(50),
            alcohol VARCHAR(50),
            sleep VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # New cancer-focused risk_scores table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS risk_scores (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            breast_cancer INT,
            cervical_cancer INT,
            prostate_cancer INT,
            colorectal_cancer INT,
            risk_details JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Migration: add new columns if table existed with old schema
    migration_cols = [
        ('breast_cancer', 'INT'),
        ('cervical_cancer', 'INT'),
        ('prostate_cancer', 'INT'),
        ('colorectal_cancer', 'INT'),
        ('risk_details', 'JSON'),
    ]
    for col_name, col_type in migration_cols:
        try:
            cursor.execute(f'ALTER TABLE risk_scores ADD COLUMN {col_name} {col_type}')
        except mysql.connector.Error:
            pass  # Column already exists

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vitals_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            bp_systolic INT,
            bp_diastolic INT,
            glucose FLOAT,
            heart_rate INT,
            weight FLOAT,
            temperature FLOAT,
            oxygen FLOAT,
            symptoms JSON,
            notes TEXT,
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS doctor_reports (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            report_content TEXT,
            status VARCHAR(50) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    cursor.close()
    conn.close()
    print('Database tables initialized.')


def get_user_id():
    return session.get('user_id')


# ============================================
# AUTH ROUTES
# ============================================

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    first_name = data.get('firstName', '').strip()
    last_name = data.get('lastName', '').strip()

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO users (email, password_hash, first_name, last_name) VALUES (%s, %s, %s, %s)',
            (email, password_hash, first_name, last_name)
        )
        conn.commit()
        user_id = cursor.lastrowid
        session['user_id'] = user_id
        session['email'] = email
        return jsonify({'message': 'Account created', 'user': {'id': user_id, 'email': email, 'first_name': first_name, 'last_name': last_name}}), 201
    except mysql.connector.IntegrityError:
        return jsonify({'error': 'An account with this email already exists'}), 409
    finally:
        cursor.close()
        conn.close()


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
        return jsonify({'error': 'Invalid email or password'}), 401

    session['user_id'] = user['id']
    session['email'] = user['email']
    return jsonify({'message': 'Logged in', 'user': {'id': user['id'], 'email': user['email'], 'first_name': user['first_name'], 'last_name': user['last_name']}}), 200


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out'}), 200


@app.route('/api/me', methods=['GET'])
def me():
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT id, email, first_name, last_name, created_at FROM users WHERE id = %s', (user_id,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if not user:
        session.clear()
        return jsonify({'error': 'User not found'}), 401

    if user.get('created_at'):
        user['created_at'] = user['created_at'].isoformat()

    return jsonify({'user': user}), 200


# ============================================
# PROFILE ROUTES
# ============================================

@app.route('/api/profile', methods=['POST'])
def save_profile():
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    dob = data.get('dob') or None

    conn = get_db()
    cursor = conn.cursor()

    # Check if profile exists
    cursor.execute('SELECT id FROM health_profiles WHERE user_id = %s', (user_id,))
    existing = cursor.fetchone()

    if existing:
        cursor.execute('''
            UPDATE health_profiles SET
                first_name=%s, last_name=%s, dob=%s, gender=%s, phone=%s, district=%s,
                height=%s, weight=%s, blood_type=%s, current_conditions=%s, medications=%s,
                father_history=%s, mother_history=%s, grand_history=%s,
                exercise=%s, diet=%s, smoke=%s, alcohol=%s, sleep=%s
            WHERE user_id=%s
        ''', (
            data.get('firstName'), data.get('lastName'), dob, data.get('gender'),
            data.get('phone'), data.get('district'),
            float(data['height']) if data.get('height') else None,
            float(data['weight']) if data.get('weight') else None,
            data.get('bloodType'),
            json.dumps(data.get('currentConditions', [])),
            data.get('medications'),
            json.dumps(data.get('fatherHistory', [])),
            json.dumps(data.get('motherHistory', [])),
            json.dumps(data.get('grandHistory', [])),
            data.get('exercise'), data.get('diet'), data.get('smoke'),
            data.get('alcohol'), data.get('sleep'),
            user_id
        ))
    else:
        cursor.execute('''
            INSERT INTO health_profiles
                (user_id, first_name, last_name, dob, gender, phone, district,
                 height, weight, blood_type, current_conditions, medications,
                 father_history, mother_history, grand_history,
                 exercise, diet, smoke, alcohol, sleep)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ''', (
            user_id,
            data.get('firstName'), data.get('lastName'), dob, data.get('gender'),
            data.get('phone'), data.get('district'),
            float(data['height']) if data.get('height') else None,
            float(data['weight']) if data.get('weight') else None,
            data.get('bloodType'),
            json.dumps(data.get('currentConditions', [])),
            data.get('medications'),
            json.dumps(data.get('fatherHistory', [])),
            json.dumps(data.get('motherHistory', [])),
            json.dumps(data.get('grandHistory', [])),
            data.get('exercise'), data.get('diet'), data.get('smoke'),
            data.get('alcohol'), data.get('sleep')
        ))

    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'message': 'Profile saved'}), 200


@app.route('/api/profile', methods=['GET'])
def get_profile():
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM health_profiles WHERE user_id = %s', (user_id,))
    profile = cursor.fetchone()
    cursor.close()
    conn.close()

    if not profile:
        return jsonify({'profile': None}), 200

    # Parse JSON fields
    for field in ['current_conditions', 'father_history', 'mother_history', 'grand_history', 'symptoms']:
        if field in profile and isinstance(profile[field], str):
            try:
                profile[field] = json.loads(profile[field])
            except (json.JSONDecodeError, TypeError):
                pass

    # Convert dates/datetimes to strings
    for key, val in profile.items():
        if isinstance(val, (datetime,)):
            profile[key] = val.isoformat()
        elif hasattr(val, 'isoformat'):
            profile[key] = val.isoformat()

    return jsonify({'profile': profile}), 200


# ============================================
# RISK SCORES ROUTES
# ============================================

@app.route('/api/risk-scores', methods=['POST'])
def save_risk_scores():
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    risk_details = data.get('details') or data.get('risk_details')

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO risk_scores
           (user_id, breast_cancer, cervical_cancer, prostate_cancer, colorectal_cancer, risk_details)
           VALUES (%s,%s,%s,%s,%s,%s)''',
        (
            user_id,
            data.get('breast_cancer'),
            data.get('cervical_cancer'),
            data.get('prostate_cancer'),
            data.get('colorectal_cancer'),
            json.dumps(risk_details) if risk_details else None,
        )
    )
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'message': 'Risk scores saved'}), 201


@app.route('/api/risk-scores', methods=['GET'])
def get_risk_scores():
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        'SELECT * FROM risk_scores WHERE user_id = %s ORDER BY created_at DESC LIMIT 1',
        (user_id,)
    )
    scores = cursor.fetchone()
    cursor.close()
    conn.close()

    if scores:
        if scores.get('created_at'):
            scores['created_at'] = scores['created_at'].isoformat()
        # Parse risk_details JSON
        if scores.get('risk_details') and isinstance(scores['risk_details'], str):
            try:
                scores['risk_details'] = json.loads(scores['risk_details'])
            except (json.JSONDecodeError, TypeError):
                pass

    return jsonify({'scores': scores}), 200


# ============================================
# ML RISK CALCULATION ENDPOINT
# ============================================

@app.route('/api/calculate-risks', methods=['POST'])
def calculate_risks():
    """Run ML + hereditary risk calculation and save results."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    if not ML_AVAILABLE:
        return jsonify({'error': 'ML engine not available'}), 503

    data = request.get_json()

    try:
        results = calculate_all_risks(data)
    except Exception as e:
        print(f'[ML] Risk calculation error: {e}')
        return jsonify({'error': 'Risk calculation failed'}), 500

    # Save to DB
    details = results.get('details')
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO risk_scores
           (user_id, breast_cancer, cervical_cancer, prostate_cancer, colorectal_cancer, risk_details)
           VALUES (%s,%s,%s,%s,%s,%s)''',
        (
            user_id,
            results.get('breast_cancer'),
            results.get('cervical_cancer'),
            results.get('prostate_cancer'),
            results.get('colorectal_cancer'),
            json.dumps(details) if details else None,
        )
    )
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'scores': results}), 200


# ============================================
# VITALS ROUTES
# ============================================

@app.route('/api/vitals', methods=['POST'])
def save_vitals():
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        def to_int(val):
            if val is None or val == '':
                return None
            return int(float(val))

        def to_float(val):
            if val is None or val == '':
                return None
            return float(val)

        bp_systolic = to_int(data.get('bpSystolic'))
        bp_diastolic = to_int(data.get('bpDiastolic'))
        glucose = to_float(data.get('glucose'))
        heart_rate = to_int(data.get('heartRate'))
        weight = to_float(data.get('weight'))
        temperature = to_float(data.get('temperature'))
        oxygen = to_float(data.get('oxygen'))
        symptoms = json.dumps(data.get('symptoms', []))
        notes = data.get('notes', '')

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO vitals_logs (user_id, bp_systolic, bp_diastolic, glucose, heart_rate, weight, temperature, oxygen, symptoms, notes)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ''', (
            user_id, bp_systolic, bp_diastolic, glucose, heart_rate,
            weight, temperature, oxygen, symptoms, notes
        ))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'message': 'Vitals saved'}), 201

    except Exception as e:
        print(f'[ERROR] Failed to save vitals: {e}')
        return jsonify({'error': f'Failed to save vitals: {str(e)}'}), 500


@app.route('/api/vitals', methods=['GET'])
def get_vitals():
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        'SELECT * FROM vitals_logs WHERE user_id = %s ORDER BY logged_at DESC LIMIT 20',
        (user_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    for row in rows:
        if isinstance(row.get('symptoms'), str):
            try:
                row['symptoms'] = json.loads(row['symptoms'])
            except (json.JSONDecodeError, TypeError):
                pass
        if row.get('logged_at'):
            row['logged_at'] = row['logged_at'].isoformat()

    return jsonify({'vitals': rows}), 200


# ============================================
# CHAT ROUTES
# ============================================

def build_system_prompt(user_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # Get profile
    cursor.execute('SELECT * FROM health_profiles WHERE user_id = %s', (user_id,))
    profile = cursor.fetchone() or {}

    # Parse JSON fields in profile
    for field in ['current_conditions', 'father_history', 'mother_history', 'grand_history']:
        if field in profile and isinstance(profile[field], str):
            try:
                profile[field] = json.loads(profile[field])
            except (json.JSONDecodeError, TypeError):
                profile[field] = []

    # Get latest risk scores
    cursor.execute('SELECT * FROM risk_scores WHERE user_id = %s ORDER BY created_at DESC LIMIT 1', (user_id,))
    scores = cursor.fetchone() or {}

    # Parse risk_details
    if scores.get('risk_details') and isinstance(scores['risk_details'], str):
        try:
            scores['risk_details'] = json.loads(scores['risk_details'])
        except (json.JSONDecodeError, TypeError):
            scores['risk_details'] = {}

    # Get last 5 vitals for trends
    cursor.execute('SELECT * FROM vitals_logs WHERE user_id = %s ORDER BY logged_at DESC LIMIT 5', (user_id,))
    vitals_rows = cursor.fetchall()

    cursor.close()
    conn.close()

    first_name = profile.get('first_name', 'User')
    last_name = profile.get('last_name', '')
    conditions = profile.get('current_conditions', [])
    if isinstance(conditions, str):
        try: conditions = json.loads(conditions)
        except: conditions = []

    father_h = profile.get('father_history', [])
    mother_h = profile.get('mother_history', [])
    grand_h = profile.get('grand_history', [])
    if isinstance(father_h, str):
        try: father_h = json.loads(father_h)
        except: father_h = []
    if isinstance(mother_h, str):
        try: mother_h = json.loads(mother_h)
        except: mother_h = []
    if isinstance(grand_h, str):
        try: grand_h = json.loads(grand_h)
        except: grand_h = []

    # Calculate BMI
    weight = float(profile.get('weight') or 0)
    height = float(profile.get('height') or 0)
    bmi_str = 'Unknown'
    if weight > 0 and height > 0:
        bmi = weight / ((height / 100) ** 2)
        bmi_str = f'{bmi:.1f}'

    def risk_label(val):
        if val is None:
            return 'Not assessed'
        if val >= 60:
            return f'{val}% (HIGH)'
        if val >= 40:
            return f'{val}% (MODERATE)'
        return f'{val}% (LOW)'

    # Vitals trends
    vitals_trend = 'No vitals logged yet.'
    if vitals_rows:
        latest = vitals_rows[0]
        bp = f"{latest.get('bp_systolic', '--')}/{latest.get('bp_diastolic', '--')} mmHg"
        readings = []
        for v in vitals_rows:
            if v.get('bp_systolic'):
                readings.append(f"  {v.get('logged_at', 'N/A')}: BP {v['bp_systolic']}/{v.get('bp_diastolic', '--')}, Glucose {v.get('glucose', '--')}, HR {v.get('heart_rate', '--')}")
        vitals_trend = f"Latest: BP {bp}, Glucose {latest.get('glucose', '--')} mmol/L, HR {latest.get('heart_rate', '--')} bpm, Weight {latest.get('weight', '--')} kg\n" + '\n'.join(readings[:5])

    # Risk details analysis
    risk_details = scores.get('risk_details', {})
    family_analysis = ''
    if isinstance(risk_details, dict) and risk_details.get('family_analysis'):
        fa = risk_details['family_analysis']
        for cancer, info in fa.items():
            if isinstance(info, dict):
                family_analysis += f"  - {cancer}: base {info.get('base_risk', '?')}%, family mult {info.get('family_multiplier', '?')}x ({info.get('family_source', 'unknown')}), combined mult {info.get('combined_multiplier', '?')}x\n"

    return f"""You are GeneShield AI — a warm, knowledgeable hereditary CANCER companion for {first_name}.

You are powered by ML models trained on real medical datasets combined with Bayesian hereditary risk analysis.

Your PRIMARY focus is CANCER — hereditary cancer risk, screening guidance, prevention, and early detection.

CONTEXT: In Botswana, most cancer patients are diagnosed at advanced stages (Stage III/IV). Only 8.4% are caught at Stage I. GeneShield exists to change this through early risk detection and proactive screening.

USER PROFILE:
- Name: {first_name} {last_name}
- Gender: {profile.get('gender', 'Unknown')}
- District: {profile.get('district', 'Botswana')}
- BMI: {bmi_str}
- Current conditions: {', '.join(conditions) if conditions else 'None reported'}
- Medications: {profile.get('medications', 'None')}
- Exercise: {profile.get('exercise', 'Unknown')} | Diet: {profile.get('diet', 'Unknown')}
- Smoking: {profile.get('smoke', 'Unknown')} | Alcohol: {profile.get('alcohol', 'Unknown')}

FAMILY CANCER HISTORY:
- Father's side: {', '.join(father_h) if father_h else 'No cancer history reported'}
- Mother's side: {', '.join(mother_h) if mother_h else 'No cancer history reported'}
- Grandparents: {', '.join(grand_h) if grand_h else 'No cancer history reported'}

HEREDITARY CANCER RISK SCORES (ML + Epidemiological Analysis):
- Breast Cancer: {risk_label(scores.get('breast_cancer'))}
- Cervical Cancer: {risk_label(scores.get('cervical_cancer'))}
- Prostate Cancer: {risk_label(scores.get('prostate_cancer'))}
- Colorectal Cancer: {risk_label(scores.get('colorectal_cancer'))}

RISK FACTOR ANALYSIS:
{family_analysis if family_analysis else '  No detailed analysis available yet.'}

VITALS TRENDS (Last 5 readings):
{vitals_trend}

FAMILY HISTORY CONSULTATION:
- When users ask about family history, EXPLAIN SPECIFICALLY how each relative's cancer affects THEIR risk
- Example: "Your mother had breast cancer, which increases your risk by 2x according to the Collaborative Group study (Lancet 2001)"
- Explain the difference between first-degree (parent = higher risk multiplier) and second-degree (grandparent = moderate risk) relatives
- Connect family cancers to recommended screenings: breast cancer history -> mammograms, colorectal -> colonoscopy, cervical -> more frequent Pap smears
- Reference actual multipliers: first-degree breast = 2.0x, prostate = 2.5x, colorectal = 2.2x, cervical = 1.6x

SCREENING GUIDANCE FOR BOTSWANA:
- Mammogram: Available at Princess Marina Hospital and private facilities in Gaborone. Every 1-2 years from age 40 (or 10 years before youngest affected relative). BRCA testing if strong family history.
- Pap smear: Available at government clinics — recommend every 3 years from age 21. HPV co-testing every 5 years from age 30. HPV vaccination if eligible.
- PSA test: Discuss with doctor from age 50 (or 40 with family history or African descent). Available at Princess Marina and Nyangabgwe.
- Colonoscopy: Recommend from age 45, earlier (age 40) with family history. Every 10 years or stool-based tests as alternative.

GUIDELINES:
- Always reference the user's SPECIFIC family history and risk percentages
- Explain WHY their score is what it is (which family members, which multipliers)
- Give actionable screening schedules based on their risk level
- Be warm and encouraging — high risk doesn't mean certainty, it means proactive action
- Mention lifestyle modifications: diet (morogo, sorghum, vegetables), exercise, limiting alcohol
- NEVER diagnose — provide risk information and guidance only
- For serious concerns, recommend Princess Marina Hospital (Gaborone) or Nyangabgwe Referral Hospital (Francistown)
- Keep responses 2-4 paragraphs
- If asked about topics outside cancer/health, gently redirect to your area of expertise"""


@app.route('/api/chat/send', methods=['POST'])
def chat_send():
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'Message is required'}), 400

    conn = get_db()
    cursor = conn.cursor()

    # Save user message
    cursor.execute(
        'INSERT INTO chat_history (user_id, role, content) VALUES (%s, %s, %s)',
        (user_id, 'user', user_message)
    )
    conn.commit()

    # Build user context from DB for the local chatbot
    ai_reply = None

    try:
        # Fetch profile + risk scores for chatbot context
        cursor_ctx = conn.cursor(dictionary=True)

        cursor_ctx.execute('SELECT * FROM health_profiles WHERE user_id = %s', (user_id,))
        profile = cursor_ctx.fetchone() or {}

        # Parse JSON fields
        for field in ['father_history', 'mother_history', 'grand_history', 'current_conditions']:
            if field in profile and isinstance(profile[field], str):
                try:
                    profile[field] = json.loads(profile[field])
                except (json.JSONDecodeError, TypeError):
                    profile[field] = []

        cursor_ctx.execute('SELECT * FROM risk_scores WHERE user_id = %s ORDER BY created_at DESC LIMIT 1', (user_id,))
        scores_row = cursor_ctx.fetchone() or {}

        # Get recent chat history for Gemini context
        cursor_ctx.execute(
            'SELECT role, content FROM chat_history WHERE user_id = %s ORDER BY created_at DESC LIMIT 10',
            (user_id,)
        )
        recent_history = cursor_ctx.fetchall()
        recent_history.reverse()  # chronological order

        cursor_ctx.close()

        first_name = profile.get('first_name', 'User')

        user_context = {
            'first_name': first_name,
            'gender': profile.get('gender', ''),
            'risk_scores': {
                'breast_cancer': scores_row.get('breast_cancer'),
                'cervical_cancer': scores_row.get('cervical_cancer'),
                'prostate_cancer': scores_row.get('prostate_cancer'),
                'colorectal_cancer': scores_row.get('colorectal_cancer'),
            },
            'father_history': profile.get('father_history', []),
            'mother_history': profile.get('mother_history', []),
            'grand_history': profile.get('grand_history', []),
        }

        # Build the rich system prompt for Gemini API
        system_prompt = build_system_prompt(user_id)

        # Use chatbot engine: Gemini API -> TF-IDF -> fallback
        if CHATBOT_AVAILABLE:
            ai_reply = chat_respond(user_message, user_context, system_prompt=system_prompt, chat_history=recent_history)
        else:
            # Fallback: try if/then rules when chatbot engine isn't loaded
            ai_reply = get_simple_response(user_message, first_name)
            if not ai_reply:
                name = first_name if first_name else 'there'
                ai_reply = (
                    f"Hello {name}! Thank you for your question. I can help you with your cancer risk scores, "
                    f"family history, screening recommendations, and prevention tips. "
                    f"Try asking me something like 'What are my risk scores?' or 'What screenings should I get?'"
                )

    except Exception as e:
        print(f'[Chat] Error generating response: {e}')
        # Even on error, give a helpful response instead of an error message
        ai_reply = get_simple_response(user_message, 'there')
        if not ai_reply:
            ai_reply = "Thank you for your question! I can help you with cancer risk scores, family history, screening recommendations, and prevention tips. Try asking about one of these topics."

    # Save assistant reply
    cursor.execute(
        'INSERT INTO chat_history (user_id, role, content) VALUES (%s, %s, %s)',
        (user_id, 'assistant', ai_reply)
    )
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'reply': ai_reply}), 200


@app.route('/api/chat/history', methods=['GET'])
def chat_history():
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        'SELECT role, content, created_at FROM chat_history WHERE user_id = %s ORDER BY created_at ASC LIMIT 50',
        (user_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    for row in rows:
        if row.get('created_at'):
            row['created_at'] = row['created_at'].isoformat()

    return jsonify({'history': rows}), 200


@app.route('/api/chat/history', methods=['DELETE'])
def clear_chat():
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM chat_history WHERE user_id = %s', (user_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'message': 'Chat history cleared'}), 200


# ============================================
# DOCTOR REPORTS ROUTES
# ============================================

@app.route('/api/reports', methods=['POST'])
def save_report():
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO doctor_reports (user_id, report_content, status) VALUES (%s, %s, %s)',
        (user_id, data.get('reportContent', ''), data.get('status', 'pending'))
    )
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'message': 'Report saved'}), 201


@app.route('/api/reports', methods=['GET'])
def get_reports():
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        'SELECT * FROM doctor_reports WHERE user_id = %s ORDER BY created_at DESC',
        (user_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    for row in rows:
        if row.get('created_at'):
            row['created_at'] = row['created_at'].isoformat()

    return jsonify({'reports': rows}), 200


# ============================================
# PDF REPORT DATA ENDPOINT
# ============================================

@app.route('/api/generate-report', methods=['GET'])
def generate_report():
    """Compile all user health data for PDF report generation."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # User info
    cursor.execute('SELECT id, email, first_name, last_name, created_at FROM users WHERE id = %s', (user_id,))
    user = cursor.fetchone() or {}
    if user.get('created_at'):
        user['created_at'] = user['created_at'].isoformat()

    # Health profile
    cursor.execute('SELECT * FROM health_profiles WHERE user_id = %s', (user_id,))
    profile = cursor.fetchone() or {}
    for field in ['current_conditions', 'father_history', 'mother_history', 'grand_history']:
        if field in profile and isinstance(profile[field], str):
            try:
                profile[field] = json.loads(profile[field])
            except (json.JSONDecodeError, TypeError):
                profile[field] = []
    for key, val in list(profile.items()):
        if hasattr(val, 'isoformat'):
            profile[key] = val.isoformat()

    # Risk scores
    cursor.execute('SELECT * FROM risk_scores WHERE user_id = %s ORDER BY created_at DESC LIMIT 1', (user_id,))
    scores = cursor.fetchone() or {}
    if scores.get('created_at'):
        scores['created_at'] = scores['created_at'].isoformat()
    if scores.get('risk_details') and isinstance(scores['risk_details'], str):
        try:
            scores['risk_details'] = json.loads(scores['risk_details'])
        except (json.JSONDecodeError, TypeError):
            pass

    # Vitals history (last 10)
    cursor.execute('SELECT * FROM vitals_logs WHERE user_id = %s ORDER BY logged_at DESC LIMIT 10', (user_id,))
    vitals = cursor.fetchall()
    for v in vitals:
        if v.get('logged_at'):
            v['logged_at'] = v['logged_at'].isoformat()
        if isinstance(v.get('symptoms'), str):
            try:
                v['symptoms'] = json.loads(v['symptoms'])
            except (json.JSONDecodeError, TypeError):
                pass

    cursor.close()
    conn.close()

    # Build screening recommendations based on risk
    gender = (profile.get('gender') or '').lower()
    recommendations = []

    breast_risk = scores.get('breast_cancer')
    if breast_risk is not None and gender != 'male':
        if breast_risk >= 40:
            recommendations.append({'type': 'Mammogram', 'urgency': 'HIGH', 'detail': 'Annual mammogram recommended. Discuss BRCA genetic testing with your doctor.'})
        else:
            recommendations.append({'type': 'Mammogram', 'urgency': 'ROUTINE', 'detail': 'Mammogram every 1-2 years from age 40. Monthly self-exams.'})

    cervical_risk = scores.get('cervical_cancer')
    if cervical_risk is not None and gender != 'male':
        if cervical_risk >= 40:
            recommendations.append({'type': 'Pap Smear', 'urgency': 'HIGH', 'detail': 'Annual Pap smear recommended. HPV co-testing every 3 years.'})
        else:
            recommendations.append({'type': 'Pap Smear', 'urgency': 'ROUTINE', 'detail': 'Pap smear every 3 years from age 21. HPV vaccination if eligible.'})

    prostate_risk = scores.get('prostate_cancer')
    if prostate_risk is not None and gender != 'female':
        if prostate_risk >= 40:
            recommendations.append({'type': 'PSA Test', 'urgency': 'HIGH', 'detail': 'Annual PSA screening from age 40 given elevated family risk.'})
        else:
            recommendations.append({'type': 'PSA Test', 'urgency': 'ROUTINE', 'detail': 'Discuss PSA testing with doctor from age 50.'})

    colorectal_risk = scores.get('colorectal_cancer')
    if colorectal_risk is not None:
        if colorectal_risk >= 40:
            recommendations.append({'type': 'Colonoscopy', 'urgency': 'HIGH', 'detail': 'Colonoscopy recommended from age 40. Available at Princess Marina Hospital.'})
        else:
            recommendations.append({'type': 'Colonoscopy', 'urgency': 'ROUTINE', 'detail': 'Colonoscopy every 10 years from age 45.'})

    # BMI calculation
    weight = float(profile.get('weight') or 0)
    height = float(profile.get('height') or 0)
    bmi = None
    bmi_category = None
    if weight > 0 and height > 0:
        bmi = round(weight / ((height / 100) ** 2), 1)
        if bmi < 18.5:
            bmi_category = 'Underweight'
        elif bmi < 25:
            bmi_category = 'Normal'
        elif bmi < 30:
            bmi_category = 'Overweight'
        else:
            bmi_category = 'Obese'

    report_data = {
        'generated_at': datetime.now().isoformat(),
        'user': user,
        'profile': profile,
        'risk_scores': scores,
        'vitals': vitals,
        'recommendations': recommendations,
        'bmi': bmi,
        'bmi_category': bmi_category,
    }

    return jsonify({'report': report_data}), 200


# ============================================
# SERVE STATIC FILES
# ============================================

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    init_db()
    if ML_AVAILABLE:
        init_models()
    if CHATBOT_AVAILABLE:
        print('[Chatbot] Initializing medical chatbot engine...')
        init_chatbot()
    app.run(debug=True, host='0.0.0.0', port=5000)
