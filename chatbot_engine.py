"""
GeneShield Medical Chatbot Engine
===================================
Hybrid AI chatbot:
  1. Google Gemini API for intelligent, context-aware responses (primary)
  2. TF-IDF retrieval from 257K doctor-patient conversations (fallback)
  3. Simple if/then rules for greetings and common phrases
"""

import os
import re
import json
import warnings
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Paths for cached index
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'chatbot_vectorizer.pkl')
MATRIX_PATH = os.path.join(MODELS_DIR, 'chatbot_tfidf_matrix.pkl')
RESPONSES_PATH = os.path.join(MODELS_DIR, 'chatbot_responses.pkl')


# ============================================
# GEMINI API INTEGRATION
# ============================================

def _call_gemini(system_prompt, user_message, history=None):
    """
    Call Google Gemini API for intelligent responses.
    Uses urllib (built-in) — no extra dependencies needed.
    Set GEMINI_API_KEY environment variable to enable.
    """
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        return None

    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}'

    # Build conversation contents
    contents = []

    # Add chat history if available (last 6 messages for context)
    if history:
        for msg in history[-6:]:
            role = 'user' if msg.get('role') == 'user' else 'model'
            contents.append({
                'role': role,
                'parts': [{'text': msg.get('content', '')}]
            })

    # Add current user message
    contents.append({
        'role': 'user',
        'parts': [{'text': user_message}]
    })

    payload = {
        'contents': contents,
        'systemInstruction': {
            'parts': [{'text': system_prompt}]
        },
        'generationConfig': {
            'temperature': 0.75,
            'maxOutputTokens': 1200,
            'topP': 0.92,
        },
        'safetySettings': [
            {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
        ]
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode('utf-8'))
            text = result['candidates'][0]['content']['parts'][0]['text']
            # Clean up markdown artifacts that don't render well in the chat
            text = text.strip()
            return text
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='ignore')
        print(f'[Gemini] HTTP {e.code}: {body[:200]}')
        return None
    except Exception as e:
        print(f'[Gemini] API call failed: {e}')
        return None


def _find_csv():
    """Find the ai-medical-chatbot.csv file."""
    candidates = [
        os.path.join(BASE_DIR, 'ai-medical-chatbot.csv'),
        os.path.join(BASE_DIR, 'data', 'ai-medical-chatbot.csv'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _clean_text(text):
    """Basic text cleaning for better matching."""
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    text = re.sub(r'hi\s+doctor[,.]?\s*', '', text)
    text = re.sub(r'hello\s+doctor[,.]?\s*', '', text)
    text = re.sub(r'dear\s+doctor[,.]?\s*', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================
# SIMPLE IF/THEN RULES FOR COMMON PHRASES
# ============================================

def get_simple_response(user_message, first_name=''):
    """
    Handle simple conversational messages with if/then rules.
    Returns a response string if matched, or None to fall through to TF-IDF.
    """
    msg = user_message.lower().strip()
    name = first_name if first_name else 'there'

    # --- Greetings ---
    if msg in ('hi', 'hello', 'hey', 'hii', 'hiii', 'heya', 'howdy', 'sup', 'yo'):
        return f"Hello {name}! I'm your GeneShield AI health companion. I can help you understand your cancer risk scores, explain your family history, recommend screenings, or answer health questions. What would you like to know?"

    if msg in ('good morning', 'morning'):
        return f"Good morning {name}! I hope you're having a great start to your day. How can I help you with your health today?"

    if msg in ('good afternoon', 'afternoon'):
        return f"Good afternoon {name}! How can I assist you with your health today?"

    if msg in ('good evening', 'evening'):
        return f"Good evening {name}! What health questions can I help you with tonight?"

    if msg in ('good night', 'goodnight', 'gn'):
        return f"Good night {name}! Remember, taking care of your health is a daily commitment. Sleep well and feel free to chat anytime."

    # --- How are you ---
    if msg in ('how are you', 'how are you doing', 'how are u', 'how r u', 'how do you do', "what's up", 'whats up', 'wassup'):
        return f"I'm doing great, thank you for asking {name}! I'm here and ready to help you with any health questions. Would you like to check your cancer risk scores, learn about screenings, or ask about a health topic?"

    # --- Thank you ---
    if msg in ('thank you', 'thanks', 'thank u', 'thanx', 'ty', 'thx', 'thanks a lot', 'thank you so much', 'much appreciated'):
        return f"You're welcome {name}! I'm always here to help. Don't hesitate to ask if you have more health questions."

    # --- Goodbye ---
    if msg in ('bye', 'goodbye', 'see you', 'see ya', 'later', 'gtg', 'gotta go', 'take care', 'farewell'):
        return f"Goodbye {name}! Take care of yourself. Remember to stay on top of your health screenings. I'll be here whenever you need me."

    # --- Who are you ---
    if msg in ('who are you', 'what are you', 'what is geneshield', 'what do you do', 'tell me about yourself'):
        return f"I'm GeneShield AI, your personal hereditary cancer risk companion. I use machine learning and medical data to help you understand your cancer risk based on family history, lifestyle, and health data. I can explain your risk scores, recommend cancer screenings, suggest prevention steps, and answer general health questions. How can I help you today {name}?"

    # --- Help ---
    if msg in ('help', 'help me', 'what can you do', 'what can i ask', 'options', 'menu'):
        return (
            f"Here's what I can help you with {name}:\n\n"
            "1. **Cancer Risk Scores** — Ask me to explain your risk percentages and what they mean.\n"
            "2. **Family History** — I can explain how your family's cancer history affects your risk.\n"
            "3. **Screening Recommendations** — I'll tell you which tests to get and when.\n"
            "4. **Prevention Tips** — Diet, exercise, and lifestyle changes to lower your risk.\n"
            "5. **General Health Questions** — I can answer medical questions from my database of 257,000 doctor consultations.\n\n"
            "Just type your question and I'll do my best to help!"
        )

    # --- Yes / No / Ok ---
    if msg in ('yes', 'yeah', 'yep', 'yea', 'sure', 'ok', 'okay', 'alright', 'right'):
        return f"Great! What would you like to know {name}? You can ask me about your cancer risk scores, family history, screening schedule, or any health topic."

    if msg in ('no', 'nah', 'nope', 'not really'):
        return f"No problem {name}! I'm here whenever you're ready. Feel free to ask me anything about your health."

    # --- Feeling unwell ---
    if any(phrase in msg for phrase in ['i feel sick', 'i am sick', "i'm sick", 'not feeling well', "don't feel well", 'feeling unwell', 'i feel unwell']):
        return f"I'm sorry to hear you're not feeling well {name}. Can you describe your symptoms? I can try to help with general health guidance. If your symptoms are severe, please visit your nearest clinic or hospital right away."

    # --- Greeting with question pattern ---
    greet_prefixes = ('hi ', 'hello ', 'hey ', 'hi, ', 'hello, ', 'hey, ')
    for prefix in greet_prefixes:
        if msg.startswith(prefix) and len(msg) > len(prefix) + 5:
            # Has a question after the greeting — let TF-IDF handle it
            return None

    return None


class MedicalChatbot:
    """TF-IDF retrieval chatbot trained on doctor-patient conversations."""

    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.responses = None
        self.descriptions = None
        self.ready = False

    def build_index(self):
        """Load CSV, build TF-IDF index, cache to disk."""
        csv_path = _find_csv()
        if not csv_path:
            print('[Chatbot] ai-medical-chatbot.csv not found — chatbot will use rule-based responses only')
            return False

        # Check if cached index exists
        if (os.path.exists(VECTORIZER_PATH) and
                os.path.exists(MATRIX_PATH) and
                os.path.exists(RESPONSES_PATH)):
            print('[Chatbot] Loading cached TF-IDF index...')
            try:
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                self.tfidf_matrix = joblib.load(MATRIX_PATH)
                data = joblib.load(RESPONSES_PATH)
                self.responses = data['responses']
                self.descriptions = data['descriptions']
                self.ready = True
                print(f'[Chatbot] Loaded {len(self.responses)} Q&A pairs from cache')
                return True
            except Exception as e:
                print(f'[Chatbot] Cache load failed: {e}, rebuilding...')

        print('[Chatbot] Building TF-IDF index from CSV (this may take a minute)...')
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(csv_path, encoding='latin-1', on_bad_lines='skip')

        # Drop rows with missing values
        df = df.dropna(subset=['Patient', 'Doctor'])

        # Remove very short or unhelpful doctor responses
        df = df[df['Doctor'].str.len() > 50]

        # Remove duplicate patient questions (keep first)
        df = df.drop_duplicates(subset=['Patient'], keep='first')

        # Combine Description + Patient for better matching
        df['query_text'] = (
            df['Description'].fillna('').apply(_clean_text) + ' ' +
            df['Patient'].fillna('').apply(_clean_text)
        )

        # Filter out empty queries
        df = df[df['query_text'].str.strip().str.len() > 10]

        print(f'[Chatbot] Processing {len(df)} Q&A pairs...')

        # Build TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(df['query_text'].values)
        self.responses = df['Doctor'].values.tolist()
        self.descriptions = df['Description'].fillna('').values.tolist()

        # Cache to disk
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        joblib.dump(self.tfidf_matrix, MATRIX_PATH)
        joblib.dump({
            'responses': self.responses,
            'descriptions': self.descriptions,
        }, RESPONSES_PATH)

        self.ready = True
        print(f'[Chatbot] Index built and cached — {len(self.responses)} Q&A pairs ready')
        return True

    def get_response(self, user_query, top_k=3, threshold=0.08):
        """
        Find the most relevant doctor response for a user query.
        Returns: (response_text, confidence, matched_topic)
        """
        if not self.ready:
            return None, 0.0, None

        cleaned = _clean_text(user_query)
        if not cleaned:
            return None, 0.0, None

        # Vectorize user query
        query_vec = self.vectorizer.transform([cleaned])

        # Compute cosine similarity against all stored queries
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        best_idx = top_indices[0]
        best_score = similarities[best_idx]

        if best_score < threshold:
            return None, best_score, None

        response = self.responses[best_idx]
        topic = self.descriptions[best_idx] if best_idx < len(self.descriptions) else ''

        return response, float(best_score), topic

    def get_cancer_response(self, user_query, user_context=None):
        """
        Get a response with cancer-context awareness.
        1. First check simple if/then rules
        2. Then try TF-IDF retrieval
        3. Then cancer-specific personalized response
        4. Finally a helpful general response
        """
        first_name = ''
        if user_context:
            first_name = user_context.get('first_name', '')

        # Step 1: Check simple if/then rules first
        simple = get_simple_response(user_query, first_name)
        if simple:
            return simple

        # Step 2: Check if query is cancer/screening/family-history related
        cancer_keywords = [
            'cancer', 'tumor', 'tumour', 'lump', 'biopsy', 'mammogram',
            'pap smear', 'psa', 'colonoscopy', 'screening', 'hereditary',
            'family history', 'genetic', 'brca', 'risk score', 'prevention',
            'chemotherapy', 'radiation', 'oncolog', 'metastas', 'malignant',
            'benign', 'breast', 'cervical', 'prostate', 'colorectal', 'colon',
        ]

        query_lower = user_query.lower()
        is_cancer_query = any(kw in query_lower for kw in cancer_keywords)

        # If cancer-related and we have user context, build a personalized answer
        if is_cancer_query and user_context:
            personalized = self._build_cancer_response(user_query, user_context)
            return personalized

        # Step 3: Try TF-IDF retrieval for medical questions
        response, confidence, topic = self.get_response(user_query)

        if response and confidence >= 0.08:
            cleaned = self._clean_response(response)
            return cleaned

        # Step 4: Helpful general response (no error message)
        return self._general_fallback(user_query, user_context)

    def _build_cancer_response(self, query, context):
        """Build a cancer-focused response using user's risk data."""
        parts = []
        query_lower = query.lower()

        first_name = context.get('first_name', 'there')
        gender = (context.get('gender') or '').lower()
        scores = context.get('risk_scores', {})
        father_h = context.get('father_history', [])
        mother_h = context.get('mother_history', [])
        grand_h = context.get('grand_history', [])

        # Family history interpretation
        if 'family history' in query_lower or 'mother' in query_lower or 'father' in query_lower or 'hereditary' in query_lower:
            parts.append(f"Hello {first_name}, that's an important question about your hereditary cancer risk. Let me explain based on your family history.")

            if mother_h:
                mother_cancers = ', '.join(c.replace('_', ' ') for c in mother_h)
                parts.append(f"Your mother's side has a history of: {mother_cancers}. As a first-degree relative, this is a significant risk factor. First-degree family history (parent) typically doubles your risk for the same cancer type, according to major epidemiological studies.")

            if father_h:
                father_cancers = ', '.join(c.replace('_', ' ') for c in father_h)
                parts.append(f"Your father's side has a history of: {father_cancers}. This first-degree history means your risk is elevated — for example, prostate cancer risk increases by 2.5x with a father's history (Kicinski et al., British Journal of Cancer 2011).")

            if grand_h:
                grand_cancers = ', '.join(c.replace('_', ' ') for c in grand_h)
                parts.append(f"Your grandparents had: {grand_cancers}. Second-degree relatives contribute a moderate risk increase (roughly 50% of the first-degree multiplier).")

            if not father_h and not mother_h and not grand_h:
                parts.append("Based on your profile, you haven't reported any family cancer history. This is reassuring — your baseline risk is closer to the general population average.")

        # Risk score explanation
        if 'risk score' in query_lower or 'risk' in query_lower or 'percentage' in query_lower:
            parts.append(f"Here are your current hereditary cancer risk scores:")
            for cancer_type in ['breast_cancer', 'cervical_cancer', 'prostate_cancer', 'colorectal_cancer']:
                val = scores.get(cancer_type)
                if val is not None:
                    name = cancer_type.replace('_', ' ').title()
                    level = 'HIGH' if val >= 60 else 'MODERATE' if val >= 40 else 'LOW'
                    parts.append(f"- {name}: {val}% ({level})")

        # Screening recommendations
        if 'screening' in query_lower or 'mammogram' in query_lower or 'test' in query_lower or 'check' in query_lower:
            parts.append("Based on your risk profile, here are my screening recommendations:")
            if gender != 'male':
                breast_risk = scores.get('breast_cancer', 0)
                if breast_risk and breast_risk >= 40:
                    parts.append("- Breast: Mammogram ANNUALLY starting now, given your elevated risk. Discuss BRCA genetic testing with your doctor at Princess Marina Hospital or a private facility in Gaborone.")
                else:
                    parts.append("- Breast: Mammogram every 1-2 years from age 40. Self-exams monthly.")
                parts.append("- Cervical: Pap smear every 3 years — available at government clinics across Botswana. HPV vaccination if eligible.")
            if gender != 'female':
                prostate_risk = scores.get('prostate_cancer', 0)
                if prostate_risk and prostate_risk >= 40:
                    parts.append("- Prostate: PSA test ANNUALLY from age 40 given your family history and elevated risk.")
                else:
                    parts.append("- Prostate: Discuss PSA testing with your doctor from age 50 (or 40 with family history).")
            colorectal_risk = scores.get('colorectal_cancer', 0)
            if colorectal_risk and colorectal_risk >= 40:
                parts.append("- Colorectal: Colonoscopy recommended from age 40 (earlier than standard) given your risk. Available at Princess Marina Hospital.")
            else:
                parts.append("- Colorectal: Colonoscopy every 10 years from age 45. Increase dietary fibre, reduce red meat.")

        # Prevention plan
        if 'prevention' in query_lower or 'prevent' in query_lower or 'reduce' in query_lower or 'lower' in query_lower or 'plan' in query_lower:
            parts.append("Here are evidence-based cancer prevention steps personalised for you:")
            parts.append("1. Diet: Increase morogo (leafy greens), sorghum, beans, and vegetables. These are rich in fibre and antioxidants that protect against colorectal cancer.")
            parts.append("2. Exercise: Aim for at least 150 minutes of moderate activity per week. This reduces breast cancer risk by up to 25%.")
            parts.append("3. Limit alcohol: Even moderate drinking increases breast and colorectal cancer risk.")
            parts.append("4. Avoid tobacco: Smoking increases risk for nearly all cancer types by 40% or more.")
            parts.append("5. Stay on schedule with your screenings — early detection is the single most important factor.")

        # If no specific sub-topic matched, give a general cancer response
        if not parts:
            # Try TF-IDF for the cancer query
            response, confidence, topic = self.get_response(query)
            if response and confidence >= 0.08:
                cleaned = self._clean_response(response)
                parts.append(cleaned)
            else:
                parts.append(f"Hello {first_name}, thank you for your question about cancer. Based on your health profile, I can help you understand your risk scores, explain how your family history affects your risk, recommend appropriate screenings, or suggest prevention strategies. Could you be more specific about what you'd like to know?")

        parts.append("Remember, these risk scores help you take proactive steps — they are not a diagnosis. For any concerns, please consult a doctor at Princess Marina Hospital (Gaborone) or Nyangabgwe Referral Hospital (Francistown).")

        return '\n\n'.join(parts)

    def _clean_response(self, response):
        """Clean up doctor response text."""
        if not response:
            return ''
        # Remove common unhelpful endings
        response = re.sub(r'For further (information|doubts|queries) consult.*?-->.*$', '', response, flags=re.IGNORECASE)
        response = re.sub(r'For more information consult.*?-->.*$', '', response, flags=re.IGNORECASE)
        response = re.sub(r'Revert (back )?(with|to).*$', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\(attachment removed.*?\)', '', response, flags=re.IGNORECASE)
        response = response.strip()
        if response and not response[-1] in '.!?':
            response += '.'
        return response

    def _general_fallback(self, query, context=None):
        """Helpful, topic-aware response when no good TF-IDF match is found."""
        first_name = ''
        if context:
            first_name = context.get('first_name', '')

        name = first_name if first_name else 'there'
        query_lower = query.lower()

        # Try to give a relevant response based on detected topic
        topic_responses = {
            ('headache', 'head hurts', 'migraine'): (
                f"{name}, headaches can have many causes including stress, dehydration, tension, "
                f"or underlying conditions. Here's what I recommend:\n\n"
                f"1. **Stay hydrated** — drink at least 2 litres of water daily\n"
                f"2. **Rest** in a dark, quiet room if the headache is severe\n"
                f"3. **Track triggers** — note food, sleep, and stress patterns\n"
                f"4. **See a doctor** if headaches are frequent, sudden and severe, or accompanied by vision changes, fever, or stiff neck\n\n"
                f"Over-the-counter painkillers like paracetamol can help, but avoid overuse."
            ),
            ('diabetes', 'blood sugar', 'insulin', 'glucose high'): (
                f"{name}, managing blood sugar is crucial for overall health and cancer prevention. "
                f"Diabetes and high blood sugar are linked to increased cancer risk. Here are key steps:\n\n"
                f"1. **Monitor glucose regularly** — fasting normal is 3.9-5.5 mmol/L\n"
                f"2. **Diet** — reduce refined sugars, increase fibre-rich foods like morogo and sorghum\n"
                f"3. **Exercise** — 150 minutes of moderate activity per week helps regulate blood sugar\n"
                f"4. **Medication** — take prescribed medications consistently\n\n"
                f"Log your glucose readings in the Vitals section so I can track your trends."
            ),
            ('blood pressure', 'hypertension', 'bp high', 'bp low'): (
                f"{name}, blood pressure management is important. Normal BP is below 120/80 mmHg. "
                f"High blood pressure can increase your risk of various health complications.\n\n"
                f"1. **Reduce sodium** — limit salt intake to less than 5g per day\n"
                f"2. **Exercise** — regular moderate activity helps lower BP naturally\n"
                f"3. **Manage stress** — try deep breathing or meditation\n"
                f"4. **Limit alcohol** — excessive drinking raises blood pressure\n\n"
                f"Log your BP in the Vitals section to track trends over time."
            ),
            ('sleep', 'insomnia', 'cant sleep', "can't sleep", 'tired'): (
                f"{name}, quality sleep is essential for health and cancer prevention. "
                f"Poor sleep is linked to weakened immunity and increased cancer risk.\n\n"
                f"1. **Consistent schedule** — sleep and wake at the same time daily\n"
                f"2. **Limit screen time** — avoid devices 1 hour before bed\n"
                f"3. **Cool, dark room** — optimal sleep temperature is 18-20°C\n"
                f"4. **Avoid caffeine** after 2pm and heavy meals before bed\n\n"
                f"If insomnia persists for more than 2 weeks, consult a healthcare provider."
            ),
            ('stress', 'anxiety', 'worried', 'mental health', 'depressed', 'depression'): (
                f"{name}, mental health is just as important as physical health. "
                f"Chronic stress can weaken your immune system and affect cancer risk.\n\n"
                f"1. **Physical activity** — exercise releases endorphins that reduce stress\n"
                f"2. **Talk to someone** — a friend, family member, or counsellor\n"
                f"3. **Breathing exercises** — try 4-7-8 breathing (inhale 4s, hold 7s, exhale 8s)\n"
                f"4. **Limit news/social media** if it's causing anxiety\n\n"
                f"In Botswana, you can reach out to Lifeline Botswana or visit your nearest clinic for mental health support."
            ),
            ('exercise', 'workout', 'fitness', 'physical activity'): (
                f"{name}, regular exercise is one of the most powerful cancer prevention tools. "
                f"It reduces breast cancer risk by up to 25% and colorectal cancer by 20%.\n\n"
                f"1. **Aim for 150 minutes** of moderate activity per week (brisk walking, cycling)\n"
                f"2. **Strength training** — 2 sessions per week helps maintain healthy weight\n"
                f"3. **Start small** — even 10-minute walks make a difference\n"
                f"4. **Stay consistent** — regular activity is more beneficial than occasional intense workouts\n\n"
                f"Your current exercise level is noted in your profile. I can help you set goals!"
            ),
            ('diet', 'nutrition', 'food', 'eat', 'eating'): (
                f"{name}, diet plays a major role in cancer prevention. Here are evidence-based recommendations:\n\n"
                f"1. **Increase fibre** — morogo, sorghum, beans, and vegetables protect against colorectal cancer\n"
                f"2. **Limit red meat** — reduce to less than 500g per week; avoid processed meats entirely\n"
                f"3. **Eat colourful fruits & vegetables** — aim for 5 servings daily (rich in antioxidants)\n"
                f"4. **Limit alcohol** — even moderate drinking increases breast and colorectal cancer risk\n"
                f"5. **Avoid sugary drinks** — linked to obesity and increased cancer risk\n\n"
                f"Traditional Batswana foods like morogo, sorghum, and phane are excellent cancer-protective foods!"
            ),
            ('pain', 'hurts', 'ache', 'sore'): (
                f"{name}, I'm sorry to hear you're in pain. Pain can have many causes, and it's important to "
                f"pay attention to persistent or unexplained pain.\n\n"
                f"**When to see a doctor immediately:**\n"
                f"- Pain that persists for more than 2 weeks\n"
                f"- Pain that wakes you from sleep\n"
                f"- Pain accompanied by unexplained weight loss\n"
                f"- Pain with swelling or a lump\n\n"
                f"For immediate relief, rest the affected area and try over-the-counter painkillers. "
                f"Could you describe where the pain is so I can give more specific guidance?"
            ),
        }

        for keywords, response in topic_responses.items():
            if any(kw in query_lower for kw in keywords):
                return response

        return (
            f"Hello {name}! That's a great question. While I specialise in hereditary cancer risk, "
            f"I'm happy to help with general health topics too. Here are some things I can assist with:\n\n"
            f"- **Your cancer risk scores** — Ask 'What are my risk scores?'\n"
            f"- **Family history impact** — Ask 'How does my family history affect my risk?'\n"
            f"- **Screening schedule** — Ask 'What screenings should I get?'\n"
            f"- **Prevention tips** — Ask 'How can I reduce my cancer risk?'\n"
            f"- **Symptoms & health questions** — Describe your symptoms and I'll do my best to help.\n\n"
            f"Feel free to ask me anything!"
        )


# Singleton instance
_chatbot_instance = None


def get_chatbot():
    """Get or create the singleton chatbot instance."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = MedicalChatbot()
    return _chatbot_instance


def init_chatbot():
    """Initialize the chatbot (call on app startup)."""
    bot = get_chatbot()
    return bot.build_index()


def chat_respond(user_query, user_context=None, system_prompt=None, chat_history=None):
    """
    Main entry point: get a response for a user query.

    Flow:
      1. Simple if/then rules for greetings (fast, no API call needed)
      2. Google Gemini API for intelligent responses (if API key set)
      3. TF-IDF retrieval from 257K medical Q&A pairs (fallback)
      4. Template-based cancer response using user context
    """
    first_name = ''
    if user_context:
        first_name = user_context.get('first_name', '')

    # Step 1: Check simple greetings first (instant, no API needed)
    simple = get_simple_response(user_query, first_name)
    if simple:
        return simple

    # Step 2: Try Gemini API for an intelligent response
    if system_prompt:
        gemini_reply = _call_gemini(system_prompt, user_query, chat_history)
        if gemini_reply:
            return gemini_reply

    # Step 3: Fall back to local TF-IDF + cancer context engine
    bot = get_chatbot()
    return bot.get_cancer_response(user_query, user_context)
