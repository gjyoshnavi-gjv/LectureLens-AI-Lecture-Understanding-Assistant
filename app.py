import streamlit as st
import ollama
from pypdf import PdfReader
import json
import re
import speech_recognition as sr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LectureLens",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🎓 LectureLens")
st.subheader("AI Lecture Understanding Assistant")

st.write(
    "Upload your lecture PDF and use AI to understand, "
    "practice and prepare for your examination."
)

st.divider()


# ============================================================
# CONSTANTS
# ============================================================

MODEL_NAME = "llama3.2:latest"


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "quiz_data": None,
    "quiz_type": None,
    "quiz_submitted": False,
    "quiz_score": 0,
    "quiz_percentage": 0,
    "quiz_results": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# AI HELPER
# ============================================================

def ask_ollama(prompt, json_format=False):

    try:

        if json_format:

            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                format="json"
            )

        else:

            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

        return response["message"]["content"]

    except Exception as e:

        return f"ERROR: {e}"


# ============================================================
# CREATE TEXT CHUNKS
# ============================================================

def create_chunks(text, chunk_size=1200, overlap=200):

    cleaned_text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    chunks = []

    start = 0

    step = chunk_size - overlap

    while start < len(cleaned_text):

        end = start + chunk_size

        chunk = cleaned_text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += step

    return chunks


# ============================================================
# SELECT REPRESENTATIVE CHUNKS
# ============================================================

def select_chunks_for_generation(chunks, max_chunks=12):

    if not chunks:
        return []

    if len(chunks) <= max_chunks:
        return chunks

    selected = []

    for i in range(max_chunks):

        index = int(
            i * (len(chunks) - 1) /
            (max_chunks - 1)
        )

        selected.append(chunks[index])

    return selected


# ============================================================
# RETRIEVE RELEVANT PDF CHUNKS
# ============================================================

def retrieve_relevant_chunks(
    question,
    chunks,
    number_of_chunks=4
):

    if not chunks:
        return []

    documents = chunks + [question]

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            documents
        )

        similarities = cosine_similarity(
            matrix[-1],
            matrix[:-1]
        ).flatten()

        best_indexes = similarities.argsort()[
            -number_of_chunks:
        ][::-1]

        relevant_chunks = []

        for index in best_indexes:

            if similarities[index] > 0:

                relevant_chunks.append(
                    chunks[index]
                )

        if relevant_chunks:

            return relevant_chunks

        return chunks[:number_of_chunks]

    except Exception:

        return chunks[:number_of_chunks]


# ============================================================
# ANSWER QUESTION FROM PDF
# ============================================================

def answer_from_pdf(question, chunks):

    relevant_chunks = retrieve_relevant_chunks(
        question,
        chunks,
        number_of_chunks=4
    )

    if not relevant_chunks:

        return (
            "This information is not available "
            "in the uploaded lecture."
        )

    context = "\n\n".join(
        relevant_chunks
    )

    prompt = f"""
You are LectureLens, an AI lecture assistant.

Answer the student's question using ONLY the
provided lecture content.

STRICT RULES:

1. Use ONLY the provided lecture content.
2. Do NOT use outside knowledge.
3. Do NOT invent information.
4. Do NOT guess.
5. If the answer cannot be found in the lecture,
   say exactly:

"This information is not available in the uploaded lecture."

6. Give a clear and understandable answer.
7. Keep the answer directly related to the question.
8. If the lecture contains steps, explain them using
   numbered steps.
9. If the lecture contains a definition, explain it clearly.
10. Do not mention retrieval, chunks or context.

LECTURE CONTENT:

{context}

STUDENT QUESTION:

{question}
"""

    result = ask_ollama(prompt)

    if result.startswith("ERROR:"):

        return (
            "❌ Ollama error.\n\n"
            "Please make sure Ollama is running and "
            "llama3.2:latest is installed."
        )

    return result


# ============================================================
# GENERATE SECTION-BY-SECTION CONTENT
# ============================================================

def generate_from_chunks(
    chunks,
    instruction,
    heading
):

    selected_chunks = select_chunks_for_generation(
        chunks,
        max_chunks=12
    )

    results = []

    for index, chunk in enumerate(
        selected_chunks,
        start=1
    ):

        prompt = f"""
You are LectureLens, an AI college lecture assistant.

Use ONLY the lecture section provided below.

{instruction}

STRICT RULES:

1. Use only information present in this section.
2. Do not use outside knowledge.
3. Do not invent information.
4. Do not add unrelated examples.
5. Explain information clearly.
6. Keep the content useful for examination preparation.

LECTURE SECTION:

{chunk}
"""

        result = ask_ollama(prompt)

        if not result.startswith("ERROR:"):

            results.append(
                f"## {heading} - Section {index}\n\n"
                f"{result}"
            )

    return "\n\n---\n\n".join(results)


# ============================================================
# UPLOAD LECTURE PDF
# ============================================================

st.header("📖 Upload Lecture PDF")

uploaded_file = st.file_uploader(
    "Choose your lecture PDF",
    type=["pdf"]
)


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

transcript = ""
reader = None

if uploaded_file is not None:

    try:

        reader = PdfReader(uploaded_file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                transcript += (
                    page_text + "\n"
                )

        if transcript.strip():

            st.success(
                "✅ Lecture PDF uploaded successfully!"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.info(
                    f"📄 Pages: {len(reader.pages)}"
                )

            with col2:

                st.info(
                    f"📝 Characters: {len(transcript):,}"
                )

        else:

            st.warning(
                "⚠️ This PDF does not contain readable text."
            )

    except Exception as e:

        st.error(
            f"❌ Error reading PDF: {e}"
        )


# ============================================================
# CREATE SEARCHABLE CHUNKS
# ============================================================

chunks = []

if transcript.strip():

    chunks = create_chunks(
        transcript
    )

    st.success(
        f"🧠 Lecture processed into "
        f"{len(chunks)} searchable sections."
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

if transcript.strip():

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "📝 Summary",
            "📌 Key Points",
            "💬 Ask Questions",
            "📚 Definitions",
            "❓ Quiz",
            "📝 Exam Questions",
            "📊 Dashboard"
        ]
    )


    # ========================================================
    # TAB 1 - SUMMARY
    # ========================================================

    with tab1:

        st.header("📝 Lecture Summary")

        st.write(
            "Generate a clear, topic-wise summary "
            "from the uploaded PDF."
        )

        if st.button(
            "📝 Generate Summary",
            key="summary_button"
        ):

            with st.spinner(
                "🧠 Reading the lecture and preparing summary..."
            ):

                summary_instruction = """
Create a clear and understandable summary.

Organize the content topic-wise.

Include:
- Introduction
- Main concepts
- Important details
- Processes or steps
- Classifications
- Important facts
- Conclusion

Do not simply copy the lecture.
Explain the important information in student-friendly language.
"""

                summary = generate_from_chunks(
                    chunks,
                    summary_instruction,
                    "Lecture Summary"
                )

                if summary:

                    st.success(
                        "✅ Summary generated successfully!"
                    )

                    st.markdown(
                        "# 📝 Lecture Summary"
                    )

                    st.markdown(summary)

                else:

                    st.error(
                        "❌ Could not generate summary."
                    )


    # ========================================================
    # TAB 2 - KEY POINTS
    # ========================================================

    with tab2:

        st.header("📌 Important Key Points")

        st.write(
            "Extract important concepts, facts, "
            "steps and classifications from the PDF."
        )

        if st.button(
            "📌 Generate Key Points",
            key="keypoints_button"
        ):

            with st.spinner(
                "🔎 Finding important points..."
            ):

                keypoint_instruction = """
Extract important key points.

DO NOT write another long summary.

Focus on:
- Important concepts
- Important facts
- Important definitions
- Important steps
- Classifications
- Formulas if present
- Important comparisons

Use bullet points and organize them topic-wise.
"""

                key_points = generate_from_chunks(
                    chunks,
                    keypoint_instruction,
                    "Important Key Points"
                )

                if key_points:

                    st.success(
                        "✅ Key points generated!"
                    )

                    st.markdown(key_points)

                else:

                    st.error(
                        "❌ Could not generate key points."
                    )


    # ========================================================
    # TAB 3 - ASK QUESTIONS
    # ========================================================

    with tab3:

        st.header("💬 Ask Questions")

        st.write(
            "Ask questions related to your lecture. "
            "Answers are generated only from the uploaded PDF."
        )

        question = st.text_input(
            "⌨️ Type your question",
            placeholder="Example: What is software risk?",
            key="lecture_question"
        )

        if st.button(
            "🔍 Get Answer",
            key="question_button"
        ):

            if question.strip():

                with st.spinner(
                    "🔎 Searching the lecture..."
                ):

                    answer = answer_from_pdf(
                        question.strip(),
                        chunks
                    )

                st.success(
                    "💡 Answer from the Lecture"
                )

                st.markdown(answer)

            else:

                st.warning(
                    "⚠️ Please type a question."
                )


        st.divider()

        st.subheader(
            "🎤 Ask using your voice"
        )

        st.write(
            "Speak your question and LectureLens "
            "will find the answer from the PDF."
        )

        if st.button(
            "🎤 Start Voice Input",
            key="voice_button"
        ):

            try:

                recognizer = sr.Recognizer()

                st.info(
                    "🎤 Listening... Please speak your question."
                )

                with sr.Microphone() as source:

                    recognizer.adjust_for_ambient_noise(
                        source,
                        duration=1
                    )

                    audio = recognizer.listen(
                        source,
                        timeout=10,
                        phrase_time_limit=15
                    )

                with st.spinner(
                    "🧠 Understanding your question..."
                ):

                    voice_text = (
                        recognizer.recognize_google(
                            audio
                        )
                    )

                with st.spinner(
                    "🔎 Finding the answer..."
                ):

                    voice_answer = answer_from_pdf(
                        voice_text,
                        chunks
                    )

                st.success(
                    "💡 Answer from the Lecture"
                )

                st.markdown(
                    voice_answer
                )

            except sr.WaitTimeoutError:

                st.warning(
                    "⚠️ No speech detected. Please try again."
                )

            except sr.UnknownValueError:

                st.warning(
                    "⚠️ Sorry, I could not understand your voice."
                )

            except sr.RequestError:

                st.error(
                    "❌ Speech recognition service is unavailable."
                )

            except Exception as e:

                st.error(
                    f"❌ Microphone error: {e}"
                )


    # ========================================================
    # TAB 4 - DEFINITIONS
    # ========================================================

    with tab4:

        st.header("📚 Important Definitions")

        st.write(
            "Find important technical terms and definitions "
            "from the lecture."
        )

        if st.button(
            "📚 Generate Definitions",
            key="definitions_button"
        ):

            with st.spinner(
                "📚 Finding important definitions..."
            ):

                definition_instruction = """
Find important technical terms and concepts
that are actually present in the lecture.

For each important term provide:

### Term Name

**Definition:**
Definition based only on the lecture.

**Simple Explanation:**
Easy student-friendly explanation based only on the lecture.

Do not create definitions for concepts that are not present.
Do not repeat terms.
"""

                definitions = generate_from_chunks(
                    chunks,
                    definition_instruction,
                    "Important Definitions"
                )

                if definitions:

                    st.success(
                        "✅ Definitions generated!"
                    )

                    st.markdown(
                        "# 📚 Important Definitions"
                    )

                    st.markdown(definitions)

                else:

                    st.error(
                        "❌ Could not generate definitions."
                    )


    # ========================================================
    # TAB 5 - QUIZ
    # ========================================================

    with tab5:

        st.header("❓ Lecture Quiz")

        st.write(
            "Test your understanding using questions "
            "generated only from the uploaded PDF."
        )

        quiz_type = st.selectbox(
            "🎯 Choose Quiz Type",
            [
                "Multiple Choice",
                "True / False",
                "Fill in the Blanks"
            ],
            key="quiz_type_select"
        )

        number_of_questions = st.selectbox(
            "🔢 Number of Questions",
            [5, 10, 15],
            key="quiz_number_select"
        )


        # ====================================================
        # GENERATE QUIZ
        # ====================================================

        if st.button(
            "🚀 Generate Quiz",
            key="generate_quiz_button"
        ):

            with st.spinner(
                "🧠 Creating quiz from different parts of the PDF..."
            ):

                selected_chunks = select_chunks_for_generation(
                    chunks,
                    max_chunks=12
                )

                questions_per_chunk = max(
                    1,
                    (
                        number_of_questions
                        + len(selected_chunks)
                        - 1
                    )
                    // len(selected_chunks)
                )

                all_questions = []


                for chunk in selected_chunks:

                    remaining = (
                        number_of_questions
                        - len(all_questions)
                    )

                    if remaining <= 0:
                        break

                    count = min(
                        questions_per_chunk,
                        remaining
                    )


                    # ========================================
                    # MCQ
                    # ========================================

                    if quiz_type == "Multiple Choice":

                        prompt = f"""
You are LectureLens.

Create {count} high-quality multiple-choice
questions from ONLY the lecture section below.

Return ONLY valid JSON.

FORMAT:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": 0,
            "explanation": "Explanation based on lecture."
        }}
    ]
}}

RULES:

- Exactly {count} questions.
- Exactly 4 options per question.
- answer must be 0, 1, 2 or 3.
- Only one option is correct.
- No duplicate questions.
- Questions must come directly from the lecture.
- Do not use outside knowledge.

LECTURE SECTION:

{chunk}
"""


                    # ========================================
                    # TRUE / FALSE
                    # ========================================

                    elif quiz_type == "True / False":

                        prompt = f"""
You are LectureLens.

Create {count} True / False questions
from ONLY the lecture section below.

Return ONLY valid JSON.

FORMAT:

{{
    "questions": [
        {{
            "question": "Statement",
            "answer": true,
            "explanation": "Explanation based on lecture."
        }}
    ]
}}

RULES:

- Exactly {count} questions.
- answer must be true or false.
- Statements must be clear.
- Do not create ambiguous questions.
- No duplicate questions.
- Use only the lecture.

LECTURE SECTION:

{chunk}
"""


                    # ========================================
                    # FILL IN BLANKS
                    # ========================================

                    else:

                        prompt = f"""
You are LectureLens.

Create {count} fill-in-the-blank questions
from ONLY the lecture section below.

Return ONLY valid JSON.

FORMAT:

{{
    "questions": [
        {{
            "question": "The ______ is an important concept.",
            "answer": "correct answer",
            "explanation": "Explanation based on lecture."
        }}
    ]
}}

RULES:

- Exactly {count} questions.
- Use exactly one ______ blank.
- Answer must come directly from lecture.
- No ambiguous questions.
- No duplicate questions.
- Use only lecture information.

LECTURE SECTION:

{chunk}
"""


                    try:

                        response_text = ask_ollama(
                            prompt,
                            json_format=True
                        )

                        if response_text.startswith(
                            "ERROR:"
                        ):
                            continue

                        data = json.loads(
                            response_text
                        )

                        generated = data.get(
                            "questions",
                            []
                        )

                        if isinstance(
                            generated,
                            list
                        ):

                            all_questions.extend(
                                generated
                            )

                    except Exception:

                        continue


                # ============================================
                # CLEAN AND VALIDATE QUESTIONS
                # ============================================

                valid_questions = []

                seen_questions = set()

                for q in all_questions:

                    if not isinstance(q, dict):
                        continue

                    question_text = str(
                        q.get(
                            "question",
                            ""
                        )
                    ).strip()

                    normalized = re.sub(
                        r"\s+",
                        " ",
                        question_text.lower()
                    )

                    if not question_text:
                        continue

                    if normalized in seen_questions:
                        continue

                    if quiz_type == "Multiple Choice":

                        options = q.get(
                            "options",
                            []
                        )

                        answer = q.get(
                            "answer"
                        )

                        if (
                            not isinstance(
                                options,
                                list
                            )
                            or len(options) != 4
                        ):
                            continue

                        try:

                            answer = int(answer)

                        except Exception:

                            continue

                        if answer not in [
                            0, 1, 2, 3
                        ]:
                            continue

                        q["answer"] = answer

                    elif quiz_type == "True / False":

                        answer = q.get(
                            "answer"
                        )

                        if isinstance(
                            answer,
                            str
                        ):

                            if answer.lower() == "true":

                                q["answer"] = True

                            elif answer.lower() == "false":

                                q["answer"] = False

                            else:

                                continue

                        elif isinstance(
                            answer,
                            bool
                        ):

                            pass

                        else:

                            continue

                    else:

                        answer = str(
                            q.get(
                                "answer",
                                ""
                            )
                        ).strip()

                        if not answer:
                            continue

                        q["answer"] = answer

                    if not q.get(
                        "explanation"
                    ):

                        q["explanation"] = (
                            "Based on the uploaded lecture."
                        )

                    seen_questions.add(
                        normalized
                    )

                    valid_questions.append(
                        q
                    )


                valid_questions = (
                    valid_questions[
                        :number_of_questions
                    ]
                )


                if not valid_questions:

                    st.error(
                        "⚠️ No quiz questions were generated. "
                        "Please try again."
                    )

                else:

                    st.session_state.quiz_data = {
                        "questions": valid_questions
                    }

                    st.session_state.quiz_type = (
                        quiz_type
                    )

                    st.session_state.quiz_submitted = (
                        False
                    )

                    st.session_state.quiz_score = 0

                    st.session_state.quiz_percentage = 0

                    st.session_state.quiz_results = []

                    st.success(
                        f"✅ {len(valid_questions)} "
                        f"quiz questions generated!"
                    )


        # ====================================================
        # DISPLAY QUIZ
        # ====================================================

        if st.session_state.quiz_data is not None:

            questions = (
                st.session_state.quiz_data[
                    "questions"
                ]
            )

            active_quiz_type = (
                st.session_state.quiz_type
            )

            st.divider()

            st.subheader(
                "🧠 Answer the Questions"
            )

            st.info(
                f"📚 Quiz Type: {active_quiz_type} | "
                f"Questions: {len(questions)}"
            )

            user_answers = []

            for i, q in enumerate(
                questions
            ):

                st.markdown(
                    f"### Question {i + 1}"
                )

                st.write(
                    q.get(
                        "question",
                        "Question unavailable."
                    )
                )


                if active_quiz_type == (
                    "Multiple Choice"
                ):

                    answer = st.radio(
                        "Choose your answer:",
                        q["options"],
                        key=f"mcq_answer_{i}"
                    )


                elif active_quiz_type == (
                    "True / False"
                ):

                    answer = st.radio(
                        "Choose your answer:",
                        ["True", "False"],
                        key=f"tf_answer_{i}"
                    )


                else:

                    answer = st.text_input(
                        "✏️ Your Answer:",
                        key=f"blank_answer_{i}"
                    )


                user_answers.append(
                    answer
                )

                st.divider()


            # =================================================
            # SUBMIT QUIZ
            # =================================================

            if st.button(
                "✅ Submit Quiz",
                key="submit_quiz_button"
            ):

                score = 0

                results = []

                for i, q in enumerate(
                    questions
                ):

                    correct = q.get(
                        "answer"
                    )

                    explanation = q.get(
                        "explanation",
                        "Explanation not available."
                    )


                    # =========================================
                    # MCQ CHECK
                    # =========================================

                    if active_quiz_type == (
                        "Multiple Choice"
                    ):

                        correct_index = int(
                            correct
                        )

                        correct_answer = (
                            q["options"][
                                correct_index
                            ]
                        )

                        is_correct = (
                            user_answers[i]
                            == correct_answer
                        )


                    # =========================================
                    # TRUE FALSE CHECK
                    # =========================================

                    elif active_quiz_type == (
                        "True / False"
                    ):

                        correct_bool = (
                            bool(correct)
                        )

                        correct_answer = (
                            "True"
                            if correct_bool
                            else "False"
                        )

                        is_correct = (
                            user_answers[i]
                            == correct_answer
                        )


                    # =========================================
                    # FILL BLANK CHECK
                    # =========================================

                    else:

                        correct_answer = str(
                            correct
                        ).strip()

                        user_answer = str(
                            user_answers[i]
                        ).strip()

                        is_correct = (
                            user_answer.lower()
                            == correct_answer.lower()
                        )


                    if is_correct:

                        score += 1


                    results.append(
                        {
                            "question": i + 1,
                            "correct": is_correct,
                            "user_answer": user_answers[i],
                            "correct_answer": correct_answer,
                            "explanation": explanation
                        }
                    )


                total = len(
                    questions
                )

                percentage = (
                    score / total * 100
                    if total > 0
                    else 0
                )

                st.session_state.quiz_submitted = (
                    True
                )

                st.session_state.quiz_score = (
                    score
                )

                st.session_state.quiz_percentage = (
                    percentage
                )

                st.session_state.quiz_results = (
                    results
                )

                st.rerun()


            # =================================================
            # SHOW RESULT
            # =================================================

            if st.session_state.quiz_submitted:

                score = (
                    st.session_state.quiz_score
                )

                percentage = (
                    st.session_state.quiz_percentage
                )

                results = (
                    st.session_state.quiz_results
                )

                st.divider()

                st.header(
                    "📊 Quiz Result"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "🎯 Score",
                        f"{score} / {len(questions)}"
                    )

                with col2:

                    st.metric(
                        "📈 Percentage",
                        f"{percentage:.0f}%"
                    )


                if percentage >= 80:

                    st.balloons()

                    st.success(
                        "🌟 Excellent! You have a strong "
                        "understanding of the lecture."
                    )

                elif percentage >= 60:

                    st.success(
                        "👍 Good job! Keep practicing "
                        "to improve further."
                    )

                elif percentage >= 40:

                    st.warning(
                        "📚 Review the lecture and try again."
                    )

                else:

                    st.error(
                        "📖 You need more practice. "
                        "Review the lecture carefully."
                    )


                st.subheader(
                    "📋 Question-wise Results"
                )


                for result in results:

                    if result["correct"]:

                        st.success(
                            f"✅ Question "
                            f"{result['question']}: Correct"
                        )

                    else:

                        st.error(
                            f"❌ Question "
                            f"{result['question']}: Wrong"
                        )

                        st.write(
                            f"Your answer: "
                            f"{result['user_answer']}"
                        )

                        st.write(
                            f"✅ Correct answer: "
                            f"{result['correct_answer']}"
                        )

                    st.info(
                        f"💡 Explanation: "
                        f"{result['explanation']}"
                    )

                    st.divider()


                if st.button(
                    "🔄 Retake Quiz",
                    key="retake_quiz_button"
                ):

                    st.session_state.quiz_submitted = (
                        False
                    )

                    st.session_state.quiz_score = 0

                    st.session_state.quiz_percentage = 0

                    st.session_state.quiz_results = []

                    st.rerun()


    # ========================================================
    # TAB 6 - EXAM QUESTIONS
    # ========================================================

    with tab6:

        st.header(
            "📝 Important Exam Questions"
        )

        st.write(
            "Generate examination-oriented questions "
            "using only the uploaded lecture."
        )

        exam_type = st.selectbox(
            "🎯 Choose Question Category",
            [
                "All Important Questions",
                "2-Mark Questions",
                "5-Mark Questions",
                "10-Mark Questions",
                "Most Important Questions"
            ],
            key="exam_type_select"
        )

        exam_number = st.selectbox(
            "🔢 Number of Questions",
            [5, 10, 15],
            key="exam_number_select"
        )


        if st.button(
            "📝 Generate Exam Questions",
            key="exam_button"
        ):

            with st.spinner(
                "🧠 Preparing exam questions..."
            ):


                if exam_type == "2-Mark Questions":

                    question_instruction = """
Generate short-answer questions suitable for 2 marks.

Focus on:
- Definitions
- Short explanations
- Important terms
- Differences
- Facts
- Simple concepts.
"""


                elif exam_type == "5-Mark Questions":

                    question_instruction = """
Generate questions suitable for 5 marks.

Focus on:
- Explain concepts
- Compare concepts
- Classifications
- Processes
- Important steps
- Descriptive questions.
"""


                elif exam_type == "10-Mark Questions":

                    question_instruction = """
Generate questions suitable for 10 marks.

Focus on:
- Detailed explanations
- Important processes
- Detailed comparisons
- Classifications
- Architecture
- Structure
- Long-answer concepts.
"""


                elif exam_type == "Most Important Questions":

                    question_instruction = """
Generate the most important examination questions.

Prioritize:
- Major concepts
- Important definitions
- Important processes
- Important classifications
- Topics emphasized in the lecture.
"""


                else:

                    question_instruction = """
Generate a balanced list of important examination
questions from different topics.

Include:
- Definitions
- Concepts
- Comparisons
- Processes
- Classifications
- Important details.
"""


                selected_chunks = (
                    select_chunks_for_generation(
                        chunks,
                        max_chunks=12
                    )
                )

                questions_per_chunk = max(
                    1,
                    (
                        exam_number
                        + len(selected_chunks)
                        - 1
                    )
                    // len(selected_chunks)
                )

                all_exam_questions = []


                for chunk in selected_chunks:

                    remaining = (
                        exam_number
                        - len(all_exam_questions)
                    )

                    if remaining <= 0:
                        break

                    count = min(
                        questions_per_chunk,
                        remaining
                    )

                    prompt = f"""
You are LectureLens, a college examination
preparation assistant.

Use ONLY the lecture section below.

Generate {count} examination questions.

CATEGORY:

{exam_type}

INSTRUCTIONS:

{question_instruction}

STRICT RULES:

1. Use only information in the lecture.
2. Do not use outside knowledge.
3. Do not invent topics.
4. Do not provide answers.
5. Do not provide explanations.
6. Generate questions only.
7. Do not repeat questions.
8. Use clear examination language.

Return ONLY valid JSON.

FORMAT:

{{
    "questions": [
        "Question 1?",
        "Question 2?"
    ]
}}

LECTURE SECTION:

{chunk}
"""


                    try:

                        response_text = ask_ollama(
                            prompt,
                            json_format=True
                        )

                        if response_text.startswith(
                            "ERROR:"
                        ):
                            continue

                        data = json.loads(
                            response_text
                        )

                        generated = data.get(
                            "questions",
                            []
                        )

                        if isinstance(
                            generated,
                            list
                        ):

                            for q in generated:

                                if isinstance(
                                    q,
                                    str
                                ):

                                    all_exam_questions.append(
                                        q.strip()
                                    )

                    except Exception:

                        continue


                # ============================================
                # REMOVE DUPLICATES
                # ============================================

                unique_questions = []

                seen = set()

                for q in all_exam_questions:

                    normalized = re.sub(
                        r"\s+",
                        " ",
                        q.lower()
                    ).strip()

                    if (
                        normalized
                        and normalized not in seen
                    ):

                        seen.add(
                            normalized
                        )

                        unique_questions.append(
                            q
                        )


                unique_questions = (
                    unique_questions[
                        :exam_number
                    ]
                )


                if unique_questions:

                    st.success(
                        f"✅ {len(unique_questions)} "
                        f"{exam_type} generated!"
                    )

                    st.markdown(
                        f"# 📝 {exam_type}"
                    )

                    for i, question_text in enumerate(
                        unique_questions,
                        start=1
                    ):

                        st.markdown(
                            f"**{i}. {question_text}**"
                        )

                else:

                    st.error(
                        "❌ Could not generate exam questions. "
                        "Please try again."
                    )


    # ========================================================
    # TAB 7 - STUDY DASHBOARD
    # ========================================================

    with tab7:

        st.header(
            "📊 Study Dashboard"
        )

        st.write(
            "Track your lecture understanding "
            "and quiz performance."
        )

        st.divider()


        # ====================================================
        # LECTURE INFORMATION
        # ====================================================

        st.subheader(
            "📄 Lecture Information"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "📄 PDF Pages",
                len(reader.pages)
            )

        with col2:

            st.metric(
                "🧠 Study Sections",
                len(chunks)
            )

        with col3:

            st.metric(
                "📝 Characters",
                f"{len(transcript):,}"
            )


        st.divider()


        # ====================================================
        # QUIZ PERFORMANCE
        # ====================================================

        st.subheader(
            "❓ Quiz Performance"
        )

        col1, col2, col3 = st.columns(3)


        with col1:

            if (
                st.session_state.quiz_data
                is not None
            ):

                total_questions = len(
                    st.session_state.quiz_data[
                        "questions"
                    ]
                )

            else:

                total_questions = 0

            st.metric(
                "🎯 Quiz Score",
                f"{st.session_state.quiz_score} "
                f"/ {total_questions}"
            )


        with col2:

            st.metric(
                "📈 Percentage",
                f"{st.session_state.quiz_percentage:.0f}%"
            )


        with col3:

            percentage = (
                st.session_state.quiz_percentage
            )

            if percentage >= 80:

                performance = "Excellent 🌟"

            elif percentage >= 60:

                performance = "Good 👍"

            elif percentage >= 40:

                performance = "Needs Practice 📚"

            else:

                performance = "Needs Improvement 📖"

            st.metric(
                "🏆 Performance",
                performance
            )


        st.divider()


        # ====================================================
        # LEARNING PROGRESS
        # ====================================================

        st.subheader(
            "📚 Learning Progress"
        )

        percentage = (
            st.session_state.quiz_percentage
        )

        st.progress(
            min(
                100,
                max(
                    0,
                    int(percentage)
                )
            )
        )

        st.write(
            f"Your current learning progress based on "
            f"quiz performance is **{percentage:.0f}%**."
        )


        if percentage >= 80:

            st.success(
                "🌟 Excellent progress! "
                "Keep revising important concepts."
            )

        elif percentage >= 60:

            st.info(
                "👍 Good progress! "
                "Review incorrectly answered questions."
            )

        elif percentage >= 40:

            st.warning(
                "📚 You are making progress. "
                "Revise the lecture and retake the quiz."
            )

        else:

            st.error(
                "📖 Spend more time reviewing the lecture "
                "before taking the quiz again."
            )


        st.divider()


        # ====================================================
        # STUDY STATUS
        # ====================================================

        st.subheader(
            "🎓 Study Status"
        )


        if st.session_state.quiz_data is None:

            st.info(
                "📌 Generate and complete a quiz "
                "to see your study performance."
            )

        elif not st.session_state.quiz_submitted:

            st.info(
                "📝 Quiz generated. Complete the quiz "
                "and click Submit Quiz."
            )

        else:

            st.success(
                "✅ Quiz completed! "
                "Your dashboard has been updated."
            )


        st.divider()


        # ====================================================
        # FEATURES COMPLETED
        # ====================================================

        st.subheader(
            "✅ LectureLens Features"
        )

        feature_col1, feature_col2 = st.columns(2)

        with feature_col1:

            st.write("✅ PDF Upload")
            st.write("✅ PDF Text Extraction")
            st.write("✅ Smart PDF Chunking")
            st.write("✅ TF-IDF Search")
            st.write("✅ AI Summary")
            st.write("✅ Key Points")
            st.write("✅ PDF Question Answering")

        with feature_col2:

            st.write("✅ Voice Questions")
            st.write("✅ Definitions")
            st.write("✅ MCQ Quiz")
            st.write("✅ True / False Quiz")
            st.write("✅ Fill-in-the-Blanks")
            st.write("✅ Exam Questions")
            st.write("✅ Study Dashboard")


# ============================================================
# NO PDF MESSAGE
# ============================================================

else:

    st.info(
        "👆 Upload a lecture PDF to start using LectureLens."
    )

    st.markdown(
        """
### 🎓 What LectureLens can do

📄 **Upload Lecture PDF**  
Upload your college lecture material.

📝 **Generate Summary**  
Get a clear topic-wise summary.

📌 **Key Points**  
Find important concepts and facts.

💬 **Ask Questions**  
Ask questions and get answers from your PDF.

🎤 **Voice Questions**  
Ask questions using your microphone.

📚 **Definitions**  
Find important definitions from the lecture.

❓ **Quiz**  
Practice using MCQ, True/False and Fill-in-the-Blanks.

📝 **Exam Questions**  
Generate 2-mark, 5-mark, 10-mark and important questions.

📊 **Dashboard**  
Track your quiz performance and study progress.
"""
    )