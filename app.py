"""
AI Career Advisor - Streamlit App
File: streamlit_ai_career_advisor.py

Description:
- Modern, aesthetic Streamlit app for hackathon MVP
- Accepts resume (PDF/DOCX) OR manual skill selection
- Extracts skills using keyword matching (and spaCy if available)
- Recommends top-3 job roles (fit score) using careers JSON
- Shows skill gap and maps missing skills to resources from resources JSON

Notes / Setup:
1. Create a project folder and place these files in it:
   - streamlit_ai_career_advisor.py  (this file)
   - careers_large.json              (your large careers dataset)
   - resources.json                  (your resources dataset)

2. Create and activate a virtualenv (Windows PowerShell):
   python -m venv venv
   venv\Scripts\Activate

3. Install required packages:
   pip install -r requirements.txt

   Suggested requirements.txt content:
   streamlit
   pandas
   pdfplumber
   python-docx
   spacy

   OPTIONAL (for better NLP matching):
   pip install sentence-transformers

4. Run:
   streamlit run streamlit_ai_career_advisor.py

"""

import streamlit as st
import json
import pandas as pd
import os
import re
from pathlib import Path

# Optional imports handled gracefully
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# --------------------------- Helpers ---------------------------

@st.cache_data
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text_from_pdf(uploaded_file):
    if pdfplumber is None:
        return ""
    text = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text.append(page_text)
    except Exception as e:
        st.warning(f"PDF parsing error: {e}")
    return "\n".join(text)


def extract_text_from_docx(uploaded_file):
    if docx is None:
        return ""
    try:
        doc = docx.Document(uploaded_file)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        return "\n".join(fullText)
    except Exception as e:
        st.warning(f"DOCX parsing error: {e}")
        return ""


def normalize_text(t: str):
    if not t:
        return ""
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s+\-#./]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_skills_from_text(text: str, skills_list: list):
    """Extract skills by fuzzy matching / simple keyword detection.
    Returns a sorted list of matched skills (case-insensitive).
    """
    if not text:
        return []
    text_norm = normalize_text(text)
    found = set()

    # exact word matching and phrase matching
    for skill in skills_list:
        skill_norm = normalize_text(skill)
        # match as phrase or word
        if not skill_norm:
            continue
        # escape special chars
        pattern = r"\b" + re.escape(skill_norm) + r"\b"
        if re.search(pattern, text_norm):
            found.add(skill)

    # If spaCy available, also check noun chunks / entities for skill-like tokens
    if nlp is not None and len(found) < 5:  # try to enrich results a bit
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            chunk_text = normalize_text(chunk.text)
            if chunk_text in [normalize_text(s) for s in skills_list]:
                # map back to original skill casing
                matched = [s for s in skills_list if normalize_text(s) == chunk_text]
                if matched:
                    found.add(matched[0])

    return sorted(found)


def compute_fit_scores(selected_skills: list, careers: dict):
    """Compute fit score for each role and missing skills list.
    Returns list of tuples: (category, role, score, missing_skills, required_skills)
    """
    results = []
    selected_set = set(selected_skills)
    for category, roles in careers.items():
        for role, req_skills in roles.items():
            total = len(req_skills)
            if total == 0:
                score = 0
            else:
                matched = len([s for s in req_skills if s in selected_set])
                score = round((matched / total) * 100, 2)
            missing = [s for s in req_skills if s not in selected_set]
            results.append((category, role, score, missing, req_skills))
    # sort and return top hits (caller chooses how many)
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
    return results_sorted


# --------------------------- UI ---------------------------

st.set_page_config(page_title="AI Career Advisor", layout="wide", initial_sidebar_state="expanded")

# Load data files
DATA_DIR = Path(".")
CAREERS_FILE = DATA_DIR / "careers_large.json"
RESOURCES_FILE = DATA_DIR / "resources.json"

if not CAREERS_FILE.exists():
    st.error("careers_large.json not found in project folder. Please add your careers dataset file.")
    st.stop()
if not RESOURCES_FILE.exists():
    st.error("resources.json not found in project folder. Please add your resources dataset file.")
    st.stop()

careers = load_json(str(CAREERS_FILE))
resources = load_json(str(RESOURCES_FILE))

# Flatten skills from careers
all_skills = sorted(list({skill for roles in careers.values() for skill in roles.values() for skill in (roles if isinstance(roles, dict) else [])}))
# Note: the above flattening expects careers to be {category: {role: [skills]}}
# But users may have different structure; let's create robust flattening:

def flatten_careers(careers_dict):
    flat = {}
    for category, roles in careers_dict.items():
        if isinstance(roles, dict):
            for role, skills in roles.items():
                flat.setdefault(role, {"category": category, "skills": skills})
        else:
            # fallback if roles is list
            pass
    return flat

careers_flat = flatten_careers(careers)
all_skills = sorted({s for v in careers_flat.values() for s in v["skills"]})

# Sidebar - Input options
with st.sidebar:
    st.title("⚙️ Options")
    st.write("Choose input method and tweak settings")
    input_mode = st.radio("Input method:", ["Manual Skills", "Upload Resume (PDF/DOCX)"])
    num_recommend = st.slider("Number of job recommendations:", 1, 10, 3)
    min_fit_filter = st.slider("Show roles with fit >=:", 0, 100, 0)
    show_resources = st.checkbox("Show learning resources for missing skills", value=True)
    st.markdown("---")
    st.write("**App Tips:**")
    st.write("• For best results, select as many relevant skills as you have.")
    st.write("• Resume parsing uses keyword matching; manual cleanup may help.")

# Main layout
st.title("🎯 AI Career Advisor — Hackathon Prototype")
st.markdown("A lightweight career advisor: pick your skills or upload a resume, get top job role matches, gap analysis, and resources.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Your Skills")
    if input_mode == "Manual Skills":
        selected = st.multiselect("Select your skills (type to search):", all_skills, default=None)
        extracted_skills = selected
    else:
        uploaded_file = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])
        extracted_text = ""
        if uploaded_file is not None:
            file_type = uploaded_file.type
            if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith('.pdf'):
                if pdfplumber is None:
                    st.warning("pdfplumber not installed—install via pip install pdfplumber to enable PDF parsing. Falling back to no parsing.")
                else:
                    extracted_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.lower().endswith('.docx'):
                if docx is None:
                    st.warning("python-docx not installed—install via pip install python-docx to enable DOCX parsing.")
                else:
                    extracted_text = extract_text_from_docx(uploaded_file)
            else:
                st.warning("Unsupported file type")

            if extracted_text:
                st.subheader("Parsed Resume Preview")
                with st.expander("Show parsed text (first 2000 chars)"):
                    st.write(extracted_text[:2000])
                # extract skills from text
                extracted_skills = extract_skills_from_text(extracted_text, all_skills)
                st.success(f"Detected {len(extracted_skills)} skills from resume")
                st.write(extracted_skills)
            else:
                extracted_skills = []
        else:
            extracted_skills = []

    # allow manual add-on to parsed skills
    st.markdown("---")
    st.write("Add / Edit detected skills")
    manual_add = st.multiselect("Add more skills (optional)", all_skills, default=None)
    final_skills = sorted(list(set(extracted_skills + manual_add)))

    st.markdown("**Final Skill Profile:**")
    if final_skills:
        st.write(", ".join(final_skills))
    else:
        st.info("No skills selected yet — select manually or upload a resume.")

with col2:
    st.header("Recommendations")
    if not final_skills:
        st.info("Please provide skills to see recommendations.")
    else:
        # compute fit scores
        results = compute_fit_scores(final_skills, careers)
        # filter and slice
        results = [r for r in results if r[2] >= min_fit_filter]
        top_results = results[:num_recommend]

        for idx, (category, role, score, missing, required) in enumerate(top_results, start=1):
            box = st.container()
            with box:
                st.subheader(f"{idx}. {role}  —  {score}% fit")
                st.write(f"**Category:** {category}")
                # show required skills with badges
                cols = st.columns([1, 4])
                with cols[0]:
                    st.metric(label="Fit", value=f"{score}%")
                with cols[1]:
                    st.write("**Required Skills:**")
                    skill_chips = []
                    for s in required:
                        if s in final_skills:
                            st.write(f"✅ {s}")
                        else:
                            st.write(f"⚪ {s}")

                if missing:
                    st.write("---")
                    st.write("**Missing skills (gap analysis):**")
                    st.write(", ".join(missing))
                    if show_resources:
                        st.write("**Recommended Resources:**")
                        for m in missing:
                            if m in resources:
                                # resources[m] is expected to be a dict/list structure
                                r = resources[m]
                                # print a compact view
                                if isinstance(r, dict):
                                    for k, v in r.items():
                                        if isinstance(v, list):
                                            # show top 2 from each category for brevity
                                            for item in v[:3]:
                                                st.write(f"- [{k}] {item}")
                                        else:
                                            st.write(f"- [{k}] {v}")
                                elif isinstance(r, list):
                                    for item in r[:5]:
                                        st.write(f"- {item}")
                                elif isinstance(r, str):
                                    st.write(f"- {r}")
                            else:
                                # fallback search links
                                st.write(f"- Quick search for {m}: https://www.google.com/search?q={m.replace(' ', '+')}")
                else:
                    st.success("No missing skills — great fit! Consider leveling up with advanced projects or leadership experience.")

        st.markdown("---")
        st.write("**Want a downloadable plan?**")
        if st.button("Export Plan (CSV)"):
            # create a dataframe of top results
            rows = []
            for (category, role, score, missing, required) in top_results:
                rows.append({
                    "role": role,
                    "category": category,
                    "fit_score": score,
                    "missing_skills": ", ".join(missing),
                    "required_skills": ", ".join(required)
                })
            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV", data=csv, file_name="career_plan.csv", mime='text/csv')

# Footer / Extra
st.markdown("---")
col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b:
    st.write("Built for hackathon — AI Career Advisor prototype. For a production system, integrate job-market APIs, add embedding-based matching, and include richer user profiles.")
    st.write("If you want, I can: generate a Streamlit Cloud deployment guide, prepare the GitHub repo structure, or add a GPT-powered chatbot tab next.")

# End

