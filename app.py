# --- 1. Imports ---
import streamlit as st
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.graph_objects as go
import spacy
from spacy.matcher import PhraseMatcher
import sys # Import sys to check executable path if needed

# --- 2. Database Definitions ---

# Defines the 10 skill categories and their associated keywords.
SKILLS_BY_CATEGORY = {
    'IT Tech': [
        'python', 'java', 'javascript', 'react', 'nodejs', 'angular',
        'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
        'sql', 'mysql', 'postgresql', 'mongodb', 'firebase',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes'
    ],
    'Art/Design': [
        'photoshop', 'illustrator', 'adobe creative suite', 'premiere pro', 
        '3d modeling', 'blender', 'figma', 'sketch', 'ui', 'ux', 'graphic design'
    ],
    'Admin/Office': [
        'microsoft office', 'excel', 'powerpoint', 'word', 'typing', 'data entry', 
        'sap', 'office management', 'administrative support', 'bookkeeping'
    ],
    'Medical/Healthcare': [
        'patient care', 'nursing', 'medical records', 'hipaa', 'anatomy', 
        'pharmacology', 'cpr', 'first aid', 'phlebotomy'
    ],
    'Business/Logistics': [
        'logistics', 'supply chain', 'fleet management', 'warehouse',
        'agile', 'scrum', 'project management', 'marketing', 'seo', 'sales'
    ],
    'Law/Safety': [
        'criminal law', 'investigation', 'public safety', 'surveillance',
        'compliance', 'legal research'
    ],
    'Physics & Engineering': [
        'physics', 'matlab', 'autocad', 'cad', 'simulation', 'mechanical engineering',
        'electrical engineering', 'robotics'
    ],
    'Chemistry & Lab Science': [
        'chemistry', 'organic chemistry', 'lab techniques', 'hplc', 
        'spectroscopy', 'r&d', 'chromatography', 'biochemistry'
    ],
    'Education': [
        'teaching', 'curriculum design', 'pedagogy', 'e-learning', 
        'classroom management', 'educational technology'
    ],
    'Culinary/Hospitality': [
        'cooking', 'food safety', 'menu planning', 'hospitality management',
        'baking', 'kitchen management', 'customer service'
    ]
}

# Flatten the category dictionary into a single list for the vectorizer vocabulary.
SKILLS_DB = [skill for skills_list in SKILLS_BY_CATEGORY.values() for skill in skills_list]

# Simulated database of job postings and their required skills.
JOBS_DB = {
    "Data Scientist": "python scikit-learn pandas tensorflow sql",
    "Frontend Developer": "javascript react figma ui ux",
    "Backend Developer": "java nodejs sql mysql mongodb docker",
    "AI/ML Engineer": "python tensorflow keras pytorch scikit-learn",
    "Cloud Engineer": "aws azure gcp docker kubernetes",
    "Graphic Designer": "photoshop illustrator adobe creative suite figma graphic design",
    "Office Administrator": "microsoft office excel powerpoint word typing data entry office management",
    "Lab Technician": "lab techniques hplc spectroscopy chromatography",
    "Registered Nurse": "patient care nursing medical records hipaa cpr first aid",
    "Chemist": "chemistry organic chemistry r&d hplc spectroscopy",
    "Logistics Manager": "logistics supply chain fleet management warehouse sap excel",
    "Police Officer": "public safety criminal law investigation first aid cpr",
    "Mechanical Engineer": "autocad cad mechanical engineering simulation matlab",
    "Teacher": "teaching curriculum design pedagogy classroom management",
    "Head Chef": "cooking kitchen management menu planning food safety baking"
}

# Simulated catalog of courses mapped to specific skills.
COURSES_DB = {
    "python": "Complete Python Bootcamp: From Zero to Hero",
    "tensorflow": "Deep Learning with TensorFlow A-Z",
    "react": "Modern React with Redux",
    "sql": "The Complete SQL Bootcamp 2025",
    "aws": "AWS Certified Cloud Practitioner (CLF-C01)",
    "docker": "Docker & Kubernetes: The Complete Guide",
    "ui": "UI/UX Design Fundamentals",
    "figma": "Figma Masterclass for UI/UX Design",
    "photoshop": "Adobe Photoshop Masterclass from Zero to Hero",
    "excel": "Advanced Excel for Business Analytics",
    "project management": "Google Project Management Professional Certificate",
    "patient care": "Patient Care Technician (PCT) Certification",
    "chemistry": "Introduction to Organic Chemistry",
    "hplc": "High-Performance Liquid Chromatography (HPLC) Workshop",
    "logistics": "Supply Chain Management Fundamentals (SCM-101)",
    "public safety": "Public Safety and Crisis Intervention Training",
    "seo": "The Complete SEO Training Masterclass",
    "autocad": "AutoCAD 2025: From Beginner to Advanced",
    "matlab": "MATLAB for Engineers and Scientists",
    "teaching": "Foundations of Teaching & Learning",
    "curriculum design": "Instructional Design & Curriculum Planning",
    "cooking": "Professional Culinary Arts Program",
    "food safety": "ServSafe Food Safety Manager Certification"
}

# Default text for the resume input box to facilitate demos.
SAMPLE_RESUME = """
John Doe - Software Engineer
Email: john.doe@email.com

Experience:
- Developed web applications using Python, JavaScript, and React.
- Managed databases with SQL and MongoDB.
- I have experience with sci-kit learn and node.js.
- Interested in AI and currently learning TensorFlow.

Skills:
- Languages: Python, JavaScript, SQL
- Frameworks: React, Node.js
- DB: MongoDB
- Other: matlab, cooking, ms excel
"""

# --- 3. AI Core Model Initialization ---

# --- 3.1. CountVectorizer (Encoder Role) ---
# This vectorizer's *only* role is to be the "encoder"
# that turns our final skill list into the correct vector for cosine similarity.
# Its vocabulary is fixed to our SKILLS_DB.
vectorizer = CountVectorizer(binary=True, vocabulary=SKILLS_DB)

# Pre-compute the skill vectors for all jobs in the database.
job_titles = list(JOBS_DB.keys())
job_skills_text = list(JOBS_DB.values())
job_vectors = vectorizer.fit_transform(job_skills_text)

# Get all feature names (skill names) from the vectorizer for later use.
all_skill_names = vectorizer.get_feature_names_out() 

# --- 3.2. SpaCy (Extractor Role) ---
# Load the small English model.
# This must be listed in requirements.txt as 'en_core_web_sm'
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    # This error block is crucial for deployment debugging
    st.error(f"SpaCy model 'en_core_web_sm' not found.")
    st.error(f"Current Python Executable: {sys.executable}")
    st.error("Please ensure 'en_core_web_sm' is in your requirements.txt")
    st.stop()

# Initialize the PhraseMatcher
skill_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

# Build the patterns for the PhraseMatcher.
# This replaces the old NORMALIZATION_MAP.
patterns = {}

# 1. Add skills from the main SKILLS_DB
for skill in SKILLS_DB:
    patterns[skill] = [nlp.make_doc(skill)]

# 2. Add synonyms and aliases
# The key (e.g., "scikit-learn") is the *standardized* skill name.
patterns["scikit-learn"].extend([
    nlp.make_doc("sklearn"),
    nlp.make_doc("sci-kit learn")
])
patterns["nodejs"].append(nlp.make_doc("node.js"))
patterns["aws"].append(nlp.make_doc("amazon web services"))
patterns["excel"].append(nlp.make_doc("ms excel"))
patterns["gcp"].append(nlp.make_doc("google cloud platform"))
patterns["ui"].extend([
    nlp.make_doc("ui design"),
    nlp.make_doc("user interface")
])
patterns["ux"].extend([
    nlp.make_doc("ux design"),
    nlp.make_doc("user experience")
])

# Add the patterns to the matcher
for skill_id, pattern_list in patterns.items():
    skill_matcher.add(skill_id, pattern_list)


# --- 4. Helper Function for Skill Gap Analysis ---

def display_skill_gap_analysis(job_title, user_found_skills):
    """
    Calculates and displays the skill gap analysis for a specific job.
    """
    
    # Get the required skills for the specified job from its vector
    job_skills_text = JOBS_DB[job_title]
    job_vector = vectorizer.transform([job_skills_text]).toarray()[0]
    job_required_skills = {all_skill_names[i] for i, v in enumerate(job_vector) if v > 0}

    # Calculate the difference (the skill gap).
    skills_you_need = job_required_skills - user_found_skills
    
    st.info(f"Analyzing the gap between you and the **{job_title}** role:")
    
    if not skills_you_need:
        st.success(f"Congratulations! You have all the key skills for **{job_title}**!")
    else:
        st.warning(f"You are missing the following **{len(skills_you_need)}** key skills:")
        for skill in skills_you_need:
            st.write(f"- **{skill}**")
            
        # Provide actionable course recommendations.
        st.subheader("Recommended Courses (Fill Your Gaps)")
        
        for skill in skills_you_need:
            if skill in COURSES_DB:
                st.success(f"**[{skill.upper()}]** -> {COURSES_DB[skill]}")
            else:
                st.warning(f"For **{skill}**: No specific course found. We recommend self-studying this skill.")

# --- 5. Streamlit UI Configuration ---

st.set_page_config(layout="wide", page_title="SkillMatch AI", page_icon="📊")

# Inject custom CSS for enhanced visual styling.
st.markdown("""
<style>
.stButton > button[kind="primary"] {
    height: 3em; width: 100%; font-size: 1.1em; font-weight: bold;
    border-radius: 10px;
    box-shadow: 0 4px 14px 0 rgba(0,118,255,0.39);
}
[data-testid="stMetric"] {
    background-color: #F0F2F6; border-radius: 10px;
    padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF; border-radius: 15px; padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 1px solid #E6E6E6;
}
code {
    background-color: #E0EFFF; color: #004085; padding: .2em .4em;
    border-radius: 6px; font-weight: 600; font-size: 1.2em;
}
</style>
""", unsafe_allow_html=True)

# --- 6. Main Application Layout ---

st.title("SkillMatch AI – Your Smart Career Advisor")
st.markdown("Paste your resume (or LinkedIn profile) and we'll analyze your career path!")

if 'resume_input' not in st.session_state:
    st.session_state.resume_input = SAMPLE_RESUME

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Paste Your Resume (Plain Text)")
    resume_text = st.text_area("Resume/CV Text", height=400, key="resume_input")

    st.header("2. Select Target Jobs (Optional)")
    target_jobs = st.multiselect(
        "Select up to 3 jobs you are interested in:",
        options=list(JOBS_DB.keys()),
        max_selections=3
    )
    
    st.write("") 
    analyze_button = st.button("Analyze My Career Path!", type="primary")

# --- 7. Analysis Execution ---

if analyze_button:
    with col2:
        st.header("AI Analysis Results")
        
        if not resume_text.strip():
            st.error("You didn't enter anything...")
            st.stop() 

        # --- 7.1. Skill Extraction (with SpaCy) ---
        
        # 1. Process the resume text with SpaCy.
        doc = nlp(resume_text)
        
        # 2. Run the PhraseMatcher on the processed doc.
        matches = skill_matcher(doc)
        
        # 3. Extract the *standardized* skill names (the match_id).
        user_found_skills = set()
        for match_id, start, end in matches:
            skill_name = nlp.vocab.strings[match_id]
            user_found_skills.add(skill_name)
        
        # --- 7.2. Skills Inventory & Radar Chart ---
        
        if not user_found_skills:
            st.warning("We couldn't find any relevant keywords in your resume.")
        else:
            st.subheader("Skills Inventory")
            st.info(f"We found {len(user_found_skills)} key skills in your resume:")
            st.write(f"`{'` `'.join(user_found_skills)}`")

            st.subheader("Skills Radar Chart")

            # Calculate the user's score (as a percentage) for each skill category.
            categories = list(SKILLS_BY_CATEGORY.keys())
            scores = []
            for category in categories:
                category_skills = SKILLS_BY_CATEGORY[category]
                total_in_category = len(category_skills)
                user_has_in_category = user_found_skills.intersection(category_skills)
                
                if total_in_category > 0:
                    score = len(user_has_in_category) / total_in_category
                else:
                    score = 0
                scores.append(score * 100) 

            # --- Plotly Chart Generation ---
            fig = go.Figure()
            categories_closed = categories + [categories[0]]
            scores_closed = scores + [scores[0]]

            fig.add_trace(go.Scatterpolar(
                r=scores_closed,
                theta=categories_closed,
                fill='toself',
                name='Your Skill Distribution',
                fillcolor='rgba(0,112,243,0.2)', 
                line=dict(color='#0070f3')
            ))

            fig.update_layout(
                height=550,
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=14), ticksuffix='%'),
                    angularaxis=dict(tickfont=dict(size=14))
                ),
                showlegend=False,
                margin=dict(l=40, r=40, t=40, b=40),
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.divider()

            # --- 7.3. Job Match Calculation ---
            
            # Bridge SpaCy (Extractor) and CountVectorizer (Encoder)
            # 1. Convert the set of found skills back into a single string.
            found_skills_string = " ".join(user_found_skills)
            
            # 2. Use the *original* vectorizer to transform this string.
            user_vector = vectorizer.transform([found_skills_string])
            
            # 3. Calculate cosine similarity.
            similarity_scores = cosine_similarity(user_vector, job_vectors)
            
            st.subheader("Top Job Matches (Based on your skills)")
            results = zip(job_titles, similarity_scores[0])
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

            # Display the top 3 matches
            metric_cols = st.columns(3)
            for i, (job, score) in enumerate(sorted_results[:3]):
                with metric_cols[i]:
                    st.metric(label=job, value=f"{score*100:.1f}% Match")

            st.divider()

            # --- 7.4. Skill Gap Analysis ---
            with st.container(border=True):
                st.subheader("Career Path Analysis")
                
                # Analyze the #1 matched job
                top_job_title = sorted_results[0][0]
                display_skill_gap_analysis(top_job_title, user_found_skills)

                # Analyze user-selected target jobs
                if target_jobs:
                    st.divider() 
                    st.subheader("Target Job Analysis")
                    
                    for job in target_jobs:
                        if job != top_job_title: 
                            display_skill_gap_analysis(job, user_found_skills)
                            st.write("---")

