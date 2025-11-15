import streamlit as st
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.graph_objects as go
import spacy
from spacy.matcher import PhraseMatcher

# --- 1. Database Definitions ---

# Defines the 10 skill categories and their associated keywords.
# This structure is used for the radar chart calculation.
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
# Used as the target for cosine similarity matching.
JOBS_DB = {
    # IT Tech
    "Data Scientist": "python scikit-learn pandas tensorflow sql",
    "Frontend Developer": "javascript react figma ui ux",
    "Backend Developer": "java nodejs sql mysql mongodb docker",
    "AI/ML Engineer": "python tensorflow keras pytorch scikit-learn",
    "Cloud Engineer": "aws azure gcp docker kubernetes",
    
    # Art/Admin
    "Graphic Designer": "photoshop illustrator adobe creative suite figma graphic design",
    "Office Administrator": "microsoft office excel powerpoint word typing data entry office management",
    
    # Science/Medical
    "Lab Technician": "lab techniques hplc spectroscopy chromatography",
    "Registered Nurse": "patient care nursing medical records hipaa cpr first aid",
    "Chemist": "chemistry organic chemistry r&d hplc spectroscopy",
    
    # Business/Law
    "Logistics Manager": "logistics supply chain fleet management warehouse sap excel",
    "Police Officer": "public safety criminal law investigation first aid cpr",
    
    # New Categories
    "Mechanical Engineer": "autocad cad mechanical engineering simulation matlab",
    "Teacher": "teaching curriculum design pedagogy classroom management",
    "Head Chef": "cooking kitchen management menu planning food safety baking"
}

# Simulated catalog of courses mapped to specific skills.
# Used to provide actionable recommendations for skill gaps.
COURSES_DB = {
    # IT Tech
    "python": "Complete Python Bootcamp: From Zero to Hero",
    "tensorflow": "Deep Learning with TensorFlow A-Z",
    "react": "Modern React with Redux",
    "sql": "The Complete SQL Bootcamp 2025",
    "aws": "AWS Certified Cloud Practitioner (CLF-C01)",
    "docker": "Docker & Kubernetes: The Complete Guide",
    "ui": "UI/UX Design Fundamentals",
    "figma": "Figma Masterclass for UI/UX Design",
    
    # Art/Admin
    "photoshop": "Adobe Photoshop Masterclass from Zero to Hero",
    "excel": "Advanced Excel for Business Analytics",
    "project management": "Google Project Management Professional Certificate",

    # Science/Medical
    "patient care": "Patient Care Technician (PCT) Certification",
    "chemistry": "Introduction to Organic Chemistry",
    "hplc": "High-Performance Liquid Chromatography (HPLC) Workshop",
    
    # Business/Law
    "logistics": "Supply Chain Management Fundamentals (SCM-101)",
    "public safety": "Public Safety and Crisis Intervention Training",
    "seo": "The Complete SEO Training Masterclass",
    
    # New Categories
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

# --- 1A. Data Normalization (REMOVED) ---
# The NORMALIZATION_MAP is no longer needed.
# SpaCy's PhraseMatcher will handle synonyms and aliases.


# --- 2. AI Core Model Initialization ---

# --- 2.1. CountVectorizer (Encoder Role) ---
# Initialize the CountVectorizer with the complete skill vocabulary.
# 'binary=True' creates a one-hot-encoded vector (skill present/absent).
# ROLE CHANGE: This is no longer the "extractor". It is now the "encoder"
# that turns our *found* skills into the correct vector for cosine similarity.
vectorizer = CountVectorizer(binary=True, vocabulary=SKILLS_DB)

# Pre-compute the skill vectors for all jobs in the database.
# This avoids re-calculating them on every run.
job_titles = list(JOBS_DB.keys())
job_skills_text = list(JOBS_DB.values())
job_vectors = vectorizer.fit_transform(job_skills_text)

# Get all feature names (skill names) from the vectorizer for later use.
all_skill_names = vectorizer.get_feature_names_out() 

# --- 2.2. SpaCy (Extractor Role) ---
# Load the small English model.
# Note: You may need to run `python -m spacy download en_core_web_sm`
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    st.error("SpaCy model 'en_core_web_sm' not found.")
    st.error("Please run: python -m spacy download en_core_web_sm")
    st.stop()

# Initialize the PhraseMatcher.
# attr='LOWER' makes matching case-insensitive.
skill_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

# Build the patterns for the PhraseMatcher.
# This replaces the old NORMALIZATION_MAP.
patterns = {}

# 1. Add skills from the main SKILLS_DB
for skill in SKILLS_DB:
    patterns[skill] = [nlp.make_doc(skill)]

# 2. Add synonyms and aliases
# The key (e.g., "scikit-learn") is the *standardized* skill name.
# The list contains all variations that should map to it.
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


# --- 3. Helper Function for Skill Gap Analysis ---

# This function encapsulates the logic for analyzing and displaying
# the skill gap for any given job title.
def display_skill_gap_analysis(job_title, user_found_skills):
    """
    Calculates and displays the skill gap analysis for a specific job.
    
    Args:
        job_title (str): The name of the job to analyze.
        user_found_skills (set): A set of skills found in the user's resume.
    """
    
    # Get the required skills for the specified job.
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
        
        # Loop through each missing skill and provide a specific recommendation.
        for skill in skills_you_need:
            if skill in COURSES_DB:
                # If a course is in our DB, recommend it.
                st.success(f"**[{skill.upper()}]** -> {COURSES_DB[skill]}")
            else:
                # If no course is found, provide the generic advice.
                st.warning(f"For **{skill}**: No specific course found. We recommend self-studying this skill.")

# --- 4. Streamlit UI Configuration ---

# Configure page settings for a wide layout and a custom title/icon.
st.set_page_config(layout="wide", page_title="SkillMatch AI", page_icon="📊")

# Inject custom CSS for enhanced visual styling.
st.markdown("""
<style>
/* Style the primary analysis button */
.stButton > button[kind="primary"] {
    height: 3em;
    width: 100%;
    font-size: 1.1em;
    font-weight: bold;
    border-radius: 10px;
    box-shadow: 0 4px 14px 0 rgba(0,118,255,0.39);
    transition: all 0.3s ease-in-out;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px 0 rgba(0,118,255,0.23);
    transform: translateY(-2px);
}

/* Style the st.metric component for job matching */
[data-testid="stMetric"] {
    background-color: #F0F2F6;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
[data-testid="stMetricLabel"] {
    font-size: 1.1em;
    font-weight: 600;
    color: #31333F;
}
[data-testid="stMetricValue"] {
    font-size: 2.5em;
    font-weight: 700;
    color: #0070f3;
}

/* Style the main container for analysis results */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    border: 1px solid #E6E6E6;
}

/* Make skill tags (code blocks) larger */
code {
    background-color: #E0EFFF;
    color: #004085;
    padding: .2em .4em;
    border-radius: 6px;
    font-weight: 600;
    font-size: 1.2em;
}
</style>
""", unsafe_allow_html=True)

# --- 5. Main Application Layout ---

st.title("SkillMatch AI – Your Smart Career Advisor")
st.markdown("Paste your resume (or LinkedIn profile) and we'll analyze your career path!")

# Initialize session state to hold the resume text.
if 'resume_input' not in st.session_state:
    st.session_state.resume_input = SAMPLE_RESUME

# Define the two-column layout: Input (col1) and Results (col2).
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Paste Your Resume (Plain Text)")
    resume_text = st.text_area("Résumé/CV Text", height=400, key="resume_input")

    # Add a multiselect box for users to choose target jobs.
    st.header("2. Select Target Jobs (Optional)")
    target_jobs = st.multiselect(
        "Select up to 3 jobs you are interested in:",
        options=job_titles,
        max_selections=3
    )
    
    st.write("") # Add some spacing
    
    # The analysis button is placed in the first column.
    analyze_button = st.button("Analyze My Career Path!", type="primary")

# --- 6. Analysis Execution ---

# The analysis only runs if the button is clicked.
if analyze_button:
    with col2:
        st.header("AI Analysis Results")
        
        # Handle the case of empty input.
        if not resume_text.strip():
            st.error("You didn't enter anything...")
            st.title("Your Recommended Position: Beggar")
            st.image("https://i.imgflip.com/1sl1m0.jpg", caption="Please enter *some* skills... (Meme)")
            st.stop() # Stop further execution

        # --- 6.1. Skill Extraction (with SpaCy) ---
        
        # 1. Process the resume text with SpaCy.
        # This replaces all the manual .lower() and .replace() calls.
        doc = nlp(resume_text)
        
        # 2. Run the PhraseMatcher on the processed doc.
        matches = skill_matcher(doc)
        
        # 3. Extract the *standardized* skill names (the match_id).
        # We use a set to automatically handle duplicates.
        user_found_skills = set()
        for match_id, start, end in matches:
            skill_name = nlp.vocab.strings[match_id]
            user_found_skills.add(skill_name)
        
        # --- 6.2. Skills Inventory & Radar Chart ---
        
        if not user_found_skills:
            st.warning("We couldn't find any relevant keywords in your resume. Please try adding more details.")
        else:
            st.subheader("Skills Inventory")
            st.info(f"We found {len(user_found_skills)} key skills in your resume:")
            # Display found skills using the styled 'code' tags.
            st.write(f"`{'` `'.join(user_found_skills)}`")

            st.subheader("Skills Radar Chart")

            # Calculate the user's score (as a percentage) for each skill category.
            categories = list(SKILLS_BY_CATEGORY.keys())
            scores = []
            for category in categories:
                category_skills = SKILLS_BY_CATEGORY[category]
                total_in_category = len(category_skills)
                # Use set intersection to find common skills
                user_has_in_category = user_found_skills.intersection(category_skills)
                
                if total_in_category > 0:
                    score = len(user_has_in_category) / total_in_category
                else:
                    score = 0
                scores.append(score * 100) # Convert to 0-100 scale

            # --- Plotly Chart Generation ---
            fig = go.Figure()

            # Add the first point to the end of the list to "close" the radar chart.
            categories_closed = categories + [categories[0]]
            scores_closed = scores + [scores[0]]

            fig.add_trace(go.Scatterpolar(
                r=scores_closed,
                theta=categories_closed,
                fill='toself',
                name='Your Skill Distribution',
                fillcolor='rgba(0,112,243,0.2)', # Fill color
                line=dict(color='#0070f3') # Line color
            ))

            # Set chart height and font sizes directly in the Plotly layout
            fig.update_layout(
                height=550, # Set chart height here
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickfont=dict(size=14), # Radial font size (0-100%)
                        showticklabels=True,
                        ticksuffix='%'
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=14) # Angular font size (Categories)
                    )
                ),
                showlegend=False,
                margin=dict(l=40, r=40, t=40, b=40),
                font=dict(size=12)
            )
            
            # Display the chart, stretching to the column width.
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()

            # --- 6.3. Job Match Calculation ---
            
            # CRITICAL STEP: Bridge SpaCy (Extractor) and CountVectorizer (Encoder)
            # 1. Convert the set of found skills back into a single string.
            found_skills_string = " ".join(user_found_skills)
            
            # 2. Use the *original* vectorizer to transform this string.
            # This creates a 'user_vector' that has the exact same dimensions
            # and vocabulary as the pre-computed 'job_vectors'.
            user_vector = vectorizer.transform([found_skills_string])
            
            # 3. Calculate cosine similarity.
            similarity_scores = cosine_similarity(user_vector, job_vectors)
            
            st.subheader("Top Job Matches (Based on your skills)")
            results = zip(job_titles, similarity_scores[0])
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

            # Display the top 3 matches in a 3-column metric layout.
            metric_cols = st.columns(3)
            for i, (job, score) in enumerate(sorted_results[:3]):
                with metric_cols[i]:
                    st.metric(label=job, value=f"{score*100:.1f}% Match")

            st.divider()

            # --- 6.4. Skill Gap Analysis ---
            
            # Group the gap analysis in a styled container.
            with st.container(border=True):
                st.subheader("Career Path Analysis")
                
                # --- Part A: Always analyze the #1 matched job ---
                top_job_title = sorted_results[0][0]
                display_skill_gap_analysis(top_job_title, user_found_skills)

                # --- Part B: Analyze user-selected target jobs ---
                if target_jobs:
                    st.divider() # Separate from the top match analysis
                    st.subheader("Target Job Analysis")
                    
                    for job in target_jobs:
                        # Avoid re-analyzing if the target job was the same as the top match
                        if job != top_job_title: 
                            display_skill_gap_analysis(job, user_found_skills)
                            # Add a small separator between target job analyses
                            st.write("---")
