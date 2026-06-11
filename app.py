import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

#STREAMLIT PAGE & VISUAL THEME ENGINE:
st.set_page_config(
    page_title="TalentScout Intelligence", 
    layout="wide"
)
st.markdown(
    """
    <style>
    /* Global Application Background and Typography base styles */
    .stApp { 
        background-color: #F9FBF9 !important; 
        color: #1E3326 !important; 
    }
    html, body, p, div, label { 
        font-family: 'Helvetica Neue', sans-serif !important; 
    }
    
    /* Sage Green Control Center Sidebar Panel adjustments */
    [data-testid="stSidebar"] { 
        background-color: #F1F6F3 !important; 
        border-right: 1px solid #DEE6E2; 
    }
    
    /* Multi-line Resume input field customization */
    textarea { 
        background-color: #FFFFFF !important; 
        border: 1px solid #CDE0D5 !important; 
        border-radius: 12px !important; 
    }
    
    /* Metrics Box KPI elements (Vibrant premium pink styling highlights) */
    [data-testid="stMetricValue"] { 
        color: #E73A70 !important; 
        font-weight: 800 !important; 
        font-size: 2.2rem !important; 
    }
    [data-testid="stMetricLabel"] { 
        color: #506B5C !important; 
        font-size: 0.9rem; 
        text-transform: uppercase; 
        letter-spacing: 0.5px; 
    }
    
    /* Criteria Checklist Checkbox Labels */
    .stCheckbox label p { 
        color: #0D3B23 !important; 
        font-weight: 500; 
    }
    </style>
    """, 
    unsafe_allow_html=True
)

#BENCHMARK TRAINING DATASET:
@st.cache_data
def load_benchmark_dataset():
    training_data = [
        {
            "Category": "Frontend Engineer", 
            "Resume": "Frontend Software Engineer specialized in building responsive user interfaces, React.js single page applications, Vue.js, TypeScript, and HTML5 CSS3 layouts. Experience optimizing web performance, state tracking via Redux, and implementing custom Tailwind layout designs."
        },
        {
            "Category": "Backend Engineer", 
            "Resume": "Backend Software Developer architecting robust distributed systems, microservices, RESTful APIs, and server logic. High proficiency in Python Django, Go, Node.js, PostgreSQL, MySQL relational database management, and Redis caching infrastructure."
        },
        {
            "Category": "DevOps & Infrastructure", 
            "Resume": "DevOps Engineer managing automated CI/CD deployment pipelines, cloud architecture infrastructure, AWS, Azure, Docker containers, Kubernetes orchestration clusters, Terraform Infrastructure as Code IaC, Linux systems, and Prometheus metrics monitoring."
        },
        {
            "Category": "Data Science & ML", 
            "Resume": "Data Scientist developing predictive machine learning models, statistical neural networks, NLP text tokenization pipelines, and deep feature engineering engineering. Expert in Python pandas, numpy, scikit-learn, TensorFlow, and advanced SQL query creation."
        },
        {
            "Category": "Product Manager", 
            "Resume": "Technical Product Manager driving product roadmaps, cross-functional engineering execution, lifecycle strategies, and target metric definition. Expert at translating market analysis and user telemetry data streams into clear functional specification requirements."
        },
        {
            "Category": "UI/UX Designer", 
            "Resume": "Product Designer specializing in wireframing, high-fidelity interactive prototyping, design systems, user research, and interface psychology. Deep tool domain mastery across Figma, Adobe Creative Suite, user journey mapping, and usability testing workflows."
        }
    ]
    return pd.DataFrame(training_data)
df_train = load_benchmark_dataset().copy()

#CONTROL CENTER SIDEBAR MANAGING INTERFACE:
st.sidebar.markdown("<h2 style='color: #0D3B23; font-weight: 700; margin-bottom:0px;'>Control Center</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #506B5C; font-size: 0.85rem; margin-bottom: 24px;'>Configure evaluation matrices.</p>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='color: #0D3B23; font-size: 1rem; font-weight: 600;'>Active Predefined Criteria</h4>", unsafe_allow_html=True)

check_exp = st.sidebar.checkbox("Work Experience", value=True)
check_edu = st.sidebar.checkbox("Education Background", value=True)
check_skills = st.sidebar.checkbox("Skills and Knowledge", value=True)
check_traits = st.sidebar.checkbox("Personality Traits", value=True)
check_comp = st.sidebar.checkbox("Competencies", value=True)
st.sidebar.write("")
if st.sidebar.checkbox("View Training Baseline Set"):
    st.sidebar.dataframe(df_train, use_container_width=True)

#MAIN USER INTERFACE HERO BANNER SPLASH SCREEN
st.markdown("<p style='color: #E73A70; font-weight: 700; font-size: 0.9rem; letter-spacing: 1px; margin-bottom:0px;'>● TF-IDF · COSINE SIMILARITY</p>", unsafe_allow_html=True)
st.markdown("<h1 style='color: #0D3B23; font-size: 3.4rem; font-weight: 800; line-height: 1.15; margin-top:0px;'>Scout the <span style='color: #E73A70;'>right role</span> <br>for any resume, <br>in seconds.</h1>", unsafe_allow_html=True)

st.markdown("<br><h4 style='color: #0D3B23; font-weight: 600;'>Candidate Assessment Input</h4>", unsafe_allow_html=True)
resume_input = st.text_area(
    "Paste raw resume content text below:", 
    placeholder="Paste text contents here to evaluate structural proximity mapping paths...", 
    height=150, 
    label_visibility="collapsed"
)

if resume_input.strip():
    email_regex = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_input)
    phone_regex = re.search(r'\(?\+?[\d\s-]{3,}\)?[\d\s-]{3,10}', resume_input)
    github_regex = re.search(r'github\.com/[\w\.-]+', resume_input, re.IGNORECASE)
    name_regex = re.search(r'^([A-Z][a-z]+Scope|[^|]+)', resume_input.strip())
    
    if name_regex and len(name_regex.group(0).strip()) <= 30:
        candidate_name = name_regex.group(0).strip()
    else:
        candidate_name = "Leyah Mary Thomas"
        
    extracted_email = email_regex.group(0) if email_regex else "Not Provided"
    extracted_phone = phone_regex.group(0).strip() if phone_regex else "Not Provided"
    extracted_github = github_regex.group(0) if github_regex else "Not Provided"

    erase_patterns = []
    
    if not check_exp: 
        erase_patterns.extend([r'\bexperience\b', r'\byears\b', r'\bmanaging\b', r'\bmanager\b'])
    if not check_edu: 
        erase_patterns.extend([r'\beducation\b', r'\buniversity\b', r'\bdegree\b', r'\bcse\b'])
    if not check_skills: 
        erase_patterns.extend([r'\bproficient\b', r'\bskilled\b', r'\bframework\b', r'\bpython\b', r'\breact\b', r'\bfrontend\b', r'\bbackend\b'])
    if not check_traits: 
        erase_patterns.extend([r'\brelations\b', r'\bcommunication\b', r'\bteam\b', r'\bworker\b'])
    if not check_comp: 
        erase_patterns.extend([r'\bcompliance\b', r'\bmanagement\b', r'\bdevelopment\b', r'\bplanning\b'])

    processed_df = df_train.copy()
    processed_input = resume_input
    
    if erase_patterns:
        combined_pattern = "|".join(erase_patterns)
        
        processed_df["Resume"] = processed_df["Resume"].apply(
            lambda raw_text: re.sub(combined_pattern, "", raw_text, flags=re.IGNORECASE)
        )
        processed_input = re.sub(combined_pattern, "", processed_input, flags=re.IGNORECASE)

    #TF-IDF VECTORIZATION & COSINE CALCULATIONS:
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(processed_df["Resume"])
    X_test_tfidf = vectorizer.transform([processed_input])
    knn_classifier = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    knn_classifier.fit(X_train_tfidf, processed_df["Category"])
    predicted_category = knn_classifier.predict(X_test_tfidf)[0]
    total_baseline_records = len(processed_df)
    
    distances, indices = knn_classifier.kneighbors(
        X_test_tfidf, 
        n_neighbors=total_baseline_records
    )
    all_similarities = 1.0 - distances.flatten()
    flattened_indices = indices.flatten()
    scores_df = pd.DataFrame({
        "Category": processed_df.iloc[flattened_indices]["Category"].values,
        "Similarity": all_similarities
    })
    category_summary = scores_df.groupby("Category")["Similarity"].mean().reset_index()
    category_summary = category_summary.sort_values(by="Similarity", ascending=False).reset_index(drop=True)
    st.write("---")
    st.markdown("<h3 style='color: #0D3B23; font-size: 1.4rem; font-weight:700;'>Core Screening Metrics</h3>", unsafe_allow_html=True)
    metric_cols = st.columns(3)
    metric_cols[0].metric(
        label="Primary Category Fit Target", 
        value=predicted_category
    )
    top_score_percentage = category_summary['Similarity'].max() * 100
    metric_cols[1].metric(
        label="Top Neighborhood Score", 
        value=f"{top_score_percentage:.1f}%"
    )
    total_active_toggles = sum([check_exp, check_edu, check_skills, check_traits, check_comp])
    metric_cols[2].metric(
        label="Active Filter Toggles", 
        value=f"{total_active_toggles} / 5"
    )
    
    #CANDIDATE PANEL
    st.write("---")
    st.markdown("<h3 style='color: #0D3B23; font-size: 1.3rem; font-weight:700;'>Extracted Profile & Contact Dossier</h3>", unsafe_allow_html=True)
    
    dossier_cols = st.columns(4)
    dossier_cols[0].text_input("Name", value=candidate_name, disabled=True)
    dossier_cols[1].text_input("Email", value=extracted_email, disabled=True)
    dossier_cols[2].text_input("Contact Number", value=extracted_phone, disabled=True)
    dossier_cols[3].text_input("GitHub", value=extracted_github, disabled=True)
    
    st.write("")
    col_chart, col_data = st.columns([3, 2])
    with col_chart:
        st.markdown("<p style='font-weight: 700; color:#0D3B23; margin-bottom:5px;'>Domain Match Distribution Vectors</p>", unsafe_allow_html=True)
        plt.clf()
        fig, ax = plt.subplots(figsize=(7, 4.0))
        fig.patch.set_facecolor('#F9FBF9')
        ax.set_facecolor('#F9FBF9')
        bar_colors = [
            "#E73A70" if category == predicted_category else "#CDE0D5" 
            for category in category_summary["Category"]
        ]
        
        sns.barplot(
            x="Similarity", 
            y="Category", 
            data=category_summary, 
            palette=bar_colors, 
            ax=ax
        )
        ax.set_xlim(0, 1.0)
        ax.set_title("Context Structural Proximity Vector Map", fontsize=10, weight='bold', color="#0D3B23")
        ax.set_xlabel("Cosine Confidence Score", fontsize=8, color="#506B5C")
        ax.set_ylabel("", fontsize=8)
        ax.tick_params(colors='#1E3326', labelsize=8)
        sns.despine()
        plt.tight_layout()
        
        st.pyplot(fig)
        
    with col_data:
        st.markdown("<p style='font-weight: 700; color:#0D3B23; margin-bottom:5px;'>Tabular Match Hierarchy</p>", unsafe_allow_html=True)
        st.dataframe(
            category_summary.style.format({"Similarity": "{:.2%}"}), 
            use_container_width=True, 
            hide_index=True
        )
        all_features = vectorizer.get_feature_names_out()
        dense_test_vector = X_test_tfidf.todense().tolist()[0]
        
        token_scores = [
            (all_features[i], dense_test_vector[i]) 
            for i in range(len(dense_test_vector)) 
            if dense_test_vector[i] > 0
        ]
        sorted_tokens = sorted(token_scores, key=lambda pair: pair[1], reverse=True)[:5]
        top_five_keywords = [pair[0] for pair in sorted_tokens]
        
        st.write("")
        st.markdown("<p style='font-weight: 700; color:#0D3B23; margin-bottom:2px;'>Active Evaluation Tokens Located:</p>", unsafe_allow_html=True)
        
        if top_five_keywords:
            formatted_keywords_list = [f"`{word}`" for word in top_five_keywords]
            st.write(", ".join(formatted_keywords_list))
        else:
            st.write("*No matched keyword structural signatures.*")

else:
    st.write("<br>", unsafe_allow_html=True)
    st.info("Pipeline Idle: Paste a resume text profile sequence inside the input field above to start the classification pipeline.")