# ================================
# AI Resume Screening System
# ================================

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ================================
# 1. Load Environment Variables
# ================================
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "resume-screening"

# ================================
# 2. Initialize LLM
# ================================
llm = ChatOpenAI(model="gpt-4o-mini")

# ================================
# 3. PROMPTS
# ================================

# --- Skill Extraction ---
extract_prompt = PromptTemplate.from_template("""
Extract the following from the resume:

1. Skills
2. Tools/Technologies
3. Years of Experience

Rules:
- Do NOT assume anything
- Only extract explicitly mentioned data

Return JSON format:
{{
  "skills": [],
  "tools": [],
  "experience": ""
}}

Resume:
{resume}
""")

# --- Matching ---
match_prompt = PromptTemplate.from_template("""
Compare the resume data with job description.

Identify:
- Matching skills
- Missing skills

Return JSON:
{{
  "matched_skills": [],
  "missing_skills": []
}}

Job Description:
{jd}

Resume Data:
{resume_data}
""")

# --- Scoring ---
score_prompt = PromptTemplate.from_template("""
Based on matching:

Give a score from 0 to 100.

Scoring Rules:
- High match → 80–100
- Partial match → 50–79
- Low match → 0–49

Return JSON:
{{
  "score": number
}}

Match Data:
{match_data}
""")

# --- Explanation ---
explain_prompt = PromptTemplate.from_template("""
Explain why the candidate received this score.

Include:
- Strengths
- Weaknesses
- Missing skills impact

Keep it concise.

Score:
{score}

Match Data:
{match_data}
""")

# ================================
# 4. CHAINS (LCEL)
# ================================
extract_chain = extract_prompt | llm
match_chain = match_prompt | llm
score_chain = score_prompt | llm
explain_chain = explain_prompt | llm

# ================================
# 5. PIPELINE FUNCTION
# ================================
def run_pipeline(resume, jd):
    
    # Step 1: Extract
    extracted = extract_chain.invoke({"resume": resume})
    
    # Step 2: Match
    match = match_chain.invoke({
        "jd": jd,
        "resume_data": extracted.content
    })
    
    # Step 3: Score
    score = score_chain.invoke({
        "match_data": match.content
    })
    
    # Step 4: Explain
    explanation = explain_chain.invoke({
        "score": score.content,
        "match_data": match.content
    })
    
    return {
        "EXTRACTED": extracted.content,
        "MATCH": match.content,
        "SCORE": score.content,
        "EXPLANATION": explanation.content
    }

# ================================
# 6. SAMPLE DATA
# ================================

job_description = """
We are hiring a Data Scientist.

Requirements:
- Python
- Machine Learning
- Deep Learning
- NLP
- SQL
- Experience with TensorFlow or PyTorch
"""

strong_resume = """
Experienced Data Scientist with 5 years of experience.
Skills: Python, Machine Learning, Deep Learning, NLP, SQL
Tools: TensorFlow, PyTorch
"""

avg_resume = """
Data Analyst with 2 years experience.
Skills: Python, SQL, Data Visualization
Tools: Excel, Power BI
"""

weak_resume = """
Fresher graduate.
Skills: MS Word, Communication
"""

# ================================
# 7. RUN FOR 3 CANDIDATES
# ================================
if __name__ == "__main__":
    
    resumes = {
        "STRONG": strong_resume,
        "AVERAGE": avg_resume,
        "WEAK": weak_resume
    }
    
    for name, resume in resumes.items():
        print(f"\n================ {name} CANDIDATE ================\n")
        
        result = run_pipeline(resume, job_description)
        
        for key, value in result.items():
            print(f"\n{key}:\n{value}")