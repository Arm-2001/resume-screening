import os
import json
import uuid
import re
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Tuple, Any

# Web framework
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Document processing
import PyPDF2
from docx import Document
import io

# Lightweight ML and text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('jobs', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Global variables
current_job = None
uploaded_resumes = []
processed_results = []

class ResumeParser:
    def __init__(self):
        # Comprehensive skill keywords dictionary
        self.skill_keywords = {
            'python': [
                'python', 'python programming', 'python development', 'django', 'flask', 
                'fastapi', 'streamlit', 'jupyter', 'pandas', 'numpy', 'scipy'
            ],
            'javascript': [
                'javascript', 'js', 'typescript', 'node.js', 'nodejs', 'react', 
                'reactjs', 'vue', 'vuejs', 'angular', 'jquery', 'express'
            ],
            'java': [
                'java', 'java programming', 'spring', 'spring boot', 'hibernate', 
                'maven', 'gradle', 'junit', 'jsp', 'servlet'
            ],
            'cpp': ['c++', 'cpp', 'c plus plus', 'stl', 'boost'],
            'csharp': ['c#', 'csharp', '.net', 'dotnet', 'asp.net', 'entity framework'],
            'machine_learning': [
                'machine learning', 'ml', 'artificial intelligence', 'ai', 
                'deep learning', 'neural networks', 'tensorflow', 'pytorch', 
                'scikit-learn', 'sklearn', 'predictive modeling'
            ],
            'data_science': [
                'data science', 'data analysis', 'data analytics', 'big data', 
                'data visualization', 'statistics', 'r programming', 'tableau', 'powerbi'
            ],
            'web_development': [
                'web development', 'frontend', 'backend', 'full-stack', 'html', 
                'css', 'sass', 'bootstrap', 'rest api', 'graphql'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 
                'nosql', 'database design', 'data modeling'
            ],
            'cloud': [
                'aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes', 
                'cloud computing', 'serverless', 'microservices'
            ],
            'devops': [
                'devops', 'ci/cd', 'jenkins', 'git', 'github', 'gitlab', 
                'docker', 'kubernetes', 'ansible', 'terraform'
            ],
            'mobile': [
                'mobile development', 'android', 'ios', 'react native', 'flutter', 
                'swift', 'kotlin', 'xamarin'
            ],
            'cybersecurity': [
                'cybersecurity', 'information security', 'penetration testing', 
                'ethical hacking', 'encryption', 'firewall', 'security audit'
            ]
        }

    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return ""

    def extract_text_from_file(self, file_path):
        """Extract text from various file formats"""
        file_extension = Path(file_path).suffix.lower()

        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            return "Unsupported file format"

    def extract_contact_info(self, text):
        """Extract contact information from resume text"""
        contact_info = {}

        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contact_info['email'] = emails[0] if emails else ""

        # Phone extraction
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        contact_info['phone'] = phones[0] if phones else ""

        return contact_info

    def extract_skills(self, text):
        """Extract skills from resume text"""
        text_lower = text.lower()
        found_skills = {}
        all_skills_found = []

        for category, skills in self.skill_keywords.items():
            category_skills = []
            for skill in skills:
                skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(skill_pattern, text_lower):
                    category_skills.append(skill)
                    all_skills_found.append(skill)

            if category_skills:
                found_skills[category] = category_skills

        unique_skills = list(set(all_skills_found))

        return {
            'categorized_skills': found_skills,
            'all_skills': unique_skills,
            'total_skills_found': len(unique_skills)
        }

    def extract_experience(self, text):
        """Extract work experience information"""
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*yrs?\s*(?:of\s*)?experience',
            r'experience\s*:?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*(?:in|with|of)',
            r'over\s*(\d+)\s*years?',
            r'more\s*than\s*(\d+)\s*years?'
        ]

        years_experience = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text.lower())
            years_experience.extend([int(match) for match in matches])

        # Extract date ranges
        date_patterns = [
            r'(\d{4})\s*[-–]\s*(\d{4})',
            r'(\d{4})\s*[-–]\s*present',
            r'(\d{4})\s*[-–]\s*current'
        ]

        current_year = 2025
        calculated_years = []

        for pattern in date_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                start_year = int(match[0])
                if len(match) > 1 and match[1].isdigit():
                    end_year = int(match[1])
                    calculated_years.append(end_year - start_year)
                elif 'present' in text.lower() or 'current' in text.lower():
                    calculated_years.append(current_year - start_year)

        all_experience = years_experience + calculated_years
        return {
            'years_experience': max(all_experience) if all_experience else 0,
            'all_experience_values': all_experience
        }

    def extract_education(self, text):
        """Extract education information"""
        text_lower = text.lower()
        education_info = {
            'degrees': [],
            'institutions': [],
            'fields': [],
            'graduation_years': []
        }

        degree_patterns = [
            r'(bachelor(?:\'s)?|bs|ba|b\.s\.|b\.a\.)\s+(?:of\s+)?(?:science|arts)?\s+in\s+([^,\n\.]+)',
            r'(master(?:\'s)?|ms|ma|m\.s\.|m\.a\.)\s+(?:of\s+)?(?:science|arts)?\s+in\s+([^,\n\.]+)',
            r'(phd|ph\.d\.|doctorate|doctoral)\s+in\s+([^,\n\.]+)',
            r'(mba|m\.b\.a\.)',
            r'(diploma|certificate)\s+in\s+([^,\n\.]+)'
        ]

        for pattern in degree_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    education_info['degrees'].append(match[0])
                    education_info['fields'].append(match[1].strip())
                elif isinstance(match, str):
                    education_info['degrees'].append(match)

        return education_info

    def parse_resume(self, file_path, filename):
        """Complete resume parsing"""
        try:
            text = self.extract_text_from_file(file_path)

            if not text or text == "Unsupported file format":
                return {
                    'filename': filename,
                    'error': 'Could not extract text from file',
                    'parsing_success': False
                }

            contact_info = self.extract_contact_info(text)
            skills_info = self.extract_skills(text)
            experience_info = self.extract_experience(text)
            education_info = self.extract_education(text)

            parsed_resume = {
                'filename': filename,
                'contact_info': contact_info,
                'skills': skills_info,
                'experience': experience_info,
                'education': education_info,
                'raw_text': text[:1000],
                'full_text': text,
                'parsing_success': True
            }

            return parsed_resume

        except Exception as e:
            return {
                'filename': filename,
                'error': f'Parsing error: {str(e)}',
                'parsing_success': False
            }

class AIScoring:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    def calculate_skills_match(self, resume_skills, job_essential_skills, job_preferred_skills):
        """Calculate skills matching score"""
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        essential_lower = [skill.lower() for skill in job_essential_skills]
        preferred_lower = [skill.lower() for skill in job_preferred_skills]

        essential_matches = 0
        for skill in essential_lower:
            if any(skill in resume_skill or resume_skill in skill for resume_skill in resume_skills_lower):
                essential_matches += 1

        essential_score = (essential_matches / len(essential_lower)) * 100 if essential_lower else 0

        preferred_matches = 0
        for skill in preferred_lower:
            if any(skill in resume_skill or resume_skill in skill for resume_skill in resume_skills_lower):
                preferred_matches += 1

        preferred_score = (preferred_matches / len(preferred_lower)) * 100 if preferred_lower else 0

        overall_skills_score = (essential_score * 0.7) + (preferred_score * 0.3)

        return {
            'essential_score': round(essential_score, 1),
            'preferred_score': round(preferred_score, 1),
            'overall_skills_score': round(overall_skills_score, 1),
            'essential_matches': essential_matches,
            'preferred_matches': preferred_matches
        }

    def calculate_experience_match(self, resume_experience, required_experience):
        """Calculate experience matching score"""
        resume_years = resume_experience.get('years_experience', 0)
        required_years = float(required_experience) if required_experience else 0

        if required_years == 0:
            return 100

        if resume_years >= required_years:
            experience_score = min(100, 80 + (resume_years - required_years) * 5)
        else:
            experience_score = (resume_years / required_years) * 80

        return round(experience_score, 1)

    def calculate_education_match(self, resume_education, job_education_req):
        """Calculate education matching score"""
        if not job_education_req or job_education_req.lower() == 'none':
            return 100

        resume_degrees = [degree.lower() for degree in resume_education.get('degrees', [])]
        job_education_lower = job_education_req.lower()

        degree_score = 0
        if any(degree in job_education_lower for degree in resume_degrees):
            degree_score = 80
        elif resume_degrees:
            degree_score = 60
        else:
            degree_score = 20

        return min(100, round(degree_score, 1))

    def calculate_semantic_similarity(self, resume_text, job_description):
        """Calculate semantic similarity using TF-IDF"""
        try:
            texts = [resume_text, job_description]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return round(similarity * 100, 1)
        except Exception as e:
            print(f"Error in semantic similarity calculation: {e}")
            return 50

    def calculate_overall_score(self, parsed_resume, job_posting):
        """Calculate comprehensive matching score"""
        try:
            resume_skills = parsed_resume['skills']['all_skills']
            resume_experience = parsed_resume['experience']
            resume_education = parsed_resume['education']
            resume_text = parsed_resume['full_text']

            job_requirements = job_posting['requirements']
            job_weights = job_posting['scoring_weights']

            skills_match = self.calculate_skills_match(
                resume_skills,
                job_requirements['essential_skills'],
                job_requirements['preferred_skills']
            )

            experience_score = self.calculate_experience_match(
                resume_experience,
                job_requirements['minimum_experience']
            )

            education_score = self.calculate_education_match(
                resume_education,
                job_requirements['education_requirements']
            )

            semantic_score = self.calculate_semantic_similarity(
                resume_text,
                job_requirements['job_description']
            )

            overall_score = (
                skills_match['overall_skills_score'] * (job_weights['essential_skills'] + job_weights['preferred_skills']) +
                experience_score * job_weights['experience'] +
                education_score * job_weights['education'] +
                semantic_score * 0.1
            )

            overall_score = round(overall_score, 1)

            detailed_scores = {
                'overall_score': overall_score,
                'skills_breakdown': skills_match,
                'experience_score': experience_score,
                'education_score': education_score,
                'semantic_score': semantic_score,
                'candidate_name': parsed_resume['filename'],
                'strengths': self.identify_strengths(skills_match, experience_score, education_score),
                'gaps': self.identify_gaps(skills_match, experience_score, education_score)
            }

            return detailed_scores

        except Exception as e:
            print(f"Error calculating overall score: {e}")
            return {
                'overall_score': 0,
                'error': str(e),
                'candidate_name': parsed_resume.get('filename', 'Unknown')
            }

    def identify_strengths(self, skills_match, experience_score, education_score):
        """Identify candidate strengths"""
        strengths = []
        if skills_match['essential_score'] >= 80:
            strengths.append("Strong match for essential skills")
        if skills_match['preferred_score'] >= 70:
            strengths.append("Good match for preferred skills")
        if experience_score >= 90:
            strengths.append("Exceeds experience requirements")
        if education_score >= 80:
            strengths.append("Strong educational background")
        return strengths if strengths else ["Meets basic requirements"]

    def identify_gaps(self, skills_match, experience_score, education_score):
        """Identify candidate gaps"""
        gaps = []
        if skills_match['essential_score'] < 50:
            gaps.append("Missing key essential skills")
        if experience_score < 60:
            gaps.append("Below required experience level")
        if education_score < 60:
            gaps.append("Education requirements not fully met")
        return gaps if gaps else ["No significant gaps identified"]

# Initialize components
resume_parser = ResumeParser()
ai_scorer = AIScoring()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create_job', methods=['GET', 'POST'])
def create_job():
    global current_job
    
    if request.method == 'POST':
        job_id = str(uuid.uuid4())[:8]
        
        job_posting = {
            "job_id": job_id,
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "company_info": {
                "company_name": request.form['company_name'],
                "location": request.form['location']
            },
            "job_details": {
                "job_title": request.form['job_title'],
                "department": request.form['department'],
                "employment_type": request.form['employment_type'],
                "experience_level": request.form['experience_level'],
                "salary_range": request.form['salary_range']
            },
            "requirements": {
                "essential_skills": [skill.strip() for skill in request.form['essential_skills'].split(',') if skill.strip()],
                "preferred_skills": [skill.strip() for skill in request.form['preferred_skills'].split(',') if skill.strip()],
                "minimum_experience": float(request.form['min_experience']) if request.form['min_experience'] else 0,
                "education_requirements": request.form['education_req'],
                "job_description": request.form['job_description']
            },
            "scoring_weights": {
                "essential_skills": 0.5,
                "preferred_skills": 0.2,
                "experience": 0.2,
                "education": 0.1
            }
        }
        
        current_job = job_posting
        
        # Save job posting
        with open(f'jobs/job_{job_id}.json', 'w') as f:
            json.dump(job_posting, f, indent=2)
        
        flash(f'Job posting created successfully! Job ID: {job_id}', 'success')
        return redirect(url_for('upload_resumes'))
    
    return render_template('create_job.html')

@app.route('/upload_resumes', methods=['GET', 'POST'])
def upload_resumes():
    global uploaded_resumes, current_job
    
    if not current_job:
        flash('Please create a job posting first!', 'error')
        return redirect(url_for('create_job'))
    
    if request.method == 'POST':
        uploaded_resumes = []
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            flash('No files selected!', 'error')
            return redirect(request.url)
        
        for file in files:
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Parse resume
                parsed_resume = resume_parser.parse_resume(file_path, filename)
                
                if parsed_resume.get('parsing_success', False):
                    uploaded_resumes.append(parsed_resume)
                
                # Clean up file
                os.remove(file_path)
        
        flash(f'Successfully processed {len(uploaded_resumes)} resumes!', 'success')
        return redirect(url_for('results'))
    
    return render_template('upload_resumes.html', job=current_job)

@app.route('/results')
def results():
    global processed_results, uploaded_resumes, current_job
    
    if not current_job:
        flash('Please create a job posting first!', 'error')
        return redirect(url_for('create_job'))
    
    if not uploaded_resumes:
        flash('Please upload resumes first!', 'error')
        return redirect(url_for('upload_resumes'))
    
    # Calculate scores
    processed_results = []
    for resume in uploaded_resumes:
        if resume.get('parsing_success', False):
            score_result = ai_scorer.calculate_overall_score(resume, current_job)
            processed_results.append(score_result)
    
    # Sort by score
    processed_results.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
    
    return render_template('results.html', 
                         results=processed_results, 
                         job=current_job,
                         total_candidates=len(processed_results))

@app.route('/api/results_data')
def api_results_data():
    """API endpoint to get results data for charts"""
    if not processed_results:
        return jsonify({'error': 'No results available'})
    
    top_10 = processed_results[:10]
    chart_data = {
        'labels': [result.get('candidate_name', f'Candidate {i+1}') for i, result in enumerate(top_10)],
        'scores': [result.get('overall_score', 0) for result in top_10],
        'colors': ['#10B981' if score >= 80 else '#F59E0B' if score >= 60 else '#EF4444' 
                  for score in [result.get('overall_score', 0) for result in top_10]]
    }
    
    return jsonify(chart_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
