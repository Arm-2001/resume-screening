from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import json
import os
import re
from datetime import datetime
import uuid
import tempfile
import io

# Document processing
import PyPDF2
from docx import Document

# NLP and ML
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for models
sentence_model = None
nlp_model = None

def load_models():
    """Load AI models on startup"""
    global sentence_model, nlp_model
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        nlp_model = spacy.load("en_core_web_sm")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

class ResumeParser:
    def __init__(self):
        self.skill_keywords = {
            'python': ['python', 'django', 'flask', 'fastapi', 'pandas', 'numpy', 'jupyter'],
            'javascript': ['javascript', 'js', 'node.js', 'react', 'vue', 'angular', 'typescript'],
            'java': ['java', 'spring', 'hibernate', 'maven', 'gradle'],
            'machine_learning': ['machine learning', 'ml', 'artificial intelligence', 'ai', 'tensorflow', 'pytorch'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'web_dev': ['html', 'css', 'rest api', 'graphql', 'microservices'],
            'data_science': ['data science', 'analytics', 'visualization', 'statistics'],
            'mobile': ['android', 'ios', 'react native', 'flutter'],
            'devops': ['ci/cd', 'jenkins', 'git', 'linux']
        }

    def extract_text_from_pdf(self, file_content):
        """Extract text from PDF file content"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def extract_text_from_docx(self, file_content):
        """Extract text from DOCX file content"""
        try:
            doc = Document(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return ""

    def extract_text_from_file(self, file_content, filename):
        """Extract text from various file formats"""
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file_content)
        elif file_extension == 'docx':
            return self.extract_text_from_docx(file_content)
        elif file_extension == 'txt':
            return file_content.decode('utf-8')
        else:
            return "Unsupported file format"

    def extract_contact_info(self, text):
        """Extract contact information"""
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
        """Extract skills from text"""
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
        
        return {
            'categorized_skills': found_skills,
            'all_skills': list(set(all_skills_found)),
            'total_skills_found': len(set(all_skills_found))
        }

    def extract_experience(self, text):
        """Extract experience information"""
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*yrs?\s*(?:of\s*)?experience',
            r'experience\s*:?\s*(\d+)\+?\s*years?'
        ]
        
        years_experience = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text.lower())
            years_experience.extend([int(match) for match in matches])
        
        return {
            'years_experience': max(years_experience) if years_experience else 0,
            'positions': []
        }

    def extract_education(self, text):
        """Extract education information"""
        text_lower = text.lower()
        education_info = {
            'degrees': [],
            'institutions': [],
            'fields': []
        }
        
        degree_patterns = [
            r'(bachelor(?:\'s)?|bs|ba|b\.s\.|b\.a\.)',
            r'(master(?:\'s)?|ms|ma|m\.s\.|m\.a\.)',
            r'(phd|ph\.d\.|doctorate)',
            r'(mba|m\.b\.a\.)'
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, text_lower)
            education_info['degrees'].extend(matches)
        
        return education_info

    def parse_resume(self, file_content, filename):
        """Parse resume and extract information"""
        try:
            text = self.extract_text_from_file(file_content, filename)
            
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
            
            return {
                'filename': filename,
                'contact_info': contact_info,
                'skills': skills_info,
                'experience': experience_info,
                'education': education_info,
                'full_text': text,
                'parsing_success': True
            }
            
        except Exception as e:
            return {
                'filename': filename,
                'error': f'Parsing error: {str(e)}',
                'parsing_success': False
            }

class AIScoring:
    def __init__(self):
        pass

    def calculate_skills_match(self, resume_skills, job_essential_skills, job_preferred_skills):
        """Calculate skills matching score"""
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        essential_lower = [skill.lower().strip() for skill in job_essential_skills if skill.strip()]
        preferred_lower = [skill.lower().strip() for skill in job_preferred_skills if skill.strip()]
        
        # Essential skills match
        essential_matches = 0
        for skill in essential_lower:
            if any(skill in resume_skill or resume_skill in skill for resume_skill in resume_skills_lower):
                essential_matches += 1
        
        essential_score = (essential_matches / len(essential_lower)) * 100 if essential_lower else 100
        
        # Preferred skills match
        preferred_matches = 0
        for skill in preferred_lower:
            if any(skill in resume_skill or resume_skill in skill for resume_skill in resume_skills_lower):
                preferred_matches += 1
        
        preferred_score = (preferred_matches / len(preferred_lower)) * 100 if preferred_lower else 100
        
        # Overall skills score
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
        
        if any(degree in job_education_lower for degree in resume_degrees):
            return 90
        elif resume_degrees:
            return 70
        else:
            return 40

    def calculate_semantic_similarity(self, resume_text, job_description):
        """Calculate semantic similarity"""
        try:
            if not sentence_model:
                return 50
            
            resume_embedding = sentence_model.encode([resume_text])
            job_embedding = sentence_model.encode([job_description])
            similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
            return round(similarity * 100, 1)
        except Exception as e:
            print(f"Error in semantic similarity: {e}")
            return 50

    def calculate_overall_score(self, parsed_resume, job_posting):
        """Calculate comprehensive matching score"""
        try:
            resume_skills = parsed_resume['skills']['all_skills']
            resume_experience = parsed_resume['experience']
            resume_education = parsed_resume['education']
            resume_text = parsed_resume['full_text']
            
            job_requirements = job_posting['requirements']
            
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
            
            # Weighted overall score
            overall_score = (
                skills_match['overall_skills_score'] * 0.5 +
                experience_score * 0.2 +
                education_score * 0.1 +
                semantic_score * 0.2
            )
            
            return {
                'overall_score': round(overall_score, 1),
                'skills_breakdown': skills_match,
                'experience_score': experience_score,
                'education_score': education_score,
                'semantic_score': semantic_score,
                'candidate_name': parsed_resume['filename']
            }
            
        except Exception as e:
            print(f"Error calculating score: {e}")
            return {
                'overall_score': 0,
                'error': str(e),
                'candidate_name': parsed_resume.get('filename', 'Unknown')
            }

# Initialize components
resume_parser = ResumeParser()
ai_scorer = AIScoring()

# Storage for job postings and resumes
active_jobs = {}
processed_resumes = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': sentence_model is not None and nlp_model is not None
    })

@app.route('/create-job', methods=['POST'])
def create_job():
    """Create a new job posting"""
    try:
        data = request.get_json()
        
        job_id = str(uuid.uuid4())[:8]
        
        job_posting = {
            'job_id': job_id,
            'created_date': datetime.now().isoformat(),
            'company_info': {
                'company_name': data.get('company_name', ''),
                'location': data.get('location', '')
            },
            'job_details': {
                'job_title': data.get('job_title', ''),
                'department': data.get('department', ''),
                'employment_type': data.get('employment_type', ''),
                'experience_level': data.get('experience_level', ''),
                'salary_range': data.get('salary_range', '')
            },
            'requirements': {
                'essential_skills': [skill.strip() for skill in data.get('essential_skills', '').split(',') if skill.strip()],
                'preferred_skills': [skill.strip() for skill in data.get('preferred_skills', '').split(',') if skill.strip()],
                'minimum_experience': data.get('minimum_experience', 0),
                'education_requirements': data.get('education_requirements', ''),
                'job_description': data.get('job_description', '')
            }
        }
        
        active_jobs[job_id] = job_posting
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Job posting created successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/upload-resumes', methods=['POST'])
def upload_resumes():
    """Upload and parse multiple resumes"""
    try:
        job_id = request.form.get('job_id')
        
        if not job_id or job_id not in active_jobs:
            return jsonify({
                'success': False,
                'error': 'Invalid job ID'
            }), 400
        
        files = request.files.getlist('resumes')
        
        if not files:
            return jsonify({
                'success': False,
                'error': 'No files uploaded'
            }), 400
        
        results = []
        parsed_resumes = []
        
        for file in files:
            if file.filename == '':
                continue
                
            file_content = file.read()
            parsed_resume = resume_parser.parse_resume(file_content, file.filename)
            
            if parsed_resume.get('parsing_success', False):
                parsed_resumes.append(parsed_resume)
                results.append({
                    'filename': file.filename,
                    'status': 'success',
                    'skills_found': parsed_resume['skills']['total_skills_found']
                })
            else:
                results.append({
                    'filename': file.filename,
                    'status': 'failed',
                    'error': parsed_resume.get('error', 'Unknown error')
                })
        
        processed_resumes[job_id] = parsed_resumes
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(parsed_resumes)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/analyze-resumes', methods=['POST'])
def analyze_resumes():
    """Analyze resumes against job requirements"""
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        
        if not job_id or job_id not in active_jobs:
            return jsonify({
                'success': False,
                'error': 'Invalid job ID'
            }), 400
        
        if job_id not in processed_resumes:
            return jsonify({
                'success': False,
                'error': 'No resumes found for this job'
            }), 400
        
        job_posting = active_jobs[job_id]
        resumes = processed_resumes[job_id]
        
        scored_candidates = []
        
        for resume in resumes:
            score_result = ai_scorer.calculate_overall_score(resume, job_posting)
            scored_candidates.append(score_result)
        
        # Sort by overall score
        scored_candidates.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
        
        return jsonify({
            'success': True,
            'job_info': {
                'job_title': job_posting['job_details']['job_title'],
                'company_name': job_posting['company_info']['company_name']
            },
            'total_candidates': len(scored_candidates),
            'candidates': scored_candidates[:20]  # Return top 20
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/get-job/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get job posting details"""
    if job_id in active_jobs:
        return jsonify({
            'success': True,
            'job': active_jobs[job_id]
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Job not found'
        }), 404

@app.route('/get-jobs', methods=['GET'])
def get_all_jobs():
    """Get all active job postings"""
    return jsonify({
        'success': True,
        'jobs': list(active_jobs.values())
    })

if __name__ == '__main__':
    print("Loading AI models...")
    if load_models():
        print("Models loaded successfully!")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
    else:
        print("Failed to load models. Please check your environment.")
