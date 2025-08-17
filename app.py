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
from pathlib import Path
from typing import List, Dict, Tuple, Any
import tempfile
import shutil

# Document processing
import PyPDF2
from docx import Document
import io

# NLP and ML
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Environment variables
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

# Global variables for system components
job_system = None
resume_parser = None
ai_scorer = None
screening_system = None

class JobPostingSystem:
    def __init__(self):
        self.current_job = None
        self.job_id = None
        self.jobs_storage = {}

    def create_job_posting(self, job_data):
        """Create a structured job posting"""
        try:
            # Generate unique job ID
            self.job_id = str(uuid.uuid4())[:8]

            # Create job posting structure
            job_posting = {
                "job_id": self.job_id,
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "company_info": {
                    "company_name": job_data.get('company_name'),
                    "location": job_data.get('location')
                },
                "job_details": {
                    "job_title": job_data.get('job_title'),
                    "department": job_data.get('department'),
                    "employment_type": job_data.get('employment_type'),
                    "experience_level": job_data.get('experience_level'),
                    "salary_range": job_data.get('salary_range')
                },
                "requirements": {
                    "essential_skills": [skill.strip() for skill in job_data.get('essential_skills', '').split(',') if skill.strip()],
                    "preferred_skills": [skill.strip() for skill in job_data.get('preferred_skills', '').split(',') if skill.strip()],
                    "minimum_experience": job_data.get('minimum_experience', 0),
                    "education_requirements": job_data.get('education_requirements'),
                    "job_description": job_data.get('job_description')
                },
                "scoring_weights": {
                    "essential_skills": 0.5,
                    "preferred_skills": 0.2,
                    "experience": 0.2,
                    "education": 0.1
                }
            }

            # Store job posting
            self.current_job = job_posting
            self.jobs_storage[self.job_id] = job_posting

            return {
                "success": True,
                "job_id": self.job_id,
                "message": f"Job posting created successfully! Job ID: {self.job_id}"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_current_job(self):
        """Get current job posting details"""
        return self.current_job

    def get_job_by_id(self, job_id):
        """Get job posting by ID"""
        return self.jobs_storage.get(job_id)

class ResumeParser:
    def __init__(self):
        # Load NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found. Using basic tokenization...")
            # Fallback to basic text processing if spaCy model not available
            self.nlp = None

        # Comprehensive skill keywords dictionary
        self.skill_keywords = {
            'python': [
                'python', 'python programming', 'python development', 'django', 'flask', 
                'fastapi', 'streamlit', 'pandas', 'numpy', 'scipy', 'matplotlib', 'jupyter'
            ],
            'javascript': [
                'javascript', 'js', 'typescript', 'node.js', 'nodejs', 'react', 'reactjs', 
                'vue', 'vuejs', 'angular', 'jquery', 'npm', 'webpack'
            ],
            'java': [
                'java', 'java programming', 'spring', 'spring boot', 'hibernate', 
                'maven', 'gradle', 'junit'
            ],
            'machine_learning': [
                'machine learning', 'ml', 'artificial intelligence', 'ai', 'deep learning',
                'neural networks', 'tensorflow', 'pytorch', 'scikit-learn', 'sklearn'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                'database', 'nosql'
            ],
            'cloud': [
                'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
                'cloud computing', 'serverless'
            ],
            'web_dev': [
                'html', 'css', 'frontend', 'backend', 'full-stack', 'rest api',
                'graphql', 'microservices'
            ],
            'devops': [
                'devops', 'ci/cd', 'jenkins', 'git', 'github', 'docker',
                'kubernetes', 'terraform'
            ]
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
        file_extension = Path(filename).suffix.lower()

        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_content)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_content)
        elif file_extension == '.txt':
            return file_content.decode('utf-8') if isinstance(file_content, bytes) else file_content
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

        unique_skills = list(dict.fromkeys(all_skills_found))  # Remove duplicates

        return {
            'categorized_skills': found_skills,
            'all_skills': unique_skills,
            'skill_count_by_category': {cat: len(skills) for cat, skills in found_skills.items()},
            'total_skills_found': len(unique_skills)
        }

    def extract_experience(self, text):
        """Extract work experience information"""
        # Enhanced experience patterns
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
            'fields': []
        }

        # Enhanced degree patterns
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

    def parse_resume(self, file_content, filename):
        """Complete resume parsing"""
        try:
            # Extract text
            text = self.extract_text_from_file(file_content, filename)

            if not text or text == "Unsupported file format":
                return {
                    'filename': filename,
                    'error': 'Could not extract text from file',
                    'parsing_success': False
                }

            # Extract all information
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
                'parsing_success': True,
                'parsing_stats': {
                    'total_skills_found': skills_info.get('total_skills_found', 0),
                    'skill_categories': len(skills_info.get('categorized_skills', {})),
                    'years_experience': experience_info.get('years_experience', 0),
                    'degrees_found': len(education_info.get('degrees', [])),
                    'text_length': len(text)
                }
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
        # Load pre-trained sentence transformer for semantic similarity
        try:
            print("Loading AI models...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ AI models loaded successfully!")
        except Exception as e:
            print(f"Error loading AI models: {e}")
            self.sentence_model = None

    def calculate_skills_match(self, resume_skills, job_essential_skills, job_preferred_skills):
        """Calculate skills matching score"""
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        essential_lower = [skill.lower() for skill in job_essential_skills]
        preferred_lower = [skill.lower() for skill in job_preferred_skills]

        # Essential skills match
        essential_matches = 0
        for skill in essential_lower:
            if any(skill in resume_skill or resume_skill in skill for resume_skill in resume_skills_lower):
                essential_matches += 1

        essential_score = (essential_matches / len(essential_lower)) * 100 if essential_lower else 0

        # Preferred skills match
        preferred_matches = 0
        for skill in preferred_lower:
            if any(skill in resume_skill or resume_skill in skill for resume_skill in resume_skills_lower):
                preferred_matches += 1

        preferred_score = (preferred_matches / len(preferred_lower)) * 100 if preferred_lower else 0

        # Weighted combination
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
        """Calculate semantic similarity between resume and job description"""
        if not self.sentence_model:
            return 50

        try:
            resume_embedding = self.sentence_model.encode([resume_text])
            job_embedding = self.sentence_model.encode([job_description])
            similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
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

            # Calculate individual scores
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

            # Calculate weighted overall score
            overall_score = (
                skills_match['overall_skills_score'] * job_weights['essential_skills'] +
                skills_match['overall_skills_score'] * job_weights['preferred_skills'] +
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

class ResumeScreeningSystem:
    def __init__(self):
        self.uploaded_resumes = []
        self.processed_results = []

    def process_uploaded_files(self, files_data):
        """Process uploaded resume files"""
        if not files_data:
            return {"success": False, "message": "No files uploaded"}

        if not job_system.current_job:
            return {"success": False, "message": "Please create a job posting first!"}

        self.uploaded_resumes = []
        results = []

        for file_info in files_data:
            filename = file_info['filename']
            file_content = file_info['content']

            try:
                parsed_resume = resume_parser.parse_resume(file_content, filename)

                if parsed_resume.get('parsing_success', False):
                    self.uploaded_resumes.append(parsed_resume)
                    results.append(f"✅ {filename} - Parsed successfully")
                else:
                    results.append(f"❌ {filename} - Failed to parse: {parsed_resume.get('error', 'Unknown error')}")

            except Exception as e:
                results.append(f"❌ {filename} - Error: {str(e)}")

        return {
            "success": True,
            "message": f"Processed {len(files_data)} files",
            "results": results,
            "processed_count": len(self.uploaded_resumes)
        }

    def analyze_resumes(self):
        """Analyze all uploaded resumes against the job posting"""
        if not self.uploaded_resumes:
            return {"success": False, "message": "No resumes to analyze"}

        if not job_system.current_job:
            return {"success": False, "message": "No job posting found"}

        scored_candidates = []

        for resume in self.uploaded_resumes:
            if resume.get('parsing_success', False):
                try:
                    score_result = ai_scorer.calculate_overall_score(resume, job_system.current_job)
                    scored_candidates.append(score_result)
                except Exception as e:
                    print(f"Error scoring {resume['filename']}: {e}")

        # Sort by overall score (descending)
        scored_candidates.sort(key=lambda x: x.get('overall_score', 0), reverse=True)

        self.processed_results = scored_candidates

        return {
            "success": True,
            "candidates": scored_candidates,
            "total_candidates": len(scored_candidates),
            "job_info": {
                "title": job_system.current_job['job_details']['job_title'],
                "company": job_system.current_job['company_info']['company_name']
            }
        }

# Initialize system components
def initialize_system():
    global job_system, resume_parser, ai_scorer, screening_system
    
    print("Initializing Resume Screening System...")
    job_system = JobPostingSystem()
    resume_parser = ResumeParser()
    ai_scorer = AIScoring()
    screening_system = ResumeScreeningSystem()
    print("✅ System initialized successfully!")

# API Routes
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Resume Screening API is running",
        "version": "1.0.0"
    })

@app.route('/api/create-job', methods=['POST'])
def create_job():
    try:
        job_data = request.json
        result = job_system.create_job_posting(job_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/get-job/<job_id>', methods=['GET'])
def get_job(job_id):
    try:
        job = job_system.get_job_by_id(job_id)
        if job:
            return jsonify({"success": True, "job": job})
        else:
            return jsonify({"success": False, "message": "Job not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/current-job', methods=['GET'])
def get_current_job():
    try:
        job = job_system.get_current_job()
        if job:
            return jsonify({"success": True, "job": job})
        else:
            return jsonify({"success": False, "message": "No current job posting"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/upload-resumes', methods=['POST'])
def upload_resumes():
    try:
        if 'files' not in request.files:
            return jsonify({"success": False, "message": "No files uploaded"}), 400

        files = request.files.getlist('files')
        files_data = []

        for file in files:
            if file.filename:
                files_data.append({
                    'filename': file.filename,
                    'content': file.read()
                })

        result = screening_system.process_uploaded_files(files_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/analyze-resumes', methods=['POST'])
def analyze_resumes():
    try:
        result = screening_system.analyze_resumes()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/system-info', methods=['GET'])
def system_info():
    return jsonify({
        "skill_categories": len(resume_parser.skill_keywords) if resume_parser else 0,
        "total_skills": sum(len(skills) for skills in resume_parser.skill_keywords.values()) if resume_parser else 0,
        "models_loaded": ai_scorer.sentence_model is not None if ai_scorer else False,
        "current_job_exists": job_system.current_job is not None if job_system else False
    })

if __name__ == '__main__':
    initialize_system()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
