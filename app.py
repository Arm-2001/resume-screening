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
import io
import tempfile
import base64
import logging

# Document processing
import PyPDF2
from docx import Document

# NLP and ML
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for storing data
current_job = None
uploaded_resumes = []
processed_results = []

class ResumeParser:
    def __init__(self):
        # Comprehensive skill keywords dictionary
        self.skill_keywords = {
            # Programming Languages
            'python': [
                'python', 'python programming', 'python development', 'python scripting', 'python 3', 'python 2',
                'django', 'flask', 'fastapi', 'streamlit', 'jupyter', 'ipython', 'anaconda',
                'pip', 'conda', 'virtualenv', 'pytest', 'unittest', 'pylint', 'black', 'flake8'
            ],
            'javascript': [
                'javascript', 'js', 'ecmascript', 'es6', 'es2015', 'es2020', 'typescript', 'ts',
                'node.js', 'nodejs', 'express', 'react', 'reactjs', 'vue', 'vuejs', 'angular',
                'jquery', 'npm', 'yarn', 'webpack', 'babel', 'eslint', 'jest', 'mocha'
            ],
            'java': [
                'java', 'java programming', 'java development', 'java 8', 'java 11', 'java 17',
                'spring', 'spring boot', 'hibernate', 'maven', 'gradle', 'junit', 'mockito',
                'jsp', 'servlet', 'struts', 'jdbc', 'jpa', 'ejb', 'jsf'
            ],
            'cpp': [
                'c++', 'cpp', 'c plus plus', 'c++ programming', 'c++ development',
                'stl', 'boost', 'cmake', 'make', 'gcc', 'clang', 'visual studio', 'qt'
            ],
            'c_language': [
                'c programming', 'c language', 'ansi c', 'iso c', 'embedded c',
                'gcc', 'clang', 'make', 'cmake', 'gdb', 'valgrind'
            ],
            'csharp': [
                'c#', 'csharp', 'c sharp', '.net', 'dotnet', 'asp.net', 'entity framework',
                'xamarin', 'blazor', 'wpf', 'winforms', 'unity', 'visual studio'
            ],
            'other_languages': [
                'go', 'golang', 'rust', 'kotlin', 'swift', 'php', 'ruby', 'perl', 'lua',
                'scala', 'clojure', 'haskell', 'erlang', 'elixir', 'dart', 'flutter',
                'r programming', 'matlab', 'octave', 'julia', 'fortran', 'cobol'
            ],
            # Machine Learning & AI
            'machine_learning': [
                'machine learning', 'ml', 'artificial intelligence', 'ai', 'predictive modeling',
                'supervised learning', 'unsupervised learning', 'reinforcement learning',
                'classification', 'regression', 'clustering', 'dimensionality reduction',
                'feature engineering', 'feature selection', 'model training', 'model validation',
                'cross validation', 'hyperparameter tuning', 'ensemble methods', 'bagging',
                'boosting', 'random forest', 'gradient boosting', 'xgboost', 'lightgbm',
                'support vector machine', 'svm', 'decision trees', 'naive bayes',
                'k-means', 'hierarchical clustering', 'dbscan', 'pca', 'lda'
            ],
            'deep_learning': [
                'deep learning', 'neural networks', 'artificial neural networks', 'ann',
                'convolutional neural networks', 'cnn', 'recurrent neural networks', 'rnn',
                'lstm', 'gru', 'transformer', 'attention mechanism', 'self-attention',
                'encoder-decoder', 'autoencoder', 'variational autoencoder', 'vae',
                'generative adversarial networks', 'gan', 'discriminator', 'generator',
                'backpropagation', 'gradient descent', 'adam optimizer', 'sgd',
                'batch normalization', 'dropout', 'regularization', 'activation functions',
                'relu', 'sigmoid', 'tanh', 'softmax', 'loss functions', 'cross entropy'
            ],
            'computer_vision': [
                'computer vision', 'cv', 'image processing', 'image recognition',
                'object detection', 'object recognition', 'face recognition', 'face detection',
                'optical character recognition', 'ocr', 'image segmentation', 'semantic segmentation',
                'instance segmentation', 'edge detection', 'feature extraction', 'sift', 'surf',
                'yolo', 'rcnn', 'faster rcnn', 'mask rcnn', 'u-net', 'resnet', 'vgg',
                'inception', 'mobilenet', 'efficientnet', 'opencv', 'pillow', 'skimage',
                'image classification', 'medical imaging', 'satellite imagery', 'video analysis'
            ],
            'nlp': [
                'natural language processing', 'nlp', 'text processing', 'text mining',
                'text analysis', 'sentiment analysis', 'named entity recognition', 'ner',
                'part of speech tagging', 'pos tagging', 'tokenization', 'stemming',
                'lemmatization', 'word embeddings', 'word2vec', 'glove', 'fasttext',
                'tf-idf', 'bag of words', 'n-grams', 'language modeling', 'text classification',
                'document classification', 'topic modeling', 'lda', 'bert', 'gpt', 'transformer',
                'attention', 'seq2seq', 'machine translation', 'text summarization',
                'question answering', 'chatbots', 'dialogue systems', 'speech recognition',
                'text to speech', 'nltk', 'hugging face', 'transformers'
            ],
            'llm': [
                'large language models', 'llm', 'gpt', 'gpt-3', 'gpt-4', 'chatgpt',
                'bert', 'roberta', 'distilbert', 'electra', 'albert', 't5', 'bart',
                'xlnet', 'ernie', 'deberta', 'claude', 'palm', 'llama', 'alpaca',
                'fine-tuning', 'prompt engineering', 'few-shot learning', 'zero-shot learning',
                'in-context learning', 'retrieval augmented generation', 'rag',
                'langchain', 'llamaindex', 'vector databases', 'embeddings', 'semantic search',
                'openai api', 'hugging face', 'transformers library'
            ],
            # ML/DL Frameworks & Libraries
            'tensorflow': [
                'tensorflow', 'tf', 'tensorflow 2', 'tensorflow.js', 'tensorflow lite',
                'tensorflow serving', 'tensorboard', 'keras', 'tf.keras', 'estimator'
            ],
            'pytorch': [
                'pytorch', 'torch', 'torchvision', 'torchaudio', 'torchtext',
                'pytorch lightning', 'fastai', 'ignite'
            ],
            'ml_libraries': [
                'scikit-learn', 'sklearn', 'pandas', 'numpy', 'scipy', 'matplotlib',
                'seaborn', 'plotly', 'xgboost', 'lightgbm', 'catboost', 'optuna',
                'hyperopt', 'mlflow', 'wandb', 'weights and biases', 'tensorboard',
                'jupyter notebook', 'colab', 'kaggle'
            ],
            # Data Science & Analytics
            'data_science': [
                'data science', 'data scientist', 'data analysis', 'data analytics',
                'statistical analysis', 'descriptive statistics', 'inferential statistics',
                'hypothesis testing', 'a/b testing', 'experimental design', 'causal inference',
                'time series analysis', 'forecasting', 'regression analysis', 'correlation',
                'data visualization', 'exploratory data analysis', 'eda', 'data mining',
                'business intelligence', 'bi', 'kpi', 'metrics', 'dashboard'
            ],
            'big_data': [
                'big data', 'hadoop', 'hdfs', 'mapreduce', 'yarn', 'hive', 'pig',
                'spark', 'apache spark', 'pyspark', 'scala spark', 'spark sql',
                'kafka', 'storm', 'flink', 'elasticsearch', 'solr', 'cassandra',
                'hbase', 'mongodb', 'redis', 'memcached', 'etl', 'data pipeline',
                'data warehouse', 'data lake', 'olap', 'oltp'
            ],
            # Databases
            'sql_databases': [
                'sql', 'mysql', 'postgresql', 'postgres', 'sqlite', 'oracle',
                'sql server', 'mariadb', 'db2', 'sybase', 'stored procedures',
                'triggers', 'views', 'indexes', 'normalization', 'acid properties',
                'transactions', 'joins', 'subqueries', 'cte', 'window functions'
            ],
            'nosql_databases': [
                'nosql', 'mongodb', 'cassandra', 'redis', 'elasticsearch', 'couchdb',
                'dynamodb', 'neo4j', 'graph database', 'document database',
                'key-value store', 'column family', 'time series database',
                'influxdb', 'prometheus'
            ],
            # Cloud Computing
            'aws': [
                'aws', 'amazon web services', 'ec2', 's3', 'rds', 'lambda',
                'cloudformation', 'cloudwatch', 'iam', 'vpc', 'route53',
                'elb', 'auto scaling', 'sqs', 'sns', 'api gateway',
                'cognito', 'dynamo db', 'redshift', 'emr', 'sagemaker'
            ],
            'azure': [
                'azure', 'microsoft azure', 'azure vm', 'azure storage', 'azure sql',
                'azure functions', 'azure devops', 'azure ad', 'azure ml',
                'cosmos db', 'service bus', 'azure kubernetes service', 'aks'
            ],
            'gcp': [
                'gcp', 'google cloud platform', 'compute engine', 'cloud storage',
                'bigquery', 'cloud sql', 'cloud functions', 'app engine',
                'kubernetes engine', 'gke', 'dataflow', 'pub/sub', 'firebase'
            ],
            'cloud_general': [
                'cloud computing', 'iaas', 'paas', 'saas', 'serverless', 'microservices',
                'containerization', 'orchestration', 'auto scaling', 'load balancing',
                'cdn', 'content delivery network'
            ],
            # DevOps & Tools
            'docker': [
                'docker', 'containerization', 'containers', 'dockerfile', 'docker compose',
                'docker swarm', 'docker hub', 'container registry'
            ],
            'kubernetes': [
                'kubernetes', 'k8s', 'pods', 'services', 'deployments', 'configmaps',
                'secrets', 'ingress', 'helm', 'kubectl', 'minikube', 'openshift'
            ],
            'ci_cd': [
                'ci/cd', 'continuous integration', 'continuous deployment', 'jenkins',
                'gitlab ci', 'github actions', 'azure devops', 'bamboo', 'teamcity',
                'travis ci', 'circle ci', 'pipeline', 'build automation'
            ],
            'version_control': [
                'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial',
                'version control', 'source control', 'branching', 'merging',
                'pull request', 'merge request', 'code review'
            ],
            'monitoring': [
                'monitoring', 'logging', 'observability', 'prometheus', 'grafana',
                'elk stack', 'elasticsearch', 'logstash', 'kibana', 'splunk',
                'new relic', 'datadog', 'nagios', 'zabbix'
            ],
            # Web Development
            'frontend': [
                'frontend', 'front-end', 'html', 'html5', 'css', 'css3', 'sass',
                'scss', 'less', 'bootstrap', 'tailwind', 'material ui', 'responsive design',
                'mobile first', 'progressive web app', 'pwa', 'single page application', 'spa',
                'webpack', 'vite', 'parcel', 'gulp', 'grunt'
            ],
            'backend': [
                'backend', 'back-end', 'server-side', 'api development', 'rest api',
                'restful services', 'graphql', 'soap', 'microservices', 'monolith',
                'middleware', 'authentication', 'authorization', 'jwt', 'oauth',
                'session management', 'caching', 'rate limiting'
            ],
            'web_frameworks': [
                'express.js', 'koa', 'fastify', 'nest.js', 'django', 'flask',
                'fastapi', 'spring boot', 'asp.net core', 'ruby on rails',
                'laravel', 'symfony', 'codeigniter', 'cakephp'
            ],
            # Mobile Development
            'mobile': [
                'mobile development', 'android', 'ios', 'react native', 'flutter',
                'xamarin', 'ionic', 'cordova', 'phonegap', 'kotlin', 'swift',
                'objective-c', 'java android', 'android studio', 'xcode'
            ],
            # Security
            'cybersecurity': [
                'cybersecurity', 'information security', 'network security', 'web security',
                'application security', 'penetration testing', 'ethical hacking',
                'vulnerability assessment', 'security audit', 'encryption', 'cryptography',
                'ssl', 'tls', 'https', 'firewall', 'ids', 'ips', 'siem',
                'malware analysis', 'forensics', 'incident response', 'risk assessment'
            ],
            # Network & Systems
            'networking': [
                'networking', 'tcp/ip', 'osi model', 'dns', 'dhcp', 'routing',
                'switching', 'vlan', 'vpn', 'load balancer', 'proxy', 'cdn',
                'http', 'https', 'ftp', 'ssh', 'telnet', 'snmp'
            ],
            'systems_admin': [
                'system administration', 'linux', 'unix', 'windows server',
                'bash', 'shell scripting', 'powershell', 'active directory',
                'ldap', 'virtualization', 'vmware', 'hyper-v', 'kvm'
            ],
            # Testing
            'testing': [
                'software testing', 'unit testing', 'integration testing',
                'system testing', 'acceptance testing', 'manual testing',
                'automated testing', 'test automation', 'selenium', 'cypress',
                'jest', 'mocha', 'junit', 'testng', 'pytest', 'cucumber',
                'performance testing', 'load testing', 'stress testing',
                'jmeter', 'gatling', 'locust'
            ],
            # Project Management & Methodologies
            'methodologies': [
                'agile', 'scrum', 'kanban', 'waterfall', 'lean', 'devops',
                'test driven development', 'tdd', 'behavior driven development', 'bdd',
                'pair programming', 'code review', 'sprint planning', 'retrospective',
                'daily standup', 'product owner', 'scrum master'
            ],
            'project_management': [
                'project management', 'pmp', 'jira', 'confluence', 'trello',
                'asana', 'monday.com', 'notion', 'slack', 'microsoft teams',
                'zoom', 'requirements gathering', 'stakeholder management'
            ]
        }

        # Education keywords
        self.education_keywords = ['bachelor', 'master', 'phd', 'diploma', 'certificate', 'degree', 'university', 'college']

    def extract_text_from_pdf(self, file_content):
        """Extract text from PDF file content"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
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
            logger.error(f"Error reading DOCX: {e}")
            return ""

    def extract_text_from_file(self, file_content, filename):
        """Extract text from various file formats"""
        file_extension = Path(filename).suffix.lower()

        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_content)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_content)
        elif file_extension == '.txt':
            try:
                return file_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return file_content.decode('latin-1')
                except:
                    return file_content.decode('utf-8', errors='ignore')
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

    def enhanced_skill_extraction(self, text):
        """Enhanced skill extraction with better matching"""
        text_lower = text.lower()
        found_skills = {}
        all_skills_found = []

        for category, skills in self.skill_keywords.items():
            category_skills = []
            for skill in skills:
                # Use word boundaries for better matching
                skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(skill_pattern, text_lower):
                    category_skills.append(skill)
                    all_skills_found.append(skill)

            if category_skills:
                found_skills[category] = category_skills

        # Remove duplicates while preserving order
        unique_skills = []
        seen = set()
        for skill in all_skills_found:
            if skill not in seen:
                unique_skills.append(skill)
                seen.add(skill)

        return {
            'categorized_skills': found_skills,
            'all_skills': unique_skills,
            'skill_count_by_category': {cat: len(skills) for cat, skills in found_skills.items()},
            'total_skills_found': len(unique_skills)
        }

    def enhanced_experience_extraction(self, text):
        """Enhanced experience extraction with better patterns"""
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

        # Extract date ranges for experience calculation
        date_patterns = [
            r'(\d{4})\s*[-–]\s*(\d{4})',  # 2020-2024
            r'(\d{4})\s*[-–]\s*present',   # 2020-present
            r'(\d{4})\s*[-–]\s*current'    # 2020-current
        ]

        current_year = 2025  # Update as needed
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

        # Combine all experience calculations
        all_experience = years_experience + calculated_years

        # Simple job title extraction without spaCy
        job_titles = [
            'engineer', 'developer', 'programmer', 'analyst', 'scientist', 'manager',
            'lead', 'senior', 'junior', 'intern', 'consultant', 'specialist',
            'architect', 'designer', 'researcher', 'technician', 'administrator'
        ]

        positions = []
        text_lines = text.split('\n')
        for line in text_lines:
            line_lower = line.lower().strip()
            for title in job_titles:
                if title in line_lower and len(line.strip()) < 100:  # Likely a job title
                    positions.append(line.strip())
                    break

        return {
            'years_experience': max(all_experience) if all_experience else 0,
            'all_experience_values': all_experience,
            'positions': list(set(positions))[:5],  # Top 5 unique positions
            'experience_calculation_method': 'enhanced_pattern_matching'
        }

    def enhanced_education_extraction(self, text):
        """Enhanced education extraction with better patterns"""
        text_lower = text.lower()
        education_info = {
            'degrees': [],
            'institutions': [],
            'fields': [],
            'graduation_years': []
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

        # Extract graduation years
        year_pattern = r'(?:graduated|graduation|completed).*?(\d{4})|(\d{4}).*?(?:graduated|graduation)'
        year_matches = re.findall(year_pattern, text_lower)
        for match in year_matches:
            year = match[0] if match[0] else match[1]
            if year and 1990 <= int(year) <= 2030:  # Reasonable year range
                education_info['graduation_years'].append(int(year))

        # Enhanced university/institution extraction
        university_patterns = [
            r'([^,\n]+)\s+university',
            r'([^,\n]+)\s+college',
            r'([^,\n]+)\s+institute',
            r'university\s+of\s+([^,\n]+)',
            r'college\s+of\s+([^,\n]+)'
        ]

        for pattern in university_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if len(match.strip()) > 3:  # Filter out very short matches
                    education_info['institutions'].append(match.strip().title())

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
                    'raw_text': text[:500] if text else "",
                    'parsing_success': False
                }

            # Extract all information using enhanced methods
            contact_info = self.extract_contact_info(text)
            skills_info = self.enhanced_skill_extraction(text)
            experience_info = self.enhanced_experience_extraction(text)
            education_info = self.enhanced_education_extraction(text)

            parsed_resume = {
                'filename': filename,
                'contact_info': contact_info,
                'skills': skills_info,
                'experience': experience_info,
                'education': education_info,
                'raw_text': text[:1000],  # First 1000 characters for preview
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
            logger.error(f"Parsing error for {filename}: {e}")
            return {
                'filename': filename,
                'error': f'Parsing error: {str(e)}',
                'parsing_success': False
            }

class AIScoring:
    def __init__(self):
        # Load pre-trained sentence transformer for semantic similarity
        logger.info("Loading AI models...")
        try:
            # Use a smaller model for better performance in resource-constrained environments
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ AI models loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading AI model: {e}")
            self.sentence_model = None

    def calculate_skills_match(self, resume_skills, job_essential_skills, job_preferred_skills):
        """Calculate skills matching score"""
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        essential_lower = [skill.lower() for skill in job_essential_skills]
        preferred_lower = [skill.lower() for skill in job_preferred_skills]

        # Essential skills match (more important)
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

        # Weighted combination (essential skills are more important)
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
            return 100  # No experience required

        if resume_years >= required_years:
            # Bonus for exceeding requirements, but cap at 100
            experience_score = min(100, 80 + (resume_years - required_years) * 5)
        else:
            # Penalty for not meeting requirements
            experience_score = (resume_years / required_years) * 80

        return round(experience_score, 1)

    def calculate_education_match(self, resume_education, job_education_req):
        """Calculate education matching score"""
        if not job_education_req or job_education_req.lower() == 'none':
            return 100

        resume_degrees = [degree.lower() for degree in resume_education.get('degrees', [])]
        resume_fields = [field.lower() for field in resume_education.get('fields', [])]

        job_education_lower = job_education_req.lower()

        # Check for degree level match
        degree_score = 0
        if any(degree in job_education_lower for degree in resume_degrees):
            degree_score = 80
        elif resume_degrees:  # Has some degree
            degree_score = 60
        else:
            degree_score = 20

        # Check for field relevance
        field_score = 0
        relevant_fields = ['computer', 'engineering', 'science', 'technology', 'business', 'mathematics']

        if resume_fields:
            for field in resume_fields:
                if any(relevant in field for relevant in relevant_fields):
                    field_score = 20
                    break
            else:
                field_score = 10

        education_score = degree_score + field_score
        return min(100, round(education_score, 1))

    def calculate_semantic_similarity(self, resume_text, job_description):
        """Calculate semantic similarity between resume and job description"""
        try:
            if not self.sentence_model:
                return 50  # Default score if model not loaded

            # Truncate texts to avoid memory issues
            resume_text = resume_text[:2000]  # First 2000 characters
            job_description = job_description[:2000]

            # Create embeddings
            resume_embedding = self.sentence_model.encode([resume_text])
            job_embedding = self.sentence_model.encode([job_description])

            # Calculate cosine similarity
            similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]

            # Convert to percentage
            semantic_score = round(similarity * 100, 1)
            return semantic_score

        except Exception as e:
            logger.error(f"Error in semantic similarity calculation: {e}")
            return 50  # Default score

    def calculate_overall_score(self, parsed_resume, job_posting):
        """Calculate comprehensive matching score"""
        try:
            # Extract data
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
                semantic_score * 0.1  # Small weight for semantic similarity
            )

            overall_score = round(overall_score, 1)

            # Create detailed scoring breakdown
            detailed_scores = {
                'overall_score': overall_score,
                'skills_breakdown': skills_match,
                'experience_score': experience_score,
                'education_score': education_score,
                'semantic_score': semantic_score,
                'candidate_name': parsed_resume['filename'],
                'strengths': self.identify_strengths(skills_match, experience_score, education_score),
                'gaps': self.identify_gaps(skills_match, experience_score, education_score, job_requirements)
            }

            return detailed_scores

        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
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

    def identify_gaps(self, skills_match, experience_score, education_score, job_requirements):
        """Identify candidate gaps"""
        gaps = []

        if skills_match['essential_score'] < 50:
            gaps.append("Missing key essential skills")
        if experience_score < 60:
            gaps.append("Below required experience level")
        if education_score < 60:
            gaps.append("Education requirements not fully met")

        return gaps if gaps else ["No significant gaps identified"]

# Initialize instances
resume_parser = ResumeParser()
ai_scorer = AIScoring()

# API Routes
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Smart Resume Screening API',
        'version': '1.0.0',
        'status': 'active',
        'endpoints': {
            'create_job': '/api/create-job',
            'upload_resumes': '/api/upload-resumes',
            'analyze_resumes': '/api/analyze-resumes',
            'get_results': '/api/get-results'
        }
    })

@app.route('/api/create-job', methods=['POST'])
def create_job():
    """Create a new job posting"""
    global current_job
    
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['company_name', 'job_title', 'essential_skills', 'job_description']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Generate unique job ID
        job_id = str(uuid.uuid4())[:8]

        # Create job posting structure
        job_posting = {
            "job_id": job_id,
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "company_info": {
                "company_name": data.get('company_name'),
                "location": data.get('location', '')
            },
            "job_details": {
                "job_title": data.get('job_title'),
                "department": data.get('department', ''),
                "employment_type": data.get('employment_type', 'Full-time'),
                "experience_level": data.get('experience_level', 'Mid Level'),
                "salary_range": data.get('salary_range', '')
            },
            "requirements": {
                "essential_skills": [skill.strip() for skill in data.get('essential_skills', '').split(',') if skill.strip()],
                "preferred_skills": [skill.strip() for skill in data.get('preferred_skills', '').split(',') if skill.strip()],
                "minimum_experience": float(data.get('minimum_experience', 0)),
                "education_requirements": data.get('education_requirements', ''),
                "job_description": data.get('job_description')
            },
            "scoring_weights": {
                "essential_skills": 0.5,
                "preferred_skills": 0.2,
                "experience": 0.2,
                "education": 0.1
            }
        }

        current_job = job_posting

        return jsonify({
            'success': True,
            'message': 'Job posting created successfully',
            'job_id': job_id,
            'job_title': data.get('job_title'),
            'company_name': data.get('company_name')
        })

    except Exception as e:
        logger.error(f"Error creating job: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-resumes', methods=['POST'])
def upload_resumes():
    """Upload and process resume files"""
    global uploaded_resumes
    
    try:
        if not current_job:
            return jsonify({'error': 'Please create a job posting first'}), 400

        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('files')
        uploaded_resumes = []
        results = []

        for file in files:
            if file.filename == '':
                continue

            try:
                # Read file content
                file_content = file.read()
                filename = file.filename

                # Parse resume
                parsed_resume = resume_parser.parse_resume(file_content, filename)

                if parsed_resume.get('parsing_success', False):
                    uploaded_resumes.append(parsed_resume)
                    results.append({
                        'filename': filename,
                        'status': 'success',
                        'message': 'Parsed successfully',
                        'stats': parsed_resume.get('parsing_stats', {})
                    })
                else:
                    results.append({
                        'filename': filename,
                        'status': 'error',
                        'message': parsed_resume.get('error', 'Unknown error')
                    })

            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'message': str(e)
                })

        return jsonify({
            'success': True,
            'message': f'Processed {len(files)} files',
            'total_uploaded': len(files),
            'successfully_parsed': len(uploaded_resumes),
            'results': results
        })

    except Exception as e:
        logger.error(f"Error uploading resumes: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-resumes', methods=['POST'])
def analyze_resumes():
    """Analyze uploaded resumes against job posting"""
    global processed_results
    
    try:
        if not uploaded_resumes:
            return jsonify({'error': 'No resumes to analyze'}), 400

        if not current_job:
            return jsonify({'error': 'No job posting found'}), 400

        scored_candidates = []

        for resume in uploaded_resumes:
            if resume.get('parsing_success', False):
                try:
                    # Calculate AI score
                    score_result = ai_scorer.calculate_overall_score(resume, current_job)
                    scored_candidates.append(score_result)
                except Exception as e:
                    logger.error(f"Error scoring {resume['filename']}: {e}")

        # Sort by overall score (descending)
        scored_candidates.sort(key=lambda x: x.get('overall_score', 0), reverse=True)

        # Store results
        processed_results = scored_candidates

        return jsonify({
            'success': True,
            'message': 'Analysis completed',
            'total_candidates': len(scored_candidates),
            'job_title': current_job['job_details']['job_title'],
            'company_name': current_job['company_info']['company_name'],
            'top_candidates': scored_candidates[:10]  # Return top 10
        })

    except Exception as e:
        logger.error(f"Error analyzing resumes: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-results', methods=['GET'])
def get_results():
    """Get detailed analysis results"""
    try:
        if not processed_results:
            return jsonify({'error': 'No analysis results available'}), 400

        if not current_job:
            return jsonify({'error': 'No job posting found'}), 400

        # Prepare summary statistics
        total_candidates = len(processed_results)
        high_scorers = len([c for c in processed_results if c.get('overall_score', 0) >= 80])
        medium_scorers = len([c for c in processed_results if 60 <= c.get('overall_score', 0) < 80])
        low_scorers = len([c for c in processed_results if c.get('overall_score', 0) < 60])

        return jsonify({
            'success': True,
            'job_info': {
                'job_title': current_job['job_details']['job_title'],
                'company_name': current_job['company_info']['company_name'],
                'job_id': current_job['job_id']
            },
            'summary': {
                'total_candidates': total_candidates,
                'high_scorers': high_scorers,
                'medium_scorers': medium_scorers,
                'low_scorers': low_scorers
            },
            'detailed_results': processed_results,
            'top_10': processed_results[:10]
        })

    except Exception as e:
        logger.error(f"Error getting results: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/candidate/<int:index>', methods=['GET'])
def get_candidate_details(index):
    """Get detailed information for a specific candidate"""
    try:
        if not processed_results or index >= len(processed_results):
            return jsonify({'error': 'Candidate not found'}), 404

        candidate = processed_results[index]
        
        # Find the original resume data
        original_resume = None
        for resume in uploaded_resumes:
            if resume['filename'] == candidate['candidate_name']:
                original_resume = resume
                break

        result = {
            'candidate_info': candidate,
            'resume_details': original_resume,
            'rank': index + 1,
            'total_candidates': len(processed_results)
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error getting candidate details: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/job-info', methods=['GET'])
def get_job_info():
    """Get current job posting information"""
    try:
        if not current_job:
            return jsonify({'error': 'No job posting found'}), 404

        return jsonify({
            'success': True,
            'job_posting': current_job
        })

    except Exception as e:
        logger.error(f"Error getting job info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_system():
    """Reset the system (clear all data)"""
    global current_job, uploaded_resumes, processed_results
    
    try:
        current_job = None
        uploaded_resumes = []
        processed_results = []

        return jsonify({
            'success': True,
            'message': 'System reset successfully'
        })

    except Exception as e:
        logger.error(f"Error resetting system: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ai_model_loaded': ai_scorer.sentence_model is not None,
        'active_job': current_job is not None,
        'uploaded_resumes': len(uploaded_resumes),
        'processed_results': len(processed_results)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
