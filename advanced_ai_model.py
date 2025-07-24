#!/usr/bin/env python3
"""
Advanced AI Resume Analysis Model
Comprehensive system that learns from categorized resume data
"""

import numpy as np
import pandas as pd
import os
import re
import pickle
import json
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import PyPDF2
from docx import Document
import nltk
from textblob import TextBlob
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedResumeAI:
    def __init__(self):
        self.setup_nltk()
        self.initialize_components()
        self.load_or_train_models()
        
    def setup_nltk(self):
        """Setup NLTK components"""
        nltk_downloads = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet', 'vader_lexicon']
        for item in nltk_downloads:
            try:
                if item == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif item in ['stopwords', 'wordnet']:
                    nltk.data.find(f'corpora/{item}')
                elif item == 'averaged_perceptron_tagger':
                    nltk.data.find('taggers/averaged_perceptron_tagger')
                elif item == 'vader_lexicon':
                    nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download(item)
        
        from nltk.corpus import stopwords
        from nltk.sentiment import SentimentIntensityAnalyzer
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def initialize_components(self):
        """Initialize all model components"""
        # Job categories mapping
        self.job_categories = {
            'ACCOUNTANT': 'Finance & Accounting',
            'ADVOCATE': 'Legal',
            'AGRICULTURE': 'Agriculture & Environment',
            'APPAREL': 'Fashion & Retail',
            'ARTS': 'Arts & Creative',
            'AUTOMOBILE': 'Automotive',
            'AVIATION': 'Aviation & Aerospace',
            'BANKING': 'Banking & Finance',
            'BPO': 'Business Process Outsourcing',
            'BUSINESS-DEVELOPMENT': 'Business Development',
            'CHEF': 'Culinary & Hospitality',
            'CONSTRUCTION': 'Construction & Engineering',
            'CONSULTANT': 'Consulting',
            'DESIGNER': 'Design & Creative',
            'DIGITAL-MEDIA': 'Digital Media & Marketing',
            'ENGINEERING': 'Engineering & Technology',
            'FINANCE': 'Finance & Investment',
            'FITNESS': 'Health & Fitness',
            'HEALTHCARE': 'Healthcare & Medical',
            'HR': 'Human Resources',
            'INFORMATION-TECHNOLOGY': 'Information Technology',
            'PUBLIC-RELATIONS': 'Public Relations & Communications',
            'SALES': 'Sales & Marketing',
            'TEACHER': 'Education & Training'
        }
        
        # Comprehensive skill taxonomy
        self.skill_taxonomy = {
            'technical_skills': {
                'programming': {
                    'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift',
                    'kotlin', 'scala', 'r', 'matlab', 'perl', 'shell', 'bash', 'powershell', 'typescript'
                },
                'web_development': {
                    'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
                    'spring', 'laravel', 'bootstrap', 'jquery', 'sass', 'less', 'webpack', 'gulp'
                },
                'databases': {
                    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle',
                    'sqlite', 'cassandra', 'dynamodb', 'neo4j', 'firebase', 'mariadb'
                },
                'cloud_devops': {
                    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
                    'gitlab', 'terraform', 'ansible', 'chef', 'puppet', 'vagrant', 'circleci'
                },
                'data_science': {
                    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 'pandas',
                    'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'jupyter', 'tableau', 'power bi',
                    'spark', 'hadoop', 'airflow', 'mlflow'
                }
            },
            'soft_skills': {
                'leadership', 'communication', 'teamwork', 'problem solving', 'analytical thinking',
                'creative thinking', 'adaptability', 'time management', 'project management',
                'negotiation', 'presentation', 'customer service', 'conflict resolution'
            }
        }
        
        # Initialize models
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.label_encoders = {}
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with better error handling"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    try:
                        text += page.extract_text() + "\n"
                    except:
                        continue
                return text.strip()
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def advanced_text_preprocessing(self, text):
        """Advanced text preprocessing"""
        if not text:
            return ""
        
        text = text.lower()
        text = re.sub(r'\S+@\S+', ' ', text)  # Remove emails
        text = re.sub(r'http\S+|www\S+', ' ', text)  # Remove URLs
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' ', text)  # Remove phone numbers
        text = re.sub(r'[^\w\s\-\+\#\.]', ' ', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        return text
    
    def extract_comprehensive_features(self, text, category=None):
        """Extract comprehensive features from resume text"""
        if not text:
            return {}
        
        features = {}
        text_lower = text.lower()
        
        # Basic text statistics
        words = text.split()
        sentences = TextBlob(text).sentences
        
        features.update({
            'word_count': len(words),
            'char_count': len(text),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': np.mean([len(str(sent).split()) for sent in sentences]) if sentences else 0
        })
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        features.update({
            'sentiment_positive': sentiment['pos'],
            'sentiment_negative': sentiment['neg'],
            'sentiment_neutral': sentiment['neu'],
            'sentiment_compound': sentiment['compound']
        })
        
        # Experience extraction
        experience_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:in|with|as)',
            r'over\s*(\d+)\s*(?:years?|yrs?)',
            r'more than\s*(\d+)\s*(?:years?|yrs?)'
        ]
        
        experience_years = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    experience_years.extend([int(x) for x in match if x.isdigit()])
                else:
                    if match.isdigit():
                        experience_years.append(int(match))
        
        features['experience_years'] = max(experience_years) if experience_years else 0
        
        # Education level detection
        education_levels = {
            'phd': ['phd', 'ph.d', 'doctorate', 'doctoral'],
            'masters': ['masters', 'master', 'mba', 'ms', 'm.s', 'ma', 'm.a', 'mtech', 'm.tech'],
            'bachelors': ['bachelor', 'bachelors', 'bs', 'b.s', 'ba', 'b.a', 'btech', 'b.tech', 'be', 'b.e'],
            'diploma': ['diploma', 'certificate', 'certification']
        }
        
        education_score = 0
        for level, keywords in education_levels.items():
            if any(keyword in text_lower for keyword in keywords):
                if level == 'phd':
                    education_score = max(education_score, 4)
                elif level == 'masters':
                    education_score = max(education_score, 3)
                elif level == 'bachelors':
                    education_score = max(education_score, 2)
                elif level == 'diploma':
                    education_score = max(education_score, 1)
        
        features['education_level'] = education_score
        
        # Skill extraction by category
        skill_features = {}
        for main_category, subcategories in self.skill_taxonomy.items():
            if isinstance(subcategories, dict):
                for sub_name, skills in subcategories.items():
                    found_skills = [skill for skill in skills if skill in text_lower]
                    skill_features[f'{main_category}_{sub_name}_count'] = len(found_skills)
                    skill_features[f'{main_category}_{sub_name}_skills'] = found_skills
            else:
                found_skills = [skill for skill in subcategories if skill in text_lower]
                skill_features[f'{main_category}_count'] = len(found_skills)
                skill_features[f'{main_category}_skills'] = found_skills
        
        features.update(skill_features)
        
        # Project and achievement indicators
        project_keywords = ['project', 'developed', 'built', 'created', 'implemented', 'designed', 
                           'led', 'managed', 'achieved', 'improved', 'increased', 'reduced']
        features['project_score'] = sum(1 for keyword in project_keywords if keyword in text_lower)
        
        # Leadership indicators
        leadership_keywords = ['led', 'managed', 'supervised', 'coordinated', 'directed', 'headed',
                              'team lead', 'project manager', 'senior', 'principal', 'architect']
        features['leadership_score'] = sum(1 for keyword in leadership_keywords if keyword in text_lower)
        
        # Resume structure quality
        section_keywords = ['experience', 'education', 'skills', 'projects', 'achievements', 
                           'certifications', 'awards', 'publications']
        features['structure_score'] = sum(1 for keyword in section_keywords if keyword in text_lower)
        
        # Contact information completeness
        contact_patterns = [
            r'\S+@\S+',  # email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # phone
            r'linkedin',  # linkedin
            r'github'  # github
        ]
        features['contact_completeness'] = sum(1 for pattern in contact_patterns 
                                             if re.search(pattern, text_lower))
        
        return features
    
    def load_training_data(self):
        """Load and process all training data"""
        print("🔄 Loading comprehensive training data...")
        training_data = []
        
        data_dir = 'data'
        if not os.path.exists(data_dir):
            print("❌ Data directory not found!")
            return []
        
        # Process PDF files by category
        for category in os.listdir(data_dir):
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path):
                print(f"📁 Processing {category}...")
                
                pdf_files = [f for f in os.listdir(category_path) if f.endswith('.pdf')]
                
                # Process up to 30 files per category for better training
                for pdf_file in pdf_files[:30]:
                    pdf_path = os.path.join(category_path, pdf_file)
                    text = self.extract_text_from_pdf(pdf_path)
                    
                    if text and len(text) > 100:  # Ensure meaningful content
                        features = self.extract_comprehensive_features(text, category)
                        features['category'] = category
                        features['category_display'] = self.job_categories.get(category, category)
                        features['text'] = text
                        training_data.append(features)
        
        # Load CSV data if available
        csv_path = 'Resume.csv'
        if os.path.exists(csv_path):
            try:
                print("📊 Loading CSV data...")
                # Read more rows for better training
                df = pd.read_csv(csv_path, nrows=2000)
                
                for _, row in df.iterrows():
                    if 'Resume_str' in row and 'Category' in row:
                        text = str(row['Resume_str'])
                        if len(text) > 100:
                            category = str(row['Category']).upper()
                            features = self.extract_comprehensive_features(text, category)
                            features['category'] = category
                            features['category_display'] = self.job_categories.get(category, category)
                            features['text'] = text
                            training_data.append(features)
            except Exception as e:
                print(f"⚠️  Error loading CSV: {e}")
        
        print(f"✅ Loaded {len(training_data)} training samples")
        return training_data
    
    def train_models(self):
        """Train all AI models"""
        print("🚀 Starting comprehensive AI model training...")
        
        # Load training data
        training_data = self.load_training_data()
        
        if not training_data:
            print("❌ No training data available!")
            return False
        
        df = pd.DataFrame(training_data)
        
        # Prepare features
        feature_columns = [
            'word_count', 'experience_years', 'education_level', 'project_score',
            'leadership_score', 'structure_score', 'contact_completeness'
        ]
        
        # Add skill category features
        for main_category, subcategories in self.skill_taxonomy.items():
            if isinstance(subcategories, dict):
                for sub_name in subcategories.keys():
                    feature_columns.append(f'{main_category}_{sub_name}_count')
            else:
                feature_columns.append(f'{main_category}_count')
        
        # Fill missing values
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)
        
        # Train role prediction model
        print("🧠 Training role prediction model...")
        X = df[feature_columns].values
        y = df['category'].values
        
        # Encode labels
        self.label_encoders['role'] = LabelEncoder()
        y_encoded = self.label_encoders['role'].fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        self.scalers['role'] = StandardScaler()
        X_train_scaled = self.scalers['role'].fit_transform(X_train)
        X_test_scaled = self.scalers['role'].transform(X_test)
        
        # Create ensemble classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        svm_classifier = SVC(probability=True, random_state=42)
        
        self.models['role_predictor'] = VotingClassifier([
            ('rf', rf_classifier),
            ('svm', svm_classifier)
        ], voting='soft')
        
        # Train the ensemble
        self.models['role_predictor'].fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.models['role_predictor'].score(X_train_scaled, y_train)
        test_score = self.models['role_predictor'].score(X_test_scaled, y_test)
        
        print(f"   📊 Role Prediction - Train Accuracy: {train_score:.3f}, Test Accuracy: {test_score:.3f}")
        
        # Train quality assessment model
        print("📈 Training quality assessment model...")
        
        # Create quality scores based on multiple factors
        quality_scores = []
        for _, row in df.iterrows():
            score = 0
            
            # Experience factor (0-25 points)
            exp_years = row.get('experience_years', 0)
            score += min(exp_years * 3, 25)
            
            # Education factor (0-20 points)
            edu_level = row.get('education_level', 0)
            score += edu_level * 5
            
            # Skills factor (0-30 points)
            total_skills = 0
            for main_category, subcategories in self.skill_taxonomy.items():
                if isinstance(subcategories, dict):
                    for sub_name in subcategories.keys():
                        total_skills += row.get(f'{main_category}_{sub_name}_count', 0)
                else:
                    total_skills += row.get(f'{main_category}_count', 0)
            score += min(total_skills * 2, 30)
            
            # Structure and presentation (0-15 points)
            score += min(row.get('structure_score', 0) * 2, 15)
            
            # Leadership and projects (0-10 points)
            score += min((row.get('leadership_score', 0) + row.get('project_score', 0)) * 0.5, 10)
            
            # Add some randomness for realistic variation
            score += np.random.normal(0, 3)
            
            # Normalize to 0-100
            quality_scores.append(max(0, min(100, score)))
        
        df['quality_score'] = quality_scores
        
        # Train quality regressor
        y_quality = df['quality_score'].values
        
        X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
            X, y_quality, test_size=0.2, random_state=42
        )
        
        # Scale features for quality model
        self.scalers['quality'] = StandardScaler()
        X_train_q_scaled = self.scalers['quality'].fit_transform(X_train_q)
        X_test_q_scaled = self.scalers['quality'].transform(X_test_q)
        
        # Create ensemble regressor
        rf_regressor = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
        xgb_regressor = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
        
        self.models['quality_assessor'] = VotingRegressor([
            ('gb', rf_regressor),
            ('xgb', xgb_regressor)
        ])
        
        # Train the ensemble
        self.models['quality_assessor'].fit(X_train_q_scaled, y_train_q)
        
        # Evaluate
        train_score_q = self.models['quality_assessor'].score(X_train_q_scaled, y_train_q)
        test_score_q = self.models['quality_assessor'].score(X_test_q_scaled, y_test_q)
        
        print(f"   📊 Quality Assessment - Train R²: {train_score_q:.3f}, Test R²: {test_score_q:.3f}")
        
        # Train TF-IDF for text similarity
        all_texts = [self.advanced_text_preprocessing(text) for text in df['text'] if text]
        self.vectorizers['tfidf'] = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            min_df=2,
            max_df=0.95
        )
        self.vectorizers['tfidf'].fit(all_texts)
        
        # Store feature columns
        self.feature_columns = feature_columns
        
        # Save models
        self.save_models()
        
        print("✅ All models trained successfully!")
        return True
    
    def save_models(self):
        """Save all trained models"""
        model_data = {
            'models': self.models,
            'vectorizers': self.vectorizers,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'skill_taxonomy': self.skill_taxonomy,
            'job_categories': self.job_categories,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, 'advanced_ai_model.pkl')
        print("💾 Advanced AI model saved successfully")
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            model_data = joblib.load('advanced_ai_model.pkl')
            self.models = model_data['models']
            self.vectorizers = model_data['vectorizers']
            self.scalers = model_data['scalers']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            self.skill_taxonomy = model_data.get('skill_taxonomy', self.skill_taxonomy)
            self.job_categories = model_data.get('job_categories', self.job_categories)
            
            print("✅ Advanced AI model loaded successfully")
            return True
        except FileNotFoundError:
            print("⚠️  No pre-trained model found")
            return False
    
    def load_or_train_models(self):
        """Load existing models or train new ones"""
        if not self.load_models():
            self.train_models()
    
    def predict_job_roles(self, resume_text, top_k=5):
        """Predict suitable job roles for a resume"""
        features = self.extract_comprehensive_features(resume_text)
        
        # Prepare feature vector
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0))
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = self.scalers['role'].transform(feature_vector)
        
        # Get prediction probabilities
        probabilities = self.models['role_predictor'].predict_proba(feature_vector_scaled)[0]
        
        # Get top k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            category = self.label_encoders['role'].inverse_transform([idx])[0]
            probability = probabilities[idx]
            display_name = self.job_categories.get(category, category)
            
            predictions.append({
                'role': display_name,
                'category': category,
                'probability': round(probability * 100, 2),
                'confidence': 'High' if probability > 0.6 else 'Medium' if probability > 0.3 else 'Low'
            })
        
        return predictions
    
    def assess_resume_quality(self, resume_text):
        """Comprehensive resume quality assessment"""
        features = self.extract_comprehensive_features(resume_text)
        
        # Prepare feature vector
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0))
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = self.scalers['quality'].transform(feature_vector)
        
        # Predict quality score
        quality_score = self.models['quality_assessor'].predict(feature_vector_scaled)[0]
        quality_score = max(0, min(100, quality_score))
        
        return {
            'quality_score': round(quality_score, 1),
            'features': features
        }
    
    def comprehensive_analysis(self, resume_text, job_description=None):
        """Perform comprehensive resume analysis"""
        print("🔍 Performing comprehensive analysis...")
        
        # Role prediction
        role_predictions = self.predict_job_roles(resume_text)
        
        # Quality assessment
        quality_assessment = self.assess_resume_quality(resume_text)
        
        # Job matching (if job description provided)
        job_matching = None
        if job_description:
            job_matching = self.calculate_job_matching_score(resume_text, job_description)
        
        return {
            'role_predictions': role_predictions,
            'quality_assessment': quality_assessment,
            'job_matching': job_matching,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def calculate_job_matching_score(self, resume_text, job_description):
        """Advanced job matching with multiple algorithms"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Extract features from both texts
        resume_features = self.extract_comprehensive_features(resume_text)
        job_features = self.extract_comprehensive_features(job_description)
        
        # Text similarity using TF-IDF
        resume_clean = self.advanced_text_preprocessing(resume_text)
        job_clean = self.advanced_text_preprocessing(job_description)
        
        if resume_clean and job_clean:
            documents = [resume_clean, job_clean]
            tfidf_matrix = self.vectorizers['tfidf'].transform(documents)
            text_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        else:
            text_similarity = 0.0
        
        # Skill matching analysis
        skill_matches = {}
        skill_gaps = {}
        
        for main_category, subcategories in self.skill_taxonomy.items():
            if isinstance(subcategories, dict):
                for sub_name, skills in subcategories.items():
                    resume_skills = set(resume_features.get(f'{main_category}_{sub_name}_skills', []))
                    job_skills = set(job_features.get(f'{main_category}_{sub_name}_skills', []))
                    
                    if job_skills:
                        common_skills = resume_skills.intersection(job_skills)
                        missing_skills = job_skills - resume_skills
                        
                        category_key = f'{main_category}_{sub_name}'
                        skill_matches[category_key] = list(common_skills)
                        skill_gaps[category_key] = list(missing_skills)
            else:
                resume_skills = set(resume_features.get(f'{main_category}_skills', []))
                job_skills = set(job_features.get(f'{main_category}_skills', []))
                
                if job_skills:
                    common_skills = resume_skills.intersection(job_skills)
                    missing_skills = job_skills - resume_skills
                    
                    skill_matches[main_category] = list(common_skills)
                    skill_gaps[main_category] = list(missing_skills)
        
        # Calculate overall skill match
        total_job_skills = sum(len(skills) for skills in skill_gaps.values()) + sum(len(skills) for skills in skill_matches.values())
        total_matched_skills = sum(len(skills) for skills in skill_matches.values())
        
        skill_match_score = (total_matched_skills / total_job_skills * 100) if total_job_skills > 0 else 0
        
        # Experience matching
        resume_exp = resume_features.get('experience_years', 0)
        job_exp = job_features.get('experience_years', 0)
        
        if job_exp > 0:
            exp_match_score = min(resume_exp / job_exp, 1.2) * 100  # Allow slight over-qualification
        else:
            exp_match_score = 100 if resume_exp > 0 else 50
        
        # Calculate weighted overall score
        overall_score = (
            text_similarity * 30 +
            skill_match_score * 40 +
            exp_match_score * 30
        ) / 100
        
        return {
            'overall_score': round(min(overall_score * 100, 100), 1),
            'text_similarity': round(text_similarity * 100, 1),
            'skill_match': round(skill_match_score, 1),
            'experience_match': round(exp_match_score, 1),
            'skill_matches': skill_matches,
            'skill_gaps': skill_gaps,
            'resume_experience': resume_exp,
            'job_experience': job_exp
        }

# Maintain backward compatibility
class SkillMatcher(AdvancedResumeAI):
    def analyze_match(self, resume_text, job_description):
        """Backward compatible method"""
        result = self.comprehensive_analysis(resume_text, job_description)
        
        # Convert to old format
        job_matching = result.get('job_matching', {})
        quality = result.get('quality_assessment', {})
        
        return {
            'overall_score': job_matching.get('overall_score', 0),
            'resume_quality': quality.get('quality_score', 0),
            'text_similarity': job_matching.get('text_similarity', 0),
            'skill_match': job_matching.get('skill_match', 0),
            'experience_match': job_matching.get('experience_match', 0),
            'skill_matches': job_matching.get('skill_matches', {}),
            'skill_gaps': job_matching.get('skill_gaps', {}),
            'suggestions': ['Improve skills alignment', 'Add more relevant experience', 'Enhance resume structure']
        }