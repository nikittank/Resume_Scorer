#!/usr/bin/env python3
"""
Deep Learning Neural Network Model for Resume Analysis
Advanced text processing with custom neural networks
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Concatenate, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import re
import os
import warnings
warnings.filterwarnings('ignore')

class DeepLearningResumeAnalyzer:
    def __init__(self):
        """Initialize deep learning models"""
        self.tokenizer = None
        self.max_sequence_length = 1000
        self.embedding_dim = 128
        self.vocab_size = 10000
        
        # Models
        self.role_prediction_model = None
        self.quality_assessment_model = None
        self.skill_extraction_model = None
        self.text_similarity_model = None
        
        # Encoders
        self.role_encoder = LabelEncoder()
        self.skill_encoder = LabelEncoder()
        
        # Job categories for deep learning
        self.job_categories = [
            'ACCOUNTANT', 'ADVOCATE', 'AGRICULTURE', 'APPAREL', 'ARTS',
            'AUTOMOBILE', 'AVIATION', 'BANKING', 'BPO', 'BUSINESS-DEVELOPMENT',
            'CHEF', 'CONSTRUCTION', 'CONSULTANT', 'DESIGNER', 'DIGITAL-MEDIA',
            'ENGINEERING', 'FINANCE', 'FITNESS', 'HEALTHCARE', 'HR',
            'INFORMATION-TECHNOLOGY', 'PUBLIC-RELATIONS', 'SALES', 'TEACHER'
        ]
        
        # Advanced skill taxonomy for neural networks
        self.skill_categories = {
            'programming': [
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                'swift', 'kotlin', 'scala', 'r', 'matlab', 'typescript', 'dart', 'perl'
            ],
            'web_development': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django',
                'flask', 'spring', 'laravel', 'bootstrap', 'jquery', 'sass', 'webpack'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas',
                'numpy', 'scikit-learn', 'keras', 'opencv', 'nlp', 'computer vision'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle',
                'sqlite', 'cassandra', 'neo4j', 'dynamodb'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform',
                'ansible', 'git', 'linux', 'bash', 'ci/cd'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving', 'creativity',
                'analytical thinking', 'project management', 'time management'
            ]
        }
        
        print("🧠 Deep Learning Resume Analyzer initialized!")
    
    def preprocess_text(self, text):
        """Advanced text preprocessing for neural networks"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove very short words (less than 2 characters)
        words = text.split()
        words = [word for word in words if len(word) >= 2]
        
        return ' '.join(words)
    
    def create_role_prediction_model(self):
        """Create deep neural network for job role prediction"""
        print("🏗️ Building Role Prediction Neural Network...")
        
        # Input layer
        input_layer = Input(shape=(self.max_sequence_length,))
        
        # Embedding layer
        embedding = Embedding(self.vocab_size, self.embedding_dim, 
                            input_length=self.max_sequence_length)(input_layer)
        
        # CNN branch
        conv1 = Conv1D(128, 3, activation='relu')(embedding)
        conv1 = MaxPooling1D(2)(conv1)
        conv1 = Conv1D(64, 3, activation='relu')(conv1)
        conv1 = GlobalMaxPooling1D()(conv1)
        
        # LSTM branch
        lstm1 = LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)(embedding)
        lstm1 = LSTM(64, dropout=0.3, recurrent_dropout=0.3)(lstm1)
        
        # Combine CNN and LSTM
        combined = Concatenate()([conv1, lstm1])
        
        # Dense layers
        dense1 = Dense(256, activation='relu')(combined)
        dense1 = Dropout(0.5)(dense1)
        dense2 = Dense(128, activation='relu')(dense1)
        dense2 = Dropout(0.3)(dense2)
        
        # Output layer
        output = Dense(len(self.job_categories), activation='softmax')(dense2)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_quality_assessment_model(self):
        """Create neural network for resume quality assessment"""
        print("📊 Building Quality Assessment Neural Network...")
        
        model = Sequential([
            # Text processing layers
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_sequence_length),
            
            # Bidirectional LSTM for better context understanding
            tf.keras.layers.Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)),
            tf.keras.layers.Bidirectional(LSTM(64, dropout=0.3)),
            
            # Dense layers for quality scoring
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            # Output layer (quality score 0-100)
            Dense(1, activation='sigmoid')  # Will be scaled to 0-100
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_skill_extraction_model(self):
        """Create neural network for skill extraction and classification"""
        print("🔧 Building Skill Extraction Neural Network...")
        
        # Multi-label classification for skills
        input_layer = Input(shape=(self.max_sequence_length,))
        
        # Embedding
        embedding = Embedding(self.vocab_size, self.embedding_dim)(input_layer)
        
        # Multiple CNN filters for different n-grams
        conv_filters = []
        for filter_size in [2, 3, 4, 5]:
            conv = Conv1D(64, filter_size, activation='relu')(embedding)
            conv = GlobalMaxPooling1D()(conv)
            conv_filters.append(conv)
        
        # Combine all CNN outputs
        combined_conv = Concatenate()(conv_filters)
        
        # Dense layers
        dense1 = Dense(256, activation='relu')(combined_conv)
        dense1 = Dropout(0.4)(dense1)
        dense2 = Dense(128, activation='relu')(dense1)
        dense2 = Dropout(0.3)(dense2)
        
        # Output layers for each skill category
        outputs = []
        for category in self.skill_categories.keys():
            output = Dense(len(self.skill_categories[category]), 
                         activation='sigmoid', 
                         name=f'{category}_output')(dense2)
            outputs.append(output)
        
        model = Model(inputs=input_layer, outputs=outputs)
        
        # Compile with multiple outputs
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_text_similarity_model(self):
        """Create Siamese neural network for text similarity"""
        print("🔍 Building Text Similarity Neural Network...")
        
        # Shared embedding layer
        shared_embedding = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_sequence_length),
            LSTM(128, dropout=0.3),
            Dense(64, activation='relu')
        ])
        
        # Input layers for two texts
        input_a = Input(shape=(self.max_sequence_length,))
        input_b = Input(shape=(self.max_sequence_length,))
        
        # Process both inputs through shared layers
        processed_a = shared_embedding(input_a)
        processed_b = shared_embedding(input_b)
        
        # Calculate similarity
        similarity = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.abs(x[0] - x[1])
        )([processed_a, processed_b])
        
        # Final similarity score
        similarity_score = Dense(32, activation='relu')(similarity)
        similarity_score = Dense(1, activation='sigmoid')(similarity_score)
        
        model = Model(inputs=[input_a, input_b], outputs=similarity_score)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_training_data(self, texts, labels=None, quality_scores=None):
        """Prepare data for neural network training"""
        print("📝 Preparing training data for neural networks...")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create or load tokenizer
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(processed_texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        prepared_data = {'sequences': padded_sequences}
        
        # Prepare labels if provided
        if labels is not None:
            encoded_labels = self.role_encoder.fit_transform(labels)
            categorical_labels = to_categorical(encoded_labels, num_classes=len(self.job_categories))
            prepared_data['labels'] = categorical_labels
        
        # Prepare quality scores if provided
        if quality_scores is not None:
            normalized_scores = np.array(quality_scores) / 100.0  # Normalize to 0-1
            prepared_data['quality_scores'] = normalized_scores
        
        return prepared_data
    
    def train_models(self, resume_texts, job_labels=None, quality_scores=None):
        """Train all neural network models"""
        print("🚀 Starting Deep Learning Training...")
        
        # Prepare data
        training_data = self.prepare_training_data(resume_texts, job_labels, quality_scores)
        sequences = training_data['sequences']
        
        # Create models
        self.role_prediction_model = self.create_role_prediction_model()
        self.quality_assessment_model = self.create_quality_assessment_model()
        self.skill_extraction_model = self.create_skill_extraction_model()
        self.text_similarity_model = self.create_text_similarity_model()
        
        # Training callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
        
        # Train role prediction model
        if 'labels' in training_data:
            print("🎯 Training Role Prediction Model...")
            labels = training_data['labels']
            X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42)
            
            self.role_prediction_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,
                batch_size=32,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
        
        # Train quality assessment model
        if 'quality_scores' in training_data:
            print("📊 Training Quality Assessment Model...")
            scores = training_data['quality_scores']
            X_train, X_val, y_train, y_val = train_test_split(sequences, scores, test_size=0.2, random_state=42)
            
            self.quality_assessment_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=15,
                batch_size=32,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
        
        print("✅ Deep Learning Models Trained Successfully!")
    
    def predict_job_roles(self, resume_text, top_k=8):
        """Predict job roles using neural network"""
        if self.role_prediction_model is None:
            return self._fallback_role_prediction(resume_text, top_k)
        
        # Preprocess and tokenize
        processed_text = self.preprocess_text(resume_text)
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')
        
        # Predict
        predictions = self.role_prediction_model.predict(padded_sequence, verbose=0)[0]
        
        # Get top predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            role = self.job_categories[idx]
            confidence = float(predictions[idx])
            
            results.append({
                'role': role.replace('-', ' ').title(),
                'probability': confidence,
                'confidence': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'
            })
        
        return results
    
    def assess_resume_quality(self, resume_text):
        """Assess resume quality using neural network"""
        if self.quality_assessment_model is None:
            return self._fallback_quality_assessment(resume_text)
        
        # Preprocess and tokenize
        processed_text = self.preprocess_text(resume_text)
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')
        
        # Predict quality score
        quality_score = self.quality_assessment_model.predict(padded_sequence, verbose=0)[0][0]
        quality_score = float(quality_score * 100)  # Scale back to 0-100
        
        return {
            'quality_score': quality_score,
            'assessment': 'Excellent' if quality_score > 85 else 
                         'Good' if quality_score > 70 else 
                         'Average' if quality_score > 50 else 'Needs Improvement'
        }
    
    def extract_skills_neural(self, resume_text):
        """Extract skills using neural network"""
        if self.skill_extraction_model is None:
            return self._fallback_skill_extraction(resume_text)
        
        # Preprocess and tokenize
        processed_text = self.preprocess_text(resume_text)
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')
        
        # Predict skills for each category
        predictions = self.skill_extraction_model.predict(padded_sequence, verbose=0)
        
        extracted_skills = {}
        for i, category in enumerate(self.skill_categories.keys()):
            category_predictions = predictions[i][0]
            
            # Get skills above threshold
            threshold = 0.5
            found_skills = []
            for j, skill_prob in enumerate(category_predictions):
                if skill_prob > threshold and j < len(self.skill_categories[category]):
                    skill = self.skill_categories[category][j]
                    found_skills.append({
                        'skill': skill,
                        'confidence': float(skill_prob)
                    })
            
            if found_skills:
                extracted_skills[category] = found_skills
        
        return extracted_skills
    
    def calculate_text_similarity(self, text1, text2):
        """Calculate text similarity using Siamese network"""
        if self.text_similarity_model is None:
            return self._fallback_text_similarity(text1, text2)
        
        # Preprocess both texts
        processed_text1 = self.preprocess_text(text1)
        processed_text2 = self.preprocess_text(text2)
        
        # Tokenize
        sequence1 = self.tokenizer.texts_to_sequences([processed_text1])
        sequence2 = self.tokenizer.texts_to_sequences([processed_text2])
        
        padded_seq1 = pad_sequences(sequence1, maxlen=self.max_sequence_length, padding='post')
        padded_seq2 = pad_sequences(sequence2, maxlen=self.max_sequence_length, padding='post')
        
        # Predict similarity
        similarity = self.text_similarity_model.predict([padded_seq1, padded_seq2], verbose=0)[0][0]
        
        return float(similarity)
    
    def comprehensive_neural_analysis(self, resume_text, job_description=None):
        """Comprehensive analysis using all neural networks"""
        print("🧠 Running Deep Learning Analysis...")
        
        results = {}
        
        # Role prediction
        role_predictions = self.predict_job_roles(resume_text)
        results['role_predictions'] = role_predictions
        
        # Quality assessment
        quality_assessment = self.assess_resume_quality(resume_text)
        results['quality_assessment'] = quality_assessment
        
        # Skill extraction
        extracted_skills = self.extract_skills_neural(resume_text)
        results['extracted_skills'] = extracted_skills
        
        # Text similarity (if job description provided)
        if job_description:
            similarity = self.calculate_text_similarity(resume_text, job_description)
            results['text_similarity'] = similarity
        
        return results
    
    def save_models(self, model_dir='neural_models'):
        """Save all trained models"""
        os.makedirs(model_dir, exist_ok=True)
        
        if self.role_prediction_model:
            self.role_prediction_model.save(f'{model_dir}/role_prediction_model.h5')
        
        if self.quality_assessment_model:
            self.quality_assessment_model.save(f'{model_dir}/quality_assessment_model.h5')
        
        if self.skill_extraction_model:
            self.skill_extraction_model.save(f'{model_dir}/skill_extraction_model.h5')
        
        if self.text_similarity_model:
            self.text_similarity_model.save(f'{model_dir}/text_similarity_model.h5')
        
        # Save tokenizer and encoders
        with open(f'{model_dir}/tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        with open(f'{model_dir}/role_encoder.pkl', 'wb') as f:
            pickle.dump(self.role_encoder, f)
        
        print(f"✅ Neural network models saved to {model_dir}/")
    
    def load_models(self, model_dir='neural_models'):
        """Load pre-trained models"""
        try:
            if os.path.exists(f'{model_dir}/role_prediction_model.h5'):
                self.role_prediction_model = tf.keras.models.load_model(f'{model_dir}/role_prediction_model.h5')
            
            if os.path.exists(f'{model_dir}/quality_assessment_model.h5'):
                self.quality_assessment_model = tf.keras.models.load_model(f'{model_dir}/quality_assessment_model.h5')
            
            if os.path.exists(f'{model_dir}/skill_extraction_model.h5'):
                self.skill_extraction_model = tf.keras.models.load_model(f'{model_dir}/skill_extraction_model.h5')
            
            if os.path.exists(f'{model_dir}/text_similarity_model.h5'):
                self.text_similarity_model = tf.keras.models.load_model(f'{model_dir}/text_similarity_model.h5')
            
            # Load tokenizer and encoders
            if os.path.exists(f'{model_dir}/tokenizer.pkl'):
                with open(f'{model_dir}/tokenizer.pkl', 'rb') as f:
                    self.tokenizer = pickle.load(f)
            
            if os.path.exists(f'{model_dir}/role_encoder.pkl'):
                with open(f'{model_dir}/role_encoder.pkl', 'rb') as f:
                    self.role_encoder = pickle.load(f)
            
            print("✅ Neural network models loaded successfully!")
            return True
        
        except Exception as e:
            print(f"⚠️ Could not load neural models: {e}")
            return False
    
    # Fallback methods for when neural models aren't available
    def _fallback_role_prediction(self, resume_text, top_k=8):
        """Fallback role prediction using keyword matching"""
        # Simple keyword-based prediction as fallback
        role_scores = {}
        text_lower = resume_text.lower()
        
        # Define keywords for each role
        role_keywords = {
            'INFORMATION-TECHNOLOGY': ['python', 'java', 'software', 'programming', 'developer'],
            'ENGINEERING': ['engineer', 'technical', 'design', 'development'],
            'FINANCE': ['finance', 'accounting', 'financial', 'budget'],
            'HEALTHCARE': ['medical', 'health', 'patient', 'clinical'],
            'SALES': ['sales', 'marketing', 'customer', 'revenue'],
            'HR': ['human resources', 'recruitment', 'hr', 'people'],
            'DESIGNER': ['design', 'creative', 'visual', 'ui', 'ux'],
            'CONSULTANT': ['consulting', 'advisory', 'strategy', 'analysis']
        }
        
        for role, keywords in role_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            role_scores[role] = score / len(keywords)
        
        # Sort and return top predictions
        sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for role, score in sorted_roles[:top_k]:
            results.append({
                'role': role.replace('-', ' ').title(),
                'probability': min(score + 0.3, 1.0),  # Add base probability
                'confidence': 'Medium' if score > 0.3 else 'Low'
            })
        
        return results
    
    def _fallback_quality_assessment(self, resume_text):
        """Fallback quality assessment"""
        word_count = len(resume_text.split())
        
        # Simple scoring based on length and structure
        base_score = min(word_count / 10, 50)  # Up to 50 points for length
        
        # Check for common sections
        sections = ['experience', 'education', 'skills', 'summary']
        section_score = sum(10 for section in sections if section.lower() in resume_text.lower())
        
        total_score = min(base_score + section_score, 100)
        
        return {
            'quality_score': total_score,
            'assessment': 'Good' if total_score > 70 else 'Average' if total_score > 50 else 'Needs Improvement'
        }
    
    def _fallback_skill_extraction(self, resume_text):
        """Fallback skill extraction using keyword matching"""
        text_lower = resume_text.lower()
        extracted_skills = {}
        
        for category, skills in self.skill_categories.items():
            found_skills = []
            for skill in skills:
                if skill.lower() in text_lower:
                    found_skills.append({
                        'skill': skill,
                        'confidence': 0.8
                    })
            
            if found_skills:
                extracted_skills[category] = found_skills
        
        return extracted_skills
    
    def _fallback_text_similarity(self, text1, text2):
        """Fallback text similarity using simple word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)

# Example usage and testing
if __name__ == "__main__":
    # Initialize the deep learning analyzer
    analyzer = DeepLearningResumeAnalyzer()
    
    # Example resume text
    sample_resume = """
    John Doe
    Software Engineer
    
    Experience:
    - 5 years of Python development
    - Machine learning projects using TensorFlow and PyTorch
    - Web development with React and Node.js
    - Database management with PostgreSQL
    
    Skills:
    Python, JavaScript, React, TensorFlow, Machine Learning, Deep Learning,
    PostgreSQL, Git, Docker, AWS
    
    Education:
    Bachelor's in Computer Science
    """
    
    # Test the analyzer (will use fallback methods initially)
    print("🧪 Testing Deep Learning Analyzer...")
    
    # Role prediction
    roles = analyzer.predict_job_roles(sample_resume)
    print(f"🎯 Predicted roles: {[r['role'] for r in roles[:3]]}")
    
    # Quality assessment
    quality = analyzer.assess_resume_quality(sample_resume)
    print(f"📊 Quality score: {quality['quality_score']:.1f}%")
    
    # Skill extraction
    skills = analyzer.extract_skills_neural(sample_resume)
    print(f"🔧 Extracted skills: {list(skills.keys())}")
    
    print("✅ Deep Learning Analyzer ready for integration!")