#!/usr/bin/env python3
"""
Neural Network Training Script for SkillMatch Pro
Train custom deep learning models on resume data
"""

import os
import pandas as pd
import numpy as np
from deep_learning_model import DeepLearningResumeAnalyzer
import glob
import PyPDF2
from docx import Document
import random
import time

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {e}")
        return ""

def load_resume_data():
    """Load resume data from various sources"""
    print("📊 Loading resume training data...")
    
    resume_texts = []
    job_labels = []
    quality_scores = []
    
    # Load from data directory (PDF files organized by job category)
    data_dir = "data"
    if os.path.exists(data_dir):
        print(f"📁 Loading from {data_dir} directory...")
        
        for category_dir in os.listdir(data_dir):
            category_path = os.path.join(data_dir, category_dir)
            if os.path.isdir(category_path):
                print(f"  📂 Processing {category_dir}...")
                
                # Get all PDF files in this category
                pdf_files = glob.glob(os.path.join(category_path, "*.pdf"))
                
                for pdf_file in pdf_files[:10]:  # Limit to 10 files per category for demo
                    text = extract_text_from_pdf(pdf_file)
                    if text and len(text.strip()) > 100:  # Only use substantial resumes
                        resume_texts.append(text)
                        job_labels.append(category_dir)
                        # Generate synthetic quality scores based on text length and structure
                        quality_score = min(100, len(text.split()) / 5 + random.randint(20, 40))
                        quality_scores.append(quality_score)
    
    # Load from CSV if available
    csv_files = ["Resume.csv", "combined_training_data.csv", "training_data.csv"]
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"📄 Loading from {csv_file}...")
            try:
                df = pd.read_csv(csv_file)
                
                # Try different column names for resume text
                text_columns = ['Resume', 'resume_text', 'text', 'content', 'Resume_str']
                label_columns = ['Category', 'job_category', 'label', 'role']
                
                text_col = None
                label_col = None
                
                for col in text_columns:
                    if col in df.columns:
                        text_col = col
                        break
                
                for col in label_columns:
                    if col in df.columns:
                        label_col = col
                        break
                
                if text_col and label_col:
                    # Sample data to avoid memory issues
                    sample_size = min(1000, len(df))
                    df_sample = df.sample(n=sample_size, random_state=42)
                    
                    for _, row in df_sample.iterrows():
                        text = str(row[text_col])
                        if len(text.strip()) > 100:
                            resume_texts.append(text)
                            job_labels.append(str(row[label_col]))
                            # Generate quality score
                            quality_score = min(100, len(text.split()) / 5 + random.randint(30, 50))
                            quality_scores.append(quality_score)
                
                print(f"  ✅ Loaded {len(df_sample)} samples from {csv_file}")
                
            except Exception as e:
                print(f"  ⚠️ Error loading {csv_file}: {e}")
    
    # Generate some synthetic data if we don't have enough
    if len(resume_texts) < 50:
        print("🔧 Generating synthetic training data...")
        synthetic_data = generate_synthetic_resumes()
        resume_texts.extend(synthetic_data['texts'])
        job_labels.extend(synthetic_data['labels'])
        quality_scores.extend(synthetic_data['scores'])
    
    print(f"✅ Total training samples: {len(resume_texts)}")
    print(f"📊 Job categories: {len(set(job_labels))}")
    print(f"🎯 Categories: {list(set(job_labels))[:10]}...")  # Show first 10
    
    return resume_texts, job_labels, quality_scores

def generate_synthetic_resumes():
    """Generate synthetic resume data for training"""
    print("🤖 Generating synthetic resume data...")
    
    synthetic_texts = []
    synthetic_labels = []
    synthetic_scores = []
    
    # Templates for different job categories
    templates = {
        'INFORMATION-TECHNOLOGY': [
            "Software Engineer with 5 years experience in Python, Java, and JavaScript. Worked on web applications using React and Node.js. Experience with databases like PostgreSQL and MongoDB. Familiar with cloud platforms AWS and Docker.",
            "Data Scientist with expertise in machine learning, deep learning, and statistical analysis. Proficient in Python, R, TensorFlow, and PyTorch. Experience with data visualization using matplotlib and seaborn.",
            "Full Stack Developer specializing in modern web technologies. Expert in React, Angular, Vue.js, and backend technologies like Express.js and Django. Database experience with MySQL and Redis."
        ],
        'FINANCE': [
            "Financial Analyst with 4 years experience in investment banking. Strong analytical skills in financial modeling, risk assessment, and portfolio management. Proficient in Excel, Bloomberg, and financial software.",
            "Accountant with CPA certification and 6 years experience in corporate accounting. Expert in financial reporting, tax preparation, and audit procedures. Familiar with QuickBooks and SAP.",
            "Investment Manager with proven track record in asset management and client relations. Experience in equity research, bond analysis, and derivatives trading."
        ],
        'HEALTHCARE': [
            "Registered Nurse with 7 years experience in critical care and emergency medicine. Strong patient care skills and experience with electronic health records. Certified in CPR and advanced life support.",
            "Medical Doctor specializing in internal medicine with 10 years clinical experience. Board certified with expertise in diagnosis, treatment, and patient management.",
            "Healthcare Administrator with MBA and 5 years experience in hospital management. Strong leadership skills in operations, budgeting, and quality improvement."
        ],
        'ENGINEERING': [
            "Mechanical Engineer with 6 years experience in product design and manufacturing. Proficient in CAD software like SolidWorks and AutoCAD. Experience in project management and quality control.",
            "Civil Engineer specializing in structural design and construction management. Licensed PE with experience in infrastructure projects and building design.",
            "Electrical Engineer with expertise in power systems and control engineering. Experience with circuit design, PLC programming, and industrial automation."
        ],
        'SALES': [
            "Sales Manager with 8 years experience in B2B sales and team leadership. Proven track record of exceeding sales targets and building client relationships. Expert in CRM systems and sales analytics.",
            "Account Executive with strong communication skills and 5 years experience in software sales. Experience in lead generation, proposal writing, and contract negotiation.",
            "Business Development Representative with expertise in market research and customer acquisition. Strong presentation skills and experience with sales automation tools."
        ]
    }
    
    # Generate synthetic resumes
    for category, resume_templates in templates.items():
        for template in resume_templates:
            # Create variations of each template
            for i in range(3):
                # Add some variation
                variations = [
                    f"Name: John Smith\nEmail: john.smith@email.com\nPhone: (555) 123-4567\n\n{template}",
                    f"Professional Summary:\n{template}\n\nEducation: Bachelor's degree in relevant field\nSkills: Leadership, Communication, Problem Solving",
                    f"{template}\n\nAchievements:\n- Increased efficiency by 25%\n- Led team of 5+ professionals\n- Completed projects on time and under budget"
                ]
                
                synthetic_texts.append(variations[i % len(variations)])
                synthetic_labels.append(category)
                synthetic_scores.append(random.randint(60, 95))
    
    print(f"✅ Generated {len(synthetic_texts)} synthetic resumes")
    
    return {
        'texts': synthetic_texts,
        'labels': synthetic_labels,
        'scores': synthetic_scores
    }

def train_neural_networks():
    """Main training function"""
    print("🧠 Starting Neural Network Training for SkillMatch Pro")
    print("=" * 60)
    
    # Initialize the analyzer
    analyzer = DeepLearningResumeAnalyzer()
    
    # Load training data
    resume_texts, job_labels, quality_scores = load_resume_data()
    
    if len(resume_texts) == 0:
        print("❌ No training data found!")
        return False
    
    print(f"📊 Training on {len(resume_texts)} resume samples")
    print(f"🎯 Job categories: {len(set(job_labels))}")
    
    # Train the models
    try:
        print("\n🚀 Starting model training...")
        start_time = time.time()
        
        analyzer.train_models(resume_texts, job_labels, quality_scores)
        
        training_time = time.time() - start_time
        print(f"⏱️ Training completed in {training_time:.2f} seconds")
        
        # Save the trained models
        print("\n💾 Saving trained models...")
        analyzer.save_models()
        
        # Test the models
        print("\n🧪 Testing trained models...")
        test_resume = """
        John Doe
        Software Engineer
        
        Experience:
        - 5 years Python development
        - Machine learning with TensorFlow
        - Web development with React
        - Database management with PostgreSQL
        
        Skills: Python, JavaScript, React, TensorFlow, PostgreSQL, Git, Docker
        Education: BS Computer Science
        """
        
        # Test role prediction
        roles = analyzer.predict_job_roles(test_resume)
        print(f"🎯 Test role predictions: {[r['role'] for r in roles[:3]]}")
        
        # Test quality assessment
        quality = analyzer.assess_resume_quality(test_resume)
        print(f"📊 Test quality score: {quality['quality_score']:.1f}%")
        
        print("\n✅ Neural network training completed successfully!")
        print("🎉 Models are ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🏆 SkillMatch Pro - Neural Network Training")
    print("Building custom deep learning models for resume analysis")
    print("=" * 60)
    
    # Check if TensorFlow is available
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check for GPU
        if tf.config.list_physical_devices('GPU'):
            print("🚀 GPU acceleration available!")
        else:
            print("💻 Using CPU for training")
        
    except ImportError:
        print("❌ TensorFlow not installed!")
        print("📦 Install with: pip install tensorflow>=2.10.0")
        return
    
    # Start training
    success = train_neural_networks()
    
    if success:
        print("\n🎊 TRAINING COMPLETE!")
        print("🚀 Your neural networks are ready for the hackathon!")
        print("💡 Run 'python app.py' and try the /neural_analysis endpoint")
    else:
        print("\n⚠️ Training had issues, but fallback methods will work")
        print("💡 The system will use traditional ML as backup")

if __name__ == "__main__":
    main()