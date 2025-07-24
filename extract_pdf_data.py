#!/usr/bin/env python3
"""
PDF Text Extraction Script
Extracts text from all PDFs in the data folder for training
"""

import os
import json
import pandas as pd
import PyPDF2
from docx import Document
import re
from tqdm import tqdm
from collections import defaultdict

class PDFTextExtractor:
    """Extract text from all PDFs in data folder"""
    
    def __init__(self):
        self.data_dir = 'data'
        self.output_file = 'extracted_resume_data.json'
        self.csv_output = 'training_data.csv'
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a single PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"Error reading page {page_num} from {pdf_path}: {e}")
                        continue
                
                # Clean the extracted text
                text = self.clean_text(text)
                return text
                
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text):
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\+\#\.\@\(\)\[\]\/\%\$\&]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_all_pdfs(self):
        """Extract text from all PDFs in data folder"""
        print("🔄 Starting PDF text extraction...")
        
        if not os.path.exists(self.data_dir):
            print(f"❌ Data directory '{self.data_dir}' not found!")
            return []
        
        extracted_data = []
        categories = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        
        print(f"📁 Found {len(categories)} job categories")
        
        total_files = 0
        successful_extractions = 0
        
        for category in categories:
            category_path = os.path.join(self.data_dir, category)
            pdf_files = [f for f in os.listdir(category_path) if f.endswith('.pdf')]
            
            print(f"\n📂 Processing {category}: {len(pdf_files)} files")
            
            category_data = []
            
            for pdf_file in tqdm(pdf_files, desc=f"Extracting {category}"):
                pdf_path = os.path.join(category_path, pdf_file)
                total_files += 1
                
                # Extract text
                text = self.extract_text_from_pdf(pdf_path)
                
                if text and len(text) > 100:  # Only keep meaningful extractions
                    resume_data = {
                        'filename': pdf_file,
                        'category': category,
                        'text': text,
                        'word_count': len(text.split()),
                        'char_count': len(text),
                        'file_path': pdf_path
                    }
                    
                    category_data.append(resume_data)
                    extracted_data.append(resume_data)
                    successful_extractions += 1
                else:
                    print(f"⚠️  Failed to extract meaningful text from {pdf_file}")
            
            print(f"   ✅ Successfully extracted {len(category_data)} resumes from {category}")
        
        print(f"\n📊 Extraction Summary:")
        print(f"   Total files processed: {total_files}")
        print(f"   Successful extractions: {successful_extractions}")
        print(f"   Success rate: {(successful_extractions/total_files)*100:.1f}%")
        
        return extracted_data
    
    def save_extracted_data(self, extracted_data):
        """Save extracted data to JSON and CSV files"""
        print(f"\n💾 Saving extracted data...")
        
        # Save to JSON
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Saved JSON data to {self.output_file}")
        
        # Save to CSV for easy analysis
        df = pd.DataFrame(extracted_data)
        df.to_csv(self.csv_output, index=False, encoding='utf-8')
        
        print(f"   ✅ Saved CSV data to {self.csv_output}")
        
        # Print statistics by category
        print(f"\n📈 Data Statistics by Category:")
        category_stats = df.groupby('category').agg({
            'filename': 'count',
            'word_count': ['mean', 'min', 'max'],
            'char_count': ['mean', 'min', 'max']
        }).round(2)
        
        print(category_stats)
        
        return df
    
    def analyze_extracted_data(self, df):
        """Analyze the extracted data"""
        print(f"\n🔍 Data Analysis:")
        print(f"   Total resumes: {len(df)}")
        print(f"   Categories: {df['category'].nunique()}")
        print(f"   Average words per resume: {df['word_count'].mean():.0f}")
        print(f"   Average characters per resume: {df['char_count'].mean():.0f}")
        
        # Find categories with most/least data
        category_counts = df['category'].value_counts()
        print(f"\n📊 Category Distribution:")
        print(f"   Most data: {category_counts.index[0]} ({category_counts.iloc[0]} resumes)")
        print(f"   Least data: {category_counts.index[-1]} ({category_counts.iloc[-1]} resumes)")
        
        # Show sample text from each category
        print(f"\n📝 Sample Text Preview:")
        for category in df['category'].unique()[:5]:  # Show first 5 categories
            sample_text = df[df['category'] == category]['text'].iloc[0][:200]
            print(f"   {category}: {sample_text}...")
    
    def merge_with_existing_csv(self):
        """Merge with existing Resume.csv if it exists"""
        existing_csv = 'Resume.csv'
        
        if os.path.exists(existing_csv):
            print(f"\n🔄 Merging with existing {existing_csv}...")
            
            try:
                existing_df = pd.read_csv(existing_csv)
                new_df = pd.read_csv(self.csv_output)
                
                # Standardize column names
                if 'Resume_str' in existing_df.columns:
                    existing_df = existing_df.rename(columns={'Resume_str': 'text', 'Category': 'category'})
                
                # Combine datasets
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # Remove duplicates based on text similarity (first 100 characters)
                combined_df['text_preview'] = combined_df['text'].str[:100]
                combined_df = combined_df.drop_duplicates(subset=['text_preview'], keep='first')
                combined_df = combined_df.drop('text_preview', axis=1)
                
                # Save combined dataset
                combined_output = 'combined_training_data.csv'
                combined_df.to_csv(combined_output, index=False, encoding='utf-8')
                
                print(f"   ✅ Combined dataset saved to {combined_output}")
                print(f"   📊 Combined dataset: {len(combined_df)} total resumes")
                
                return combined_df
                
            except Exception as e:
                print(f"   ⚠️  Error merging datasets: {e}")
                return pd.read_csv(self.csv_output)
        
        return pd.read_csv(self.csv_output)

def main():
    """Main extraction function"""
    print("🚀 PDF Text Extraction for Resume Training Data")
    print("=" * 60)
    
    extractor = PDFTextExtractor()
    
    # Extract text from all PDFs
    extracted_data = extractor.extract_all_pdfs()
    
    if not extracted_data:
        print("❌ No data extracted. Please check your data folder.")
        return
    
    # Save extracted data
    df = extractor.save_extracted_data(extracted_data)
    
    # Analyze the data
    extractor.analyze_extracted_data(df)
    
    # Merge with existing data if available
    final_df = extractor.merge_with_existing_csv()
    
    print(f"\n🎉 PDF Text Extraction Complete!")
    print(f"📁 Files created:")
    print(f"   • {extractor.output_file} - JSON format")
    print(f"   • {extractor.csv_output} - CSV format")
    print(f"   • combined_training_data.csv - Combined dataset")
    
    print(f"\n🚀 Ready for ML training!")
    print(f"   Total training samples: {len(final_df)}")
    print(f"   Categories: {final_df['category'].nunique()}")

if __name__ == "__main__":
    main()