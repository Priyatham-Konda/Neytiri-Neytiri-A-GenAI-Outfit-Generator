import os
from pinecone import Pinecone, ServerlessSpec  # Updated import
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
import torch
import time
from dotenv import find_dotenv, load_dotenv
# Load environment variables from the root .env file
root_env_path = find_dotenv()
load_dotenv(root_env_path)

# Add these imports
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

def load_data():
    """Load data from CSV"""
    try:
        csv_path = "C:/Projects/flipkart-grid-5/backend/dataset/top_100.csv"  # Update path
        df = pd.read_csv(csv_path, encoding='latin1')
        
        # Keep original column names - no mapping needed
        print("Columns in dataset:", df.columns.tolist())
        
        # Fill missing values with 'none'
        df = df.fillna('none')
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def process_images(df):
    """Process images with correct URL column"""
    images = []
    valid_indices = []
    
    for idx, url in enumerate(tqdm(df['style_image'])):  # Using style_image column
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
                valid_indices.append(idx)
            else:
                print(f"Skipping row {idx}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Skipping row {idx}: {str(e)}")
            continue
            
    print(f"Successfully processed {len(valid_indices)}/{len(df)} images")
    
    # Clean dataframe to keep only valid rows
    df_cleaned = df.iloc[valid_indices].reset_index(drop=True)
    return images, df_cleaned

def create_embeddings(df, images, model, bm25):
    """Create dense and sparse embeddings"""
    dense_embeddings = []
    
    # Process text data with BM25
    texts = df['product_display_name'].tolist()
    bm25.fit(texts)
    sparse_vecs = bm25.encode_documents(texts)
    
    # Process images - should match cleaned dataframe
    for img in tqdm(images):
        dense_vec = model.encode(img)
        dense_embeddings.append(dense_vec)
            
    return dense_embeddings, sparse_vecs

def upsert_to_pinecone(pinecone_index, df, dense_vecs, sparse_vecs):
    """Upsert vectors to Pinecone"""
    batch_size = 100
    print("CSV columns:", df.columns.tolist())
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i+batch_size]
        ids = [str(x) for x in batch_df.index]
        
        vectors = []
        for j, row in batch_df.iterrows():
            # Create metadata
            try:
                metadata = {
                    'productDisplayName': str(row['product_display_name']),
                    'brand_name': str(row['brand_name']), 
                    'master_category': str(row['master_category']),
                    'sub_category': str(row['sub_category']),
                    'article_type': str(row['article_type']),
                    'gender': str(row['gender']),
                    'color': str(row['color']),
                    'season': str(row['season']),
                    'usage': str(row['usage']),
                    'fit': str(row['fit']),
                    'pattern': str(row['pattern']),
                    'shape': str(row['shape']),
                    'occasion': str(row['occasion']),
                    'sleeve_styling': str(row['sleeve_styling']),
                    'sleeve_length': str(row['sleeve_length']), 
                    'fabric': str(row['fabric']),
                    'neck': str(row['neck']),
                    'is_jewellery': str(row['is_jewellery']),
                    'link': str(row['style_image'])
                }

                # Create vector entry
                vector = {
                    'id': str(j),
                    'values': dense_vecs[j].tolist(),
                    'sparse_values': sparse_vecs[j],
                    'metadata': metadata
                }
                vectors.append(vector)
            except Exception as e:
                print(f"Error processing row {j}: {e}")
                continue
        if vectors:
            try:   
            # Upsert batch
                pinecone_index.upsert(vectors=vectors)
            except Exception as e:
                print(f"Error upserting batch: {e}")
                continue

def initialize_pinecone():
    try:
        # Create Pinecone client instance
        api_key = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=api_key)
        return pc
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return None

def create_index(pc, index_name):
    try:
        # Check if index exists
        if index_name not in pc.list_indexes().names():
            # Create the index using the new API
            pc.create_index(
                name=index_name,
                dimension=512,
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
        # Get index instance
        return pc.Index(name=index_name)
    except Exception as e:
        print(f"Error creating index: {e}")
        return None

def setup_pinecone():
    start_time = time.time()

    try:
        print('Initializing Pinecone...')
        pc = initialize_pinecone()
        if pc is None:
            raise Exception("Failed to initialize Pinecone client")
        print('Initialization completed.')

        print('Getting CLIP and BM25 model...')
        model, bm25 = get_clip_and_bm25_model()
        print('Models obtained:')
        print('---- Model:', model)
        print('---- BM25:', bm25)

        print('Creating index...')
        index_name = "final-database"
        pinecone_index = create_index(pc, index_name)
        stats = pinecone_index.describe_index_stats()
        if stats.total_vector_count == 0:
            print('Loading and processing data...')
            df = load_data()
            if df is None:
                raise Exception("Failed to load data")
            images, df_cleaned = process_images(df)
            dense_vecs, sparse_vecs = create_embeddings(df, images, model, bm25)
            print('Upserting vectors to Pinecone...')
            upsert_to_pinecone(pinecone_index, df_cleaned, dense_vecs, sparse_vecs)

        if pinecone_index is None:
            raise Exception("Failed to create/get index")
        print('Index created:', pinecone_index)
        print('Setup completed.')
        end_time = time.time()
        print(f'Time taken: {end_time - start_time} seconds')
        return pinecone_index, model, bm25

    except Exception as e:
        print(f"Error setting up Pinecone: {e}")
        return None, None, None

def get_clip_and_bm25_model():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)
        bm25 = BM25Encoder()
        return model, bm25
    except Exception as e:
        print(f"Error getting CLIP and BM25 model: {e}")
