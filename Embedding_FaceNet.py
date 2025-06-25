"""
Created on Wed Jun 25 16:05:39 2025
Last updated: Wed Jun 25 18:20:00 2025

@author: Furkan Sancak
@version: 1.0
@description: Face recognition with FaceNet embeddings
@dependencies: facenet-pytorch, torch, PIL, MTCNN, InceptionResnetV1
"""

#%% Libs

import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

#%% Face Detection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Face detector
mtcnn = MTCNN(image_size=160, margin=0, device=device)

model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


#%% Load Embeddings

def load_embeddings(directory='.'):
    embeddings = {}
    for file in os.listdir(directory):
        if file.endswith('.pt'):
            name = file[:-3]
            tensor = torch.load(os.path.join(directory, file)).to(device)
            embeddings[name] = tensor
    return embeddings

#%% Get&Save Embedding

def get_embedding(image_path):
    try:
        img = Image.open(image_path)
        face_tensor = mtcnn(img)
        if face_tensor is None:
            print("❌ No face detected.")
            return None
        return model(face_tensor.unsqueeze(0).to(device)).squeeze(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
#%% Recognize

def recognize(embedding, known_embeddings, threshold=0.6):
    best_score = -1
    best_match = None
    for name, emb in known_embeddings.items():
        score = torch.cosine_similarity(embedding, emb, dim=0).item()
        if score > best_score:
            best_score = score
            best_match = name
    return best_match if best_score >= threshold else None, best_score

#%% Main

print("🧠 Face Recognition CLI – type 'exit' to quit, 'ekle <name>' to register.")
known_embeddings = load_embeddings()

while True:
    command = input(">> ").strip()

    if command.lower() == "exit":
        break

    elif command.startswith("ekle "):
        name = command[5:].strip()
        image_path = name + ".jpg"
        emb = get_embedding(image_path)
        if emb is not None:
            torch.save(emb, f"{name}.pt")
            known_embeddings[name] = emb
            print(f"✅ Saved embedding for '{name}'.")

    else:
        image_path = command + ".jpg"
        emb = get_embedding(image_path)
        if emb is not None:
            match_name, score = recognize(emb, known_embeddings)
            if match_name:
                print(f"✅ Tanındı: {match_name} (score={score:.3f})")
            else:
                print(f"❌ Tanınmadı. Skor={score:.3f} (eşik=0.6) — 'ekle {command}' yazarak ekleyebilirsin.")

