# -*- coding: utf-8 -*-
"""
Created on 05/10/2025

@author: Furkan Sancak

@version: 1.0
@description: Image processor for multiple purposes
"""

#%% Libs

#Common
import torch
from PIL import Image

#Main
import torchvision
from torchvision import transforms
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

#FaceNet
import os
from facenet_pytorch import MTCNN, InceptionResnetV1

#%% Model

#Common
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Main
mainModel = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
mainModel.eval().to(device)

#FaceNet
mtcnn = MTCNN(image_size=160, margin=0, device=device)
facenetModel = InceptionResnetV1(pretrained='vggface2').eval().to(device)

#%% FaceNet Functions

def load_embeddings(directory='.'):
    embeddings = {}
    for file in os.listdir(directory):
        if file.endswith('.pt'):
            name = file[:-3]
            tensor = torch.load(os.path.join(directory, file)).to(device)
            embeddings[name] = tensor
    return embeddings

def recognize(embedding, known_embeddings, threshold=0.6):
    best_score = -1
    best_match = None
    for name, emb in known_embeddings.items():
        score = torch.cosine_similarity(embedding, emb, dim=0).item()
        if score > best_score:
            best_score = score
            best_match = name
    return best_match if best_score >= threshold else None, best_score

#%% Start

#Common
print("\nImage proceccor starting...\n\"Q\" for quitting\nCommand Syntax:[person] [add/find/tell] -imageName -name")

#FaceNet
known_embeddings = load_embeddings()

#%%Döngü

while True:
    command = input("Command Input: ")

    if command == "q":
        print("Quitting...")
        break
    else:
        command = command.split()
        
        #Get Tensor
        img = Image.open(command[2]).convert("RGB")
        transform = transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = mainModel(img_tensor)

        output = outputs[0]
        
        if command[0] == "person":
        
            #Get Persons COCO’da 1
            person_indices = [i for i, label in enumerate(output["labels"]) if label == 1 and output["scores"][i] > 0.8]
            
            persons = []
            for i in person_indices:
                box = output["boxes"][i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                width, height = x2 - x1, y2 - y1
                area = width * height
                persons.append((i, area, box))
        
            if command[1] == "find":
                
                for person in persons:
                    _, _, box = person
                    x1, y1, x2, y2 = box
                    person_img = img.crop((x1, y1, x2, y2))
                    
                    face_tensor = mtcnn(person_img)
                    emb = facenetModel(face_tensor.unsqueeze(0).to(device)).squeeze(0)
                    
                    if emb is not None:
                        match_name, score = recognize(emb, known_embeddings)
                        if match_name:
                            print(f"✅ Tanındı: {match_name} Konum: {x1} {y1}")
                        else:
                            print(f"❌ Tanınmadı. Konum: {x1} {y1}")
                
            elif command[1] == "add":
                
                # Alan büyüklüğüne göre sırala (büyükten küçüğe)
                persons = sorted(persons, key=lambda x: x[1], reverse=True)
                
                _, _, box = persons[0]
                x1, y1, x2, y2 = box
                person_img = img.crop((x1, y1, x2, y2))
                
                face_tensor = mtcnn(person_img)
                emb = facenetModel(face_tensor.unsqueeze(0).to(device)).squeeze(0)
                
                if emb is not None:
                    torch.save(emb, f"{command[3]}.pt")
                    known_embeddings[command[3]] = emb
                    print(f"✅ Saved embedding for '{command[3]}'.")

            elif command[1] == "tell":
                
                # Alan büyüklüğüne göre sırala (büyükten küçüğe)
                persons = sorted(persons, key=lambda x: x[1], reverse=True)
                
                _, _, box = persons[0]
                x1, y1, x2, y2 = box
                person_img = img.crop((x1, y1, x2, y2))
                
                face_tensor = mtcnn(person_img)
                emb = facenetModel(face_tensor.unsqueeze(0).to(device)).squeeze(0)
                
                if emb is not None:
                    match_name, score = recognize(emb, known_embeddings)
                    if match_name:
                        print(f"✅ Tanındı: {match_name} (score={score:.3f})")
                    else:
                        print("❌ Tanınmadı.")
                
            else:
                print("Geçersiz komut")
            
        else:
            print("Geçersiz komut")
