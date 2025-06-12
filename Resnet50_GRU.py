#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/KiHwanLee123/Samsung_ImageCaptioning/blob/main/Resnet50_GRU.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import random
from tqdm import tqdm
import warnings

warnings.filterwarnings(action='ignore')

# In[ ]:


import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import random
import warnings
warnings.filterwarnings(action='ignore')

# In[ ]:


from google.colab import drive
import zipfile
import os

drive.mount('/content/drive')

# # 하이퍼마라미터 설정
# # Random Seed 고정

# In[ ]:


# 하이퍼파라미터
CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 3,
    'LR': 0.001,
    'BATCH_SIZE': 16,
    'SEED': 41
}

# Random Seed 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])

# # Custom Dataset 정의

# In[ ]:



class CustomDataset(Dataset):
    def __init__(self, dataframe, word2idx=None, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.word2idx = word2idx

    def __len__(self):
        return len(self.dataframe)

    def _tokenize(self, text):
        tokens = ['<SOS>'] + text.split() + ['<EOS>']
        if self.word2idx is None:
            return tokens
        return torch.tensor([self.word2idx.get(t, self.word2idx['<PAD>']) for t in tokens], dtype=torch.long)

    def __getitem__(self, idx):
        img_path = os.path.join('/content/drive/MyDrive/dacon/Samsung_Image Quality Assessment/dataset/train', self.dataframe.iloc[idx]['img_path'][2:])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # mos column 존재 여부에 따라 값을 설정
        mos = float(self.dataframe.iloc[idx]['mos']) if 'mos' in self.dataframe.columns else 0.0
        comment = self.dataframe.iloc[idx]['comments'] if 'comments' in self.dataframe.columns else ""
        if self.word2idx is not None:
            comment = self._tokenize(comment)

        return img, mos, comment



# # 모델 정의

# In[ ]:


import torch.nn as nn
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super(BaseModel, self).__init__()

        # Image feature extraction using ResNet50
        self.cnn_backbone = models.resnet50(pretrained=True)
        # Remove the last fully connected layer to get features
        modules = list(self.cnn_backbone.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        # Image quality assessment head
        self.regression_head = nn.Linear(2048, 1)  # ResNet50 last layer has 2048 features

        # Captioning head
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim + 2048, hidden_dim)  # Replace LSTM with GRU
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions=None):
        # CNN
        features = self.cnn(images)
        features_flat = features.view(features.size(0), -1)

        # Image quality regression
        mos = self.regression_head(features_flat)

        # GRU captioning
        if captions is not None:
            embeddings = self.embedding(captions)
            # Concatenate image features and embeddings for each word in the captions
            combined = torch.cat([features_flat.unsqueeze(1).repeat(1, embeddings.size(1), 1), embeddings], dim=2)
            gru_out, _ = self.gru(combined)
            outputs = self.fc(gru_out)
            return mos, outputs
        else:
            return mos, None


# 

# # Train

# In[ ]:


# 압축 해제할 zip 파일의 경로
zip_file_path = '/content/drive/MyDrive/dacon/Samsung_Image Quality Assessment/dataset/open.zip'

# 대상 폴더명
target_folder_name = 'train'  # 압축 해제할 대상 폴더 이름

# 압축 해제할 디렉토리의 경로
extracted_folder_path = '/content/drive/MyDrive/dacon/Samsung_Image Quality Assessment/dataset'

# Zip 파일 열기
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Zip 파일 내의 파일 및 폴더 목록 확인
    file_list = zip_ref.namelist()

    # 대상 폴더와 이름이 일치하는 폴더를 찾아서 압축 해제
    for file_name in file_list:
        file_name_parts = file_name.split('/')
        if file_name_parts[0] == target_folder_name and len(file_name_parts) == 2:
            zip_ref.extract(file_name, os.path.join(extracted_folder_path, target_folder_name))

print(f"'{target_folder_name}' 폴더를 '{extracted_folder_path}'로 성공적으로 압축 해제했습니다.")


# In[ ]:


# 압축 해제할 zip 파일의 경로
zip_file_path = '/content/drive/MyDrive/dacon/Samsung_Image Quality Assessment/dataset/open.zip'

# 대상 폴더명
target_folder_name = 'test'  # 압축 해제할 대상 폴더 이름

# 압축 해제할 디렉토리의 경로
extracted_folder_path = '/content/drive/MyDrive/dacon/Samsung_Image Quality Assessment/dataset'

# Zip 파일 열기
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Zip 파일 내의 파일 및 폴더 목록 확인
    file_list = zip_ref.namelist()

    # 대상 폴더와 이름이 일치하는 폴더를 찾아서 압축 해제
    for file_name in file_list:
        file_name_parts = file_name.split('/')
        if file_name_parts[0] == target_folder_name and len(file_name_parts) == 2:
            zip_ref.extract(file_name, os.path.join(extracted_folder_path, target_folder_name))

print(f"'{target_folder_name}' 폴더를 '{extracted_folder_path}'로 성공적으로 압축 해제했습니다.")


# In[ ]:


folder_path = '/content/drive/MyDrive/dacon/Samsung_Image Quality Assessment/dataset/test/test'

# 폴더 내의 파일 목록을 가져옴
file_list = os.listdir(folder_path)

len(file_list)

# In[ ]:


# 데이터 로드
train_data = pd.read_csv('/content/drive/MyDrive/dacon/Samsung_Image Quality Assessment/dataset/train.csv')

# In[ ]:




# 단어 사전 생성
all_comments = ' '.join(train_data['comments']).split()
vocab = set(all_comments)
vocab = ['<PAD>', '<SOS>', '<EOS>'] + list(vocab)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# 데이터셋 및 DataLoader 생성
transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor()
])
train_dataset = CustomDataset(train_data, word2idx=word2idx, transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=CFG['BATCH_SIZE'],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

# 모델, 손실함수, 옵티마이저
model = BaseModel(len(vocab)).cuda()
criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LR'])

# 학습
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()

model.train()
for epoch in range(CFG["EPOCHS"]):
    total_loss = 0
    loop = tqdm(train_loader, leave=True)
    for imgs, mos, comments in loop:
        imgs = imgs.float().cuda(non_blocking=True)
        mos = mos.float().cuda(non_blocking=True)
        comments = comments.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            predicted_mos, predicted_comments = model(imgs, comments)
            loss1 = criterion1(predicted_mos.squeeze(1), mos)
            loss2 = criterion2(predicted_comments.view(-1, len(vocab)), comments.view(-1))
            loss = loss1 + loss2

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1} finished with average loss: {total_loss / len(train_loader):.4f}")

# In[ ]:



# 모델 상태 및 가중치 저장
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'word2idx': word2idx,
    'idx2word': idx2word
}, '/content/drive/MyDrive/dacon/Samsung_Image Quality Assessment/submission/model1.pth')


# In[ ]:


torch.save(model.state_dict(), '/content/drive/MyDrive/dacon/Samsung_Image Quality Assessment/submission/model1.pth')

# # 추론 및 결과 저장

# In[ ]:


# 추론을 위한 데이터셋 클래스 정의 (테스트 데이터셋 예시)
class TestDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transform = transforms.Compose([
            transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join('/content/drive/MyDrive/dacon/Samsung_Image Quality Assessment/dataset/test', self.dataframe.iloc[idx]['img_path'][2:])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)  # 이미지를 텐서로 변환
        return img

# In[ ]:


# 테스트 데이터 로드
test_data = pd.read_csv('/content/drive/MyDrive/dacon/Samsung_Image Quality Assessment/dataset/test.csv')

# In[ ]:


test_data = pd.read_csv('test.csv')
test_dataset = CustomDataset(test_data, word2idx=word2idx, transform=transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

model.eval()
predicted_mos_list = []
predicted_comments_list = []

def greedy_decode(model, image, max_length=50):
    image = image.unsqueeze(0).cuda()
    features = model.cnn(image)
    features_flat = features.view(features.size(0), -1)
    mos = model.regression_head(features_flat)
    output_sentence = []

    current_token = torch.tensor([word2idx['<SOS>']]).cuda()
    hidden = None

    for _ in range(max_length):
        embeddings = model.embedding(current_token).unsqueeze(0)
        combined = torch.cat([features_flat.unsqueeze(1), embeddings], dim=2)
        out, hidden = model.gru(combined, hidden)

        output = model.fc(out.squeeze(0))
        _, current_token = torch.max(output, dim=1)

        # <EOS> 토큰에 도달하면 멈춤
        if current_token.item() == word2idx['<EOS>']:
            break

        # <SOS> 또는 <PAD> 토큰은 생성한 캡션에 추가하지 않음
        if current_token.item() not in [word2idx['<SOS>'], word2idx['<PAD>']]:
            output_sentence.append(idx2word[current_token.item()])

    return mos.item(), ' '.join(output_sentence)

# 추론 과정
with torch.no_grad():
    for imgs, _, _ in tqdm(test_loader):
        for img in imgs:
            img = img.float().cuda()
            mos, caption = greedy_decode(model, img)
            predicted_mos_list.append(mos)
            predicted_comments_list.append(caption)

# 결과 저장
result_df = pd.DataFrame({
    'img_name': test_data['img_name'],
    'mos': predicted_mos_list,
    'comments': predicted_comments_list  # 캡션 부분은 위에서 생성한 것을 사용
})

# 예측 결과에 NaN이 있다면, 제출 시 오류가 발생하므로 후처리 진행 (sample_submission.csv과 동일하게)
result_df['comments'] = result_df['comments'].fillna('Nice Image.')
result_df.to_csv('submit.csv', index=False)

print("Inference completed and results saved to submit.csv.")

# In[ ]:



