# training_model_c.py
# Model C: RoBERTa Regressor

import torch
import pandas as pd
import os
from tqdm import tqdm
from google.colab import drive
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.optim as optim

drive.mount("/content/drive")

# 설정
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
MAX_LEN = 128
RANDOM_SEED = 42
MODEL_NAME			= "klue/roberta-base"
DATA_PATH			= "training_dataset.csv"
MODEL_SAVE_PATH		= "/content/drive/MyDrive/Model_C"

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def get_balanced_dataset(csv_path, max_per_bucket):
	df = pd.read_csv(csv_path)

	# 점수대별 버킷 생성
	df['bucket'] = (df['score'] // 10) * 10

	print("--- 기존 점수대별 데이터 개수 ---")
	print(df['bucket'].value_counts().sort_index())

	balanced_df = df.groupby('bucket').apply(lambda x: x.sample(n=min(len(x), max_per_bucket), random_state=RANDOM_SEED)).reset_index(drop=True)

	print("\n--- 조정 점수대별 데이터 개수 ---")
	print(balanced_df['bucket'].value_counts().sort_index())

	return balanced_df.drop(columns=['bucket'])

# 데이터셋
class YoutubeCommentDataset(Dataset):
	def __init__(self, csv_file, tokenizer, max_len):
		print("데이터 로드 및 전처리 중...")
		self.df = get_balanced_dataset(csv_file, 15000)

		print(f"학습에 사용할 댓글 개수: {len(self.df):,}개")

		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		title = str(row['video_title'])
		score = float(int(row['score'])) / 100.0
		comment = str(row['comment'])

		# 토크나이징
		encoding = self.tokenizer(title, comment, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")

		return {
			'input_ids': encoding['input_ids'].squeeze(),
			'attention_mask': encoding['attention_mask'].squeeze(),
			'labels': torch.tensor(score, dtype=torch.float)
		}

# 학습
def train():
	# 데이터셋 준비
	dataset = YoutubeCommentDataset(DATA_PATH, tokenizer, MAX_LEN)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

	# 모델 생성
	model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
	model.to(device)
	model.train()

	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
	loss_fn = torch.nn.MSELoss()

	if not os.path.exists(MODEL_SAVE_PATH):
		os.makedirs(MODEL_SAVE_PATH)

	print("\n--- 학습 시작 (Model C: RoBERTa Regressor) ---")
	for epoch in range(EPOCHS):
		total_loss = 0

		progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False, dynamic_ncols=True)
		
		for batch in progress_bar:
			optimizer.zero_grad()
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)

			outputs = model(input_ids=input_ids, attention_mask=attention_mask)

			loss = loss_fn(outputs.logits.squeeze(), labels)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

		avg_loss = total_loss / len(dataloader)
		print(f"Epoch {epoch+1} 완료 | 평균 Loss: {avg_loss:.4f}")

		save_path = os.path.join(MODEL_SAVE_PATH, f"epoch_{epoch+1}")
		model.save_pretrained(save_path)
		tokenizer.save_pretrained(save_path)
		print(f"모델 저장 완료: {save_path}")

if __name__ == "__main__":
	if not os.path.exists(DATA_PATH):
		print(f"{DATA_PATH} 파일이 없습니다.")
	else:
		train()