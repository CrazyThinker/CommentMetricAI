# training_model_b.py
# Model B: Fine-tuned GPT-2

import torch
import pandas as pd
import os
from tqdm.auto import tqdm
from google.colab import drive
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch.optim as optim

drive.mount("/content/drive")

# 설정
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_LEN = 128
MODEL_NAME			= "beomi/kcgpt2"
DATA_PATH			= "training_dataset.csv"
MODEL_SAVE_PATH		= "/content/drive/MyDrive/Model_B"

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저
try:
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, bos_token='</s>', eos_token='</s>', pad_token='<pad>', unk_token='<unk>')
except:
	print("토크나이저 로드 실패. 인터넷 연결을 확인하세요.")
	exit()

# 데이터셋
class YoutubeCommentDataset(Dataset):
	def __init__(self, csv_file, tokenizer, max_len):
		print("데이터 로드 및 전처리 중...")
		self.df = pd.read_csv(csv_file)
		self.df = self.df[self.df['score'] >= 50].reset_index(drop=True) # 50점 이상만 학습
		print(f"학습에 사용할 댓글 개수: {len(self.df):,}개")

		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		title = str(row['video_title'])
		score = int(row['score'])
		comment = str(row['comment'])

		# 형식: 영상제목, 80점, <unk> 댓글
		text = f"{title}, {score}점, <unk> {comment} </s>"

		# 토크나이징
		encodings = self.tokenizer(text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")

		input_ids = encodings['input_ids'].squeeze()
		labels = input_ids.clone()

		labels[labels == self.tokenizer.pad_token_id] = -100

		return {
			'input_ids': input_ids,
			'attention_mask': encodings['attention_mask'].squeeze(),
			'labels': labels
		}

# 학습
def train():
	# 데이터셋 준비
	dataset = YoutubeCommentDataset(DATA_PATH, tokenizer, MAX_LEN)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

	# 모델 생성
	model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
	model.resize_token_embeddings(len(tokenizer))
	model.to(device)
	model.train()

	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

	if not os.path.exists(MODEL_SAVE_PATH):
		os.makedirs(MODEL_SAVE_PATH)

	print("\n--- 학습 시작 (Model B: Fine-tuned GPT-2) ---")

	for epoch in range(EPOCHS):
		total_loss = 0

		progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

		for batch in progress_bar:
			optimizer.zero_grad()
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)

			outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
			loss = outputs.loss
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

		avg_loss = total_loss / len(dataloader)
		print(f"Epoch {epoch+1} 완료 | 평균 Loss: {avg_loss:.4f}")

		save_path = f"{MODEL_SAVE_PATH}_epoch_{epoch + 1}"
		model.save_pretrained(save_path)
		tokenizer.save_pretrained(save_path)
		print(f"모델 저장 완료: {save_path}")

if __name__ == "__main__":
	if not os.path.exists(DATA_PATH):
		print(f"{DATA_PATH} 파일이 없습니다.")
	else:
		train()