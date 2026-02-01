# training_model_a.py
# Model A: Base-scratch (사전 학습 없이 처음부터 학습)

import torch

from google.colab import drive
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import os
from tqdm import tqdm

drive.mount("/content/drive")

# 설정
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 5e-4
MAX_LEN = 128
DATA_PATH			= "training_dataset.csv"
MODEL_SAVE_PATH		= "/content/drive/MyDrive/Model_A"

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# 토크나이저
try:
	tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', es_token='</s>', pad_token='<pad>', unk_token='<unk>')
except:
	print("토크나이저 로드 실패. 인터넷 연결을 확인하세요.")
	exit()

# 데이터셋
class YoutubeCommentDataset(Dataset):
	def __init__(self, csv_file, tokenizer, max_len):
		print("데이터 로드 및 전처리 중...")
		self.df = pd.read_csv(csv_file)
		self.df = self.df[self.df['score'] >= 50].reset_index(drop=True) # 50점 이상만 학습
		#self.df.loc[self.df['score'] < 10, 'score'] = 1.0
		print(f"학습에 사용할 댓글 개수: {len(self.df):,}개")

		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		title = str(row['video_title'])
		comment = str(row['comment'])

		# 형식: 영상제목 [sep] 댓글
		text = f"{title} <unk> {comment} </s>" 

		# 토크나이징
		encodings = self.tokenizer(text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")

		return {
			'input_ids': encodings['input_ids'].squeeze(),
			'attention_mask': encodings['attention_mask'].squeeze(),
			'labels': encodings['input_ids'].squeeze()
		}

# 모델 정의
def create_scratch_model():
	config = GPT2Config(
		vocab_size=tokenizer.vocab_size,
		n_positions=MAX_LEN,
		n_ctx=MAX_LEN,
		n_embd=768,
		n_layer=6,
		n_head=12
	)
	# 가중치 초기화된 모델 생성
	model = GPT2LMHeadModel(config)
	return model

# 학습
def train():
	# 데이터셋 준비
	dataset = YoutubeCommentDataset(DATA_PATH, tokenizer, MAX_LEN)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

	# 모델 생성
	model = create_scratch_model()
	model.to(device)
	model.train()

	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

	print("\n--- 학습 시작 (Model A: Base-scratch) ---")

	for epoch in range(EPOCHS):
		total_loss = 0
		progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

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