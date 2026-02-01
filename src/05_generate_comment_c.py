# generate_model_c.py

import torch
from transformers import PreTrainedTokenizerFast, AutoTokenizer, GPT2LMHeadModel, AutoModelForSequenceClassification

# 모델 경로
MODEL_A_PATH	= "../model/ModelA_epoch"
MODEL_B_PATH	= "../model/ModelB_epoch"
MODEL_C_PATH	= "../model/ModelC_epoch"

# 장치 설정 및 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

try:
	tokenizer_a = PreTrainedTokenizerFast.from_pretrained(MODEL_A_PATH)
	model_a = GPT2LMHeadModel.from_pretrained(MODEL_A_PATH).to(device)
	model_a.eval()

	tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B_PATH)
	model_b = GPT2LMHeadModel.from_pretrained(MODEL_B_PATH).to(device)
	model_b.eval()

	tokenizer_c = AutoTokenizer.from_pretrained(MODEL_C_PATH)
	model_c = AutoModelForSequenceClassification.from_pretrained(MODEL_C_PATH).to(device)
	model_c.eval()

	print("AI 모델 로드 완료! 댓글 생성 준비가 되었습니다.")
except Exception as e:
	print(f"모델 로드 실패: {e}\n폴더 이름과 파일 구성을 확인해주세요.")
	exit()

def generate_commentA(title):
	# 형식: 제목 <unk>
	input_text = f"{title} <unk>"
	encodings = tokenizer_a(input_text, return_tensors='pt').to(device)
	input_len = encodings['input_ids'].shape[1]

	with torch.no_grad():
		output = model_a.generate(
			input_ids=encodings['input_ids'],
			attention_mask=encodings['attention_mask'],
			max_length=64,										# 최대 토큰 길이
			do_sample=True,										# 확률 기반 샘플링 활성화
			top_k=30,											# 상위 K개의 단어 후보군
			top_p=0.85,											# 누적 확률의 합이 p(0~1) 이하인 후보군만 선택
			temperature=1.2,									# 창의성 수치
			no_repeat_ngram_size=3								# n-gram 이상의 반복 문구 억제
		)
	generated_tokens = output[0][input_len:]
	comment = tokenizer_a.decode(generated_tokens, skip_special_tokens=True).strip()

	return comment

def generate_commentB(title):
	# 형식: {title}, {score}점, <unk>
	target_score = 90

	input_text = f"{title}, {target_score}점, <unk>"
	encodings = tokenizer_b(input_text, return_tensors='pt').to(device)
	input_len = encodings['input_ids'].shape[1]

	with torch.no_grad():
		output = model_b.generate(
			input_ids=encodings['input_ids'],
			attention_mask=encodings['attention_mask'],
			max_length=128,									# 최대 토큰 길이
			do_sample=True,									# 확률 기반 샘플링 활성화
			top_k=40,										# 상위 K개의 단어 후보군
			top_p=0.92,										# 누적 확률의 합이 p(0~1) 이하인 후보군만 선택
			temperature=1.1,								# 창의성 수치
			no_repeat_ngram_size=3,							# n-gram 이상의 반복 문구 억제
			eos_token_id=tokenizer_b.eos_token_id			# 문장 종결 토큰 설정
		)
	generated_tokens = output[0][input_len:]
	comment = tokenizer_b.decode(generated_tokens, skip_special_tokens=True).strip()

	return comment

def predict_score_C(title, comment):
	encoding = tokenizer_c(
		title, comment,
		max_length=128, padding="max_length", truncation=True, return_tensors="pt"
	).to(device)

	with torch.no_grad():
		outputs = model_c(**encoding)
		# logit 값은 0~1 사이고 점수는 0~100 사이
		score = outputs.logits.squeeze().item() * 100
	return round(score, 2)

print("\n" + "="*50)
print("댓글 생성기 (종료하려면 'exit' 입력)   ")
print("="*50)

while True:
	print("\n영상 제목을 입력하세요: ", end="")
	title = input().strip()

	if title.lower() == "exit":
		print("프로그램을 종료합니다.")
		break

	if not title:
		continue

	print("댓글을 입력하세요(generate_a, generate_b 입력 시 해당 모델으로 랜덤 댓글 생성): ", end="")
	comment = input().strip()

	if comment.lower() == 'generate_a':
		comment = generate_commentA(title)

	if comment.lower() == 'generate_b':
		comment = generate_commentB(title)

	print("\n[AI 댓글 평가 중...]")
	score = predict_score_C(title, comment)

	print("-" * 30)
	print(f"영상 제목: {title}")
	print(f"AI 생성 댓글: {comment}")
	print(f"판독기 예상 점수: {score}점")
	print("-" * 30)