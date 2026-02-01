# generate_model_b.py

import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# 모델 경로
MODEL_PATH			= "../model/ModelB_epoch"

# 장치 설정 및 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

try:
	tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
	model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
	model.eval()

	print("AI 모델 로드 완료! 댓글 생성 준비가 되었습니다.")
except Exception as e:
	print(f"모델 로드 실패: {e}\n폴더 이름과 파일 구성을 확인해주세요.")
	exit()

def generate_commentB(title):
	# 형식: {title}, {score}점, <unk>
	target_score = 90

	input_text = f"{title}, {target_score}점, <unk>"
	encodings = tokenizer(input_text, return_tensors='pt').to(device)
	input_len = encodings['input_ids'].shape[1]

	with torch.no_grad():
		output = model.generate(
			input_ids=encodings['input_ids'],
			attention_mask=encodings['attention_mask'],
			max_length=128,									# 최대 토큰 길이
			do_sample=True,									# 확률 기반 샘플링 활성화
			top_k=40,										# 상위 K개의 단어 후보군
			top_p=0.92,										# 누적 확률의 합이 p(0~1) 이하인 후보군만 선택
			temperature=1.1,								# 창의성 수치
			no_repeat_ngram_size=3,							# n-gram 이상의 반복 문구 억제
			eos_token_id=tokenizer.eos_token_id				# 문장 종결 토큰 설정
		)
	generated_tokens = output[0][input_len:]
	comment = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

	return comment

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

	print("\n[AI 댓글 생성 중...]")
	result = generate_commentB(title)
	
	print("-" * 30)
	print(f"추천 베댓: {result}")
	print("-" * 30)