# 03_analyze_and_build.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# 경로
DATA_PARQUET_PATH		= "../data/data/integrated_data.parquet"
OUTPUT_PATH				= "../data/data/train_dataset.csv"

# w 가중치 설정
W_VIEWS = 0.51
W_SUBS  = 0.48
W_TIME  = 0.01

# 댓글 최대 길이
MAX_LEN = 200
# 좋아요 기준
MAX_LIKES = 1000

# 텍스트 전처리
def clean_text(text):
	if not isinstance(text, str): return ""
	# HTML 태그 제거
	text = re.sub(r'<[^>]+>', '', text)
	# 타임스탬프 제거
	text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', '', text)
	# HTML 엔티티 제거
	text = re.sub(r'&[a-z]+;', '', text)
	# 공백 정리
	text = re.sub(r'\s+', ' ', text).strip()
	return text

def main():
	print("--- [1/4] 데이터 로드 중... ---")

	if not os.path.exists(DATA_PARQUET_PATH):
		print(f"오류: 데이터 파일이 없습니다. ({DATA_PARQUET_PATH})")
		return

	# 통합 데이터 로드
	df = pd.read_parquet(DATA_PARQUET_PATH)

	print(f"--- [2/4] 점수 계산 중 (가중치: {W_VIEWS}/{W_SUBS}/{W_TIME}) ---")

	log_likes = np.log1p(df['likes'])
	log_views = np.log1p(df['video_views'])
	log_subs  = np.log1p(df['subscribers'])
	log_time  = np.log1p(df['time_diff'] + 60)

	# 점수 계산 및 정규화 (0~100)
	metric = log_likes / ((W_VIEWS * log_views) + (W_SUBS * log_subs) + (W_TIME * log_time) + 1e-9)

	min_s = metric.min()
	max_s = metric.max()
	df['score'] = (metric - min_s) / (max_s - min_s) * 100

	# 데이터 분석 및 시각화
	df_top = df[df['likes'] >= MAX_LIKES].copy()

	# 채널별 통계
	stats = df_top.groupby('channel_name')['score'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)
	stats_filtered = stats[stats['count'] >= 5].copy()
	mean_deviation = stats_filtered['mean'].std()

	print("\n" + "="*60)
	print(f"분석 결과 요약")
	print("-" * 60)
	print(f" - 전체 데이터 수 : {len(df):,}개")
	print(f" - 베스트 댓글({MAX_LIKES}+) : {len(df_top):,}개")
	print(f" - 표준편차(공정성) : {mean_deviation:.4f} (낮을수록 좋음, 목표 3.0 이하)")
	print("="*60)

	# TOP 5 출력
	print("\nTOP 5 댓글")
	df_legends = df_top.sort_values(by='score', ascending=False).head(5)
	for i, (_, row) in enumerate(df_legends.iterrows(), 1):
		print(f"{i}. [{row['channel_name']}] {row['comment'][:40]}... (점수: {row['score']:.1f})")

	# 그래프 그리기 (창을 닫아야 다음 단계로 넘어감)
	print("\n그래프를 그립니다. 확인 후 창을 닫아주세요...")
	top_channels = stats_filtered.index.tolist()
	df_plot = df_top[df_top['channel_name'].isin(top_channels)]

	plt.rcParams['font.family'] = 'Malgun Gothic'
	plt.figure(figsize=(12, 6))
	sns.barplot(x='channel_name', y='score', data=df_plot, palette='magma', errorbar=None)
	plt.xticks(rotation=80)
	plt.title(f"좋아요 1000개 이상 채널별 평균 점수 (표준편차: {mean_deviation:.2f})")
	plt.tight_layout()
	plt.show()

	# 저장 여부 확인 및 데이터셋 구축
	print("\n" + "="*60)
	user_input = input(f"위 분석 결과로 학습 데이터를 저장하시겠습니까? (yes/no): ").strip().lower()

	if user_input == 'yes':
		print("\n--- [3/4] 데이터 정제(Cleaning) 시작 ---")

		# 컬럼 추출
		train_df = df[['video_title', 'comment', 'score']].copy()
		train_df = train_df.dropna()

		# 링크 포함 댓글 제거
		initial_len = len(train_df)
		train_df = train_df[~train_df['comment'].str.contains(r'http|https|www', flags=re.IGNORECASE, na=False)]

		# 텍스트 정제
		train_df['comment'] = train_df['comment'].apply(clean_text)

		# 길이 필터링
		mask_len = (train_df['comment'].str.len() >= 2) & (train_df['comment'].str.len() <= MAX_LEN)
		train_df = train_df[mask_len]

		# 저장
		print("--- [4/4] CSV 파일 저장 중... ---")
		train_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

		print(f"저장 완료!")
		print(f" - 경로: {OUTPUT_PATH}")
		print(f" - 최종 데이터 개수: {len(train_df):,}개 (삭제된 노이즈: {initial_len - len(train_df):,}개)")
		
	else:
		print("저장을 취소했습니다. 프로그램을 종료합니다.")

if __name__ == "__main__":
	main()